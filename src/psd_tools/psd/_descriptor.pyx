# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Compiled descriptor reader.

Reads a descriptor body from bytes into the objects
``descriptor._DescriptorMixin._read_body`` reads, or gives up (returns None)
on anything else, leaving the Python reader to read it and raise what it
raises. Elements are created the way their attrs ``__init__`` would leave
them, without the per-read budget calls; the counts come back to the caller
to charge at once.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.float cimport PyFloat_FromDouble
from cpython.long cimport PyLong_FromLongLong
from cpython.object cimport PyObject_GenericSetAttr
from cpython.unicode cimport PyUnicode_Decode
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy

cdef object _new = object.__new__
cdef dict _kinds = {}
cdef dict _units = {}
cdef object _terms = frozenset()
cdef object _ZeroLengthKey = None
cdef object _OrderedDict = None

cdef enum:
    K_DESCRIPTOR
    K_OBJECT_ARRAY
    K_LIST
    K_PROPERTY
    K_UNIT_FLOAT
    K_UNIT_FLOATS
    K_DOUBLE
    K_STRING
    K_ENUMERATED_REFERENCE
    K_OFFSET
    K_BOOL
    K_LARGE_INTEGER
    K_INTEGER
    K_ENUMERATED
    K_RAW_DATA
    K_CLASS
    K_NAME

KINDS = {
    "descriptor": K_DESCRIPTOR,
    "object_array": K_OBJECT_ARRAY,
    "list": K_LIST,
    "property": K_PROPERTY,
    "unit_float": K_UNIT_FLOAT,
    "unit_floats": K_UNIT_FLOATS,
    "double": K_DOUBLE,
    "string": K_STRING,
    "enumerated_reference": K_ENUMERATED_REFERENCE,
    "offset": K_OFFSET,
    "bool": K_BOOL,
    "large_integer": K_LARGE_INTEGER,
    "integer": K_INTEGER,
    "enumerated": K_ENUMERATED,
    "raw_data": K_RAW_DATA,
    "class": K_CLASS,
    "name": K_NAME,
}


def setup(kinds, units, terms, ZeroLengthKey, OrderedDict):
    """Set the OSType table (``{bytes: (class, kind)}``), units and terms."""
    global _kinds, _units, _terms, _ZeroLengthKey, _OrderedDict
    _kinds, _units, _terms = dict(kinds), dict(units), terms
    _ZeroLengthKey, _OrderedDict = ZeroLengthKey, OrderedDict


class _GiveUp(Exception):
    pass


cdef struct State:
    const unsigned char *b
    Py_ssize_t n
    Py_ssize_t p
    long long nbytes
    long long nobjects
    long long maxread
    long depth
    long cap


cdef inline int need(State *st, long long size) except -1:
    """``read_fmt``/``read_exact``: give up on a short read; charge it."""
    if size < 0 or size > st.n - st.p:
        raise _GiveUp
    st.nbytes += size
    if size > st.maxread:
        st.maxread = size
    return 0


cdef inline uint32_t be32(const unsigned char *b) noexcept:
    return (<uint32_t>b[0] << 24) | (<uint32_t>b[1] << 16) | (<uint32_t>b[2] << 8) | b[3]


cdef inline uint64_t be64(const unsigned char *b) noexcept:
    return (<uint64_t>be32(b) << 32) | be32(b + 4)


cdef inline double be_double(const unsigned char *b) noexcept:
    cdef uint64_t bits = be64(b)
    cdef double value
    memcpy(&value, &bits, 8)
    return value


cdef inline long long read_u32(State *st) except -1:
    """``read_fmt("I", fp)[0]``."""
    need(st, 4)
    st.nobjects += 1
    cdef uint32_t value = be32(st.b + st.p)
    st.p += 4
    return value


cdef object read_bytes(State *st, long long size):
    """``read_exact(fp, size)``."""
    need(st, size)
    value = PyBytes_FromStringAndSize(<const char *>(st.b + st.p), size)
    st.p += size
    return value


cdef object read_unicode(State *st):
    """``read_unicode_string(fp)`` with its padding of 1."""
    cdef long long size = read_u32(st) * 2
    need(st, size)
    value = PyUnicode_Decode(<const char *>(st.b + st.p), size, "utf-16-be", NULL)
    st.p += size
    return value


cdef object read_key(State *st, list unknown):
    """``read_length_and_key(fp)``; an unknown term is logged by the caller."""
    cdef long long length = read_u32(st)
    key = read_bytes(st, length if length else 4)
    if length == 0 and key not in _terms:
        key = _ZeroLengthKey(key)
        unknown.append(key)
    return key


cdef inline int enter(State *st) except -1:
    """``parse_container()``: one more level, if the cap allows it."""
    st.depth += 1
    if st.depth > st.cap:
        raise _GiveUp
    return 0


cdef tuple read_body(State *st, list unknown):
    """``_DescriptorMixin._read_body(fp)``."""
    cdef long long count, i
    enter(st)
    name = read_unicode(st)
    classID = read_key(st, unknown)
    count = read_u32(st)
    items = []
    for i in range(count):
        key = read_key(st, unknown)
        items.append((key, read_value(st, unknown)))
    st.depth -= 1
    return name, classID, items


cdef inline void put(object obj, str name, object value) except *:
    PyObject_GenericSetAttr(obj, name, value)


cdef object read_value(State *st, list unknown):
    """``TYPES[OSType(read_exact(fp, 4))].read(fp)``."""
    cdef long long count, i
    cdef int kind
    cdef double number
    entry = _kinds.get(read_bytes(st, 4))
    if entry is None:
        raise _GiveUp
    cls, kind = entry
    obj = _new(cls)
    st.nobjects += 1
    if kind == K_DESCRIPTOR:
        name, classID, items = read_body(st, unknown)
        put(obj, "_items", _OrderedDict(items))
        put(obj, "name", name)
        put(obj, "classID", classID)
    elif kind == K_OBJECT_ARRAY:
        items_count = read_u32(st)
        name, classID, items = read_body(st, unknown)
        put(obj, "_items", _OrderedDict(items))
        put(obj, "items_count", items_count)
        put(obj, "name", name)
        put(obj, "classID", classID)
    elif kind == K_LIST:
        enter(st)
        count = read_u32(st)
        values = []
        for i in range(count):
            values.append(read_value(st, unknown))
        st.depth -= 1
        put(obj, "_items", values)
    elif kind == K_PROPERTY:
        put(obj, "name", read_unicode(st))
        put(obj, "classID", read_key(st, unknown))
        put(obj, "keyID", read_key(st, unknown))
    elif kind == K_UNIT_FLOAT:
        need(st, 12)
        st.nobjects += 2
        unit = _units.get(PyBytes_FromStringAndSize(<const char *>(st.b + st.p), 4))
        if unit is None:
            raise _GiveUp
        number = be_double(st.b + st.p + 4)
        st.p += 12
        put(obj, "value", PyFloat_FromDouble(number))
        put(obj, "unit", unit)
    elif kind == K_UNIT_FLOATS:
        need(st, 8)
        st.nobjects += 2
        unit = _units.get(PyBytes_FromStringAndSize(<const char *>(st.b + st.p), 4))
        if unit is None:
            raise _GiveUp
        count = be32(st.b + st.p + 4)
        st.p += 8
        need(st, count * 8)
        st.nobjects += count
        values = []
        for i in range(count):
            values.append(PyFloat_FromDouble(be_double(st.b + st.p)))
            st.p += 8
        put(obj, "unit", unit)
        put(obj, "values", values)
    elif kind == K_DOUBLE:
        need(st, 8)
        st.nobjects += 1
        number = be_double(st.b + st.p)
        st.p += 8
        put(obj, "value", PyFloat_FromDouble(number))
    elif kind == K_STRING:
        put(obj, "value", read_unicode(st))
    elif kind == K_ENUMERATED_REFERENCE:
        put(obj, "name", read_unicode(st))
        put(obj, "classID", read_key(st, unknown))
        put(obj, "typeID", read_key(st, unknown))
        put(obj, "enum", read_key(st, unknown))
    elif kind == K_OFFSET:
        put(obj, "name", read_unicode(st))
        put(obj, "classID", read_key(st, unknown))
        put(obj, "value", read_u32(st))
    elif kind == K_BOOL:
        need(st, 1)
        st.nobjects += 1
        put(obj, "value", st.b[st.p] != 0)
        st.p += 1
    elif kind == K_LARGE_INTEGER:
        need(st, 8)
        st.nobjects += 1
        put(obj, "value", PyLong_FromLongLong(<int64_t>be64(st.b + st.p)))
        st.p += 8
    elif kind == K_INTEGER:
        need(st, 4)
        st.nobjects += 1
        put(obj, "value", PyLong_FromLongLong(<int32_t>be32(st.b + st.p)))
        st.p += 4
    elif kind == K_ENUMERATED:
        put(obj, "typeID", read_key(st, unknown))
        put(obj, "enum", read_key(st, unknown))
    elif kind == K_RAW_DATA:
        put(obj, "value", read_bytes(st, read_u32(st)))
    elif kind == K_CLASS:
        put(obj, "name", read_unicode(st))
        put(obj, "classID", read_key(st, unknown))
    elif kind == K_NAME:
        put(obj, "name", read_unicode(st))
        put(obj, "classID", read_key(st, unknown))
        put(obj, "value", read_unicode(st))
    else:
        raise _GiveUp
    return obj


def read(bytes data, Py_ssize_t position, long cap):
    """Read a descriptor body at ``position`` of ``data``.

    :param cap: container levels allowed, the body's own included.
    :return: ``(name, classID, items, end, bytes, objects, largest read,
        unknown terms)``, or None when the Python reader has to decide.
    """
    cdef State st
    cdef list unknown = []
    if position < 0 or position > len(data):
        return None
    st.b = <const unsigned char *>data
    st.n = len(data)
    st.p = position
    st.nbytes = st.nobjects = st.maxread = 0
    st.depth = 0
    st.cap = cap
    try:
        name, classID, items = read_body(&st, unknown)
    except _GiveUp:
        return None
    return name, classID, items, st.p, st.nbytes, st.nobjects, st.maxread, unknown
