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
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject_GenericSetAttr
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from cpython.unicode cimport PyUnicode_DecodeUTF16
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy

cdef object _new = object.__new__
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


cdef struct Entry:
    uint32_t code  # the 4 bytes, big endian
    int kind
    PyObject *value  # kept alive by _held

# OSTypes and units by code, filled by setup().
cdef Entry *_kinds = NULL
cdef Py_ssize_t _nkinds = 0
cdef Entry *_units = NULL
cdef Py_ssize_t _nunits = 0
cdef list _held = []

# The last zero-length key read per slot, so equal keys share one object.
# Changed only under the GIL (the module does not claim free-threading).
cdef enum:
    KEY_SLOTS = 4096
cdef uint32_t _key_codes[KEY_SLOTS]
cdef bint _key_unknown[KEY_SLOTS]
cdef PyObject *_key_objs[KEY_SLOTS]  # owned; NULL when free


cdef Py_ssize_t fill(Entry **table, dict rows, bint pairs) except -1:
    """Put ``rows``' 4-byte keys in a new ``*table``; return how many."""
    cdef list items = [(k, v) for k, v in rows.items() if len(k) == 4]
    cdef Entry *made = <Entry *>PyMem_Malloc((len(items) or 1) * sizeof(Entry))
    cdef Py_ssize_t i
    if made == NULL:
        raise MemoryError
    for i, (k, v) in enumerate(items):
        made[i].code = be32(<const unsigned char *><bytes>k)
        made[i].kind = v[1] if pairs else 0
        value = v[0] if pairs else v
        made[i].value = <PyObject *>value
        _held.append(value)
    PyMem_Free(table[0])
    table[0] = made
    return len(items)


def setup(kinds, units, terms, ZeroLengthKey, OrderedDict):
    """Set the OSType table (``{bytes: (class, kind)}``), units and terms."""
    global _nkinds, _nunits, _terms, _ZeroLengthKey, _OrderedDict
    cdef Py_ssize_t i
    cdef PyObject *old
    _nkinds = _nunits = 0
    _held.clear()
    _nkinds = fill(&_kinds, dict(kinds), True)
    _nunits = fill(&_units, dict(units), False)
    _terms, _ZeroLengthKey, _OrderedDict = terms, ZeroLengthKey, OrderedDict
    for i in range(KEY_SLOTS):
        old = _key_objs[i]
        _key_objs[i] = NULL
        Py_XDECREF(old)


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
    cdef int order = 1  # the "utf-16-be" codec: big endian, a BOM kept
    need(st, size)
    value = PyUnicode_DecodeUTF16(<char *>(st.b + st.p), size, NULL, &order)
    st.p += size
    return value


cdef object read_key(State *st, list unknown):
    """``read_length_and_key(fp)``; an unknown term is logged by the caller."""
    cdef long long length = read_u32(st)
    cdef uint32_t code
    cdef Py_ssize_t slot
    cdef bint unknown_key
    cdef PyObject *old
    if length:
        return read_bytes(st, length)
    need(st, 4)
    code = be32(st.b + st.p)
    slot = ((code * <uint32_t>2654435761u) >> 20) & (KEY_SLOTS - 1)
    if _key_objs[slot] != NULL and _key_codes[slot] == code:
        key = <object>_key_objs[slot]
        unknown_key = _key_unknown[slot]
    else:
        key = PyBytes_FromStringAndSize(<const char *>(st.b + st.p), 4)
        unknown_key = key not in _terms
        if unknown_key:
            key = _ZeroLengthKey(key)
        old = _key_objs[slot]
        Py_INCREF(key)
        _key_codes[slot] = code
        _key_unknown[slot] = unknown_key
        _key_objs[slot] = <PyObject *>key
        Py_XDECREF(old)
    st.p += 4
    if unknown_key:
        unknown.append(key)
    return key


cdef object unit_of(uint32_t code):
    """``_unit(...)`` of a known unit, or give up."""
    cdef Py_ssize_t j
    for j in range(_nunits):
        if _units[j].code == code:
            return <object>_units[j].value
    raise _GiveUp


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
    cdef uint32_t code
    cdef Py_ssize_t j
    need(st, 4)
    code = be32(st.b + st.p)
    st.p += 4
    for j in range(_nkinds):
        if _kinds[j].code == code:
            break
    else:
        raise _GiveUp
    kind = _kinds[j].kind
    obj = _new(<object>_kinds[j].value)
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
        unit = unit_of(be32(st.b + st.p))
        number = be_double(st.b + st.p + 4)
        st.p += 12
        put(obj, "value", PyFloat_FromDouble(number))
        put(obj, "unit", unit)
    elif kind == K_UNIT_FLOATS:
        need(st, 8)
        st.nobjects += 2
        unit = unit_of(be32(st.b + st.p))
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


# ------------------------------ writer ------------------------------
# ``descriptor._body_bytes`` for the exact classes and values it packs itself;
# any other value is written by its own ``write`` after the buffer is flushed,
# and odd numbers go through the same ``struct`` packers, as there.

from cpython.bytes cimport PyBytes_AsString, PyBytes_Size
from cpython.mem cimport PyMem_Realloc
from libc.string cimport memcpy as _memcpy

cdef dict _ostypes = {}  # exact class -> its OSType's 4 bytes
cdef long _WRITE_DEPTH = 100  # nesting written here; deeper values write themselves
cdef object W_UNIT_FLOAT = None, W_ENUMERATED = None, W_DOUBLE = None, W_DESCRIPTOR = None
cdef object W_GLOBAL = None, W_BOOL = None, W_INTEGER = None, W_STRING = None, W_LIST = None
cdef object W_LARGE = None, W_OBJECT_ARRAY = None, W_RAW = None
cdef object P_U32 = None, P_I32 = None, P_I64 = None, P_F64 = None, P_UNIT_F64 = None, W_BYTES = None


def setup_writer(ostypes, classes, packers, write_bytes):
    """Set every class's OSType bytes, the classes written here, the packers and ``write_bytes``."""
    global W_UNIT_FLOAT, W_ENUMERATED, W_DOUBLE, W_DESCRIPTOR, W_GLOBAL, W_BOOL, W_INTEGER, W_STRING, W_LIST
    global W_LARGE, W_OBJECT_ARRAY, W_RAW, P_U32, P_I32, P_I64, P_F64, P_UNIT_F64, W_BYTES
    _ostypes.clear()
    _ostypes.update(ostypes)
    (W_UNIT_FLOAT, W_ENUMERATED, W_DOUBLE, W_DESCRIPTOR, W_GLOBAL, W_BOOL, W_INTEGER, W_STRING, W_LIST,
     W_LARGE, W_OBJECT_ARRAY, W_RAW) = classes
    P_U32, P_I32, P_I64, P_F64, P_UNIT_F64 = packers
    W_BYTES = write_bytes


cdef class _WOut:
    cdef char *buf
    cdef Py_ssize_t n, cap
    cdef object fp
    cdef object written  # what went to fp
    cdef bint bad  # a piece b"".join refuses: its TypeError at the flush, as there

    def __cinit__(self):
        self.cap = 1 << 14
        self.n = 0
        self.buf = <char *>PyMem_Malloc(self.cap)
        if self.buf == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.buf != NULL:
            PyMem_Free(self.buf)

    cdef int grow(self, Py_ssize_t extra) except -1:
        cdef Py_ssize_t cap = self.cap
        cdef char *b
        while self.n + extra > cap:
            cap *= 2
        b = <char *>PyMem_Realloc(self.buf, cap)
        if b == NULL:
            raise MemoryError()
        self.buf, self.cap = b, cap
        return 0

    cdef inline int put(self, const char *s, Py_ssize_t k) except -1:
        if self.n + k > self.cap:
            self.grow(k)
        _memcpy(self.buf + self.n, s, k)
        self.n += k
        return 0

    cdef inline int u32(self, uint32_t v) except -1:
        cdef char b[4]
        b[0] = <char>(v >> 24)
        b[1] = <char>(v >> 16)
        b[2] = <char>(v >> 8)
        b[3] = <char>v
        return self.put(b, 4)

    cdef inline int u64(self, uint64_t v) except -1:
        cdef char b[8]
        cdef int i
        for i in range(8):
            b[i] = <char>(v >> (56 - 8 * i))
        return self.put(b, 8)

    cdef inline int f64(self, double d) except -1:
        cdef uint64_t u
        _memcpy(&u, &d, 8)
        return self.u64(u)

    cdef int data(self, object data) except -1:
        """A bytes object (or subclass) as it is; another bytes-like by its bytes."""
        if not isinstance(data, bytes):
            try:
                data = bytes(memoryview(data))
            except TypeError:
                self.bad = True
                return 0
        self.put(PyBytes_AsString(data), PyBytes_Size(data))
        return 0

    cdef int flush(self) except -1:
        if self.bad:
            raise TypeError("expected a bytes-like object")
        if self.n:
            self.written += W_BYTES(self.fp, PyBytes_FromStringAndSize(self.buf, self.n))
            self.n = 0
        return 0

    cdef int delegate(self, object v) except -1:
        """``write_joined(fp, out) + v.write(fp)``."""
        self.flush()
        self.written += v.write(self.fp)
        return 0


cdef int _w_len(_WOut out, Py_ssize_t n) except -1:
    """``_U32(n)``: the packer past 32 bits, for its error."""
    if 0 <= n <= 0xFFFFFFFF:
        out.u32(<uint32_t>n)
    else:
        out.data(P_U32(n))
    return 0


cdef int _w_key(_WOut out, object value) except -1:
    """``_key_bytes``."""
    if value in _terms or isinstance(value, _ZeroLengthKey):
        out.u32(0)
    else:
        _w_len(out, len(value))
    out.data(value)
    return 0


cdef int _w_ostype(_WOut out, object v) except -1:
    code = _ostypes.get(type(v))
    if code is None:
        code = v.ostype.value
    out.data(code)
    return 0


cdef int _w_body(_WOut out, object d, long depth) except -1:
    cdef bytes name = d.name.encode("utf-16-be")
    out.u32(<uint32_t>(PyBytes_Size(name) // 2))
    out.data(name)
    _w_key(out, d.classID)
    items = d._items
    _w_len(out, len(items))
    for key, value in items.items():
        _w_key(out, key)
        _w_ostype(out, value)
        _w_value(out, value, depth + 1)
    return 0


cdef int _w_value(_WOut out, object v, long depth) except -1:
    cdef object t = type(v), x, unit
    cdef long long n
    cdef bytes data
    if depth > _WRITE_DEPTH:  # deeper: its own write, with Python's recursion limit
        t = None
    if t is W_UNIT_FLOAT:
        unit = v.unit.value
        x = v.value
        if type(unit) is bytes and PyBytes_Size(unit) == 4 and type(x) is float:
            out.data(unit)
            out.f64(x)
        else:
            out.data(P_UNIT_F64(unit, x))
    elif t is W_ENUMERATED:
        _w_key(out, v.typeID)
        _w_key(out, v.enum)
    elif t is W_DOUBLE:
        x = v.value
        if type(x) is float:
            out.f64(x)
        else:
            out.data(P_F64(x))
    elif t is W_DESCRIPTOR or t is W_GLOBAL:
        _w_body(out, v, depth)
    elif t is W_BOOL and type(v.value) is bool:
        out.put(b"\x01" if v.value else b"\x00", 1)
    elif t is W_INTEGER:
        x = v.value
        if type(x) is int and -2147483648 <= x <= 2147483647:
            n = x
            out.u32(<uint32_t>(<int32_t>n))
        else:
            out.data(P_I32(x))
    elif t is W_STRING and type(v.value) is str:
        data = v.value.encode("utf-16-be")
        out.u32(<uint32_t>(PyBytes_Size(data) // 2))
        out.data(data)
    elif t is W_LIST:
        items = v._items
        _w_len(out, len(items))
        for item in items:
            _w_ostype(out, item)
            _w_value(out, item, depth + 1)
    elif t is W_LARGE:
        x = v.value
        if type(x) is int and -9223372036854775808 <= x <= 9223372036854775807:
            n = x
            out.u64(<uint64_t>n)
        else:
            out.data(P_I64(x))
    elif t is W_OBJECT_ARRAY:
        out.data(P_U32(v.items_count))
        _w_body(out, v, depth)
    elif t is W_RAW and type(v.value) is bytes:
        _w_len(out, len(v.value))
        out.data(v.value)
    else:
        out.delegate(v)
    return 0


def write_body(d, fp):
    """``_body_bytes(d, out, fp) + write_joined(fp, out)``: the bytes written to ``fp``."""
    cdef _WOut out = _WOut()
    out.fp = fp
    out.written = 0
    _w_body(out, d, 0)
    out.flush()
    return out.written
