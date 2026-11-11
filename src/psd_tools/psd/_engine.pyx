# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Compiled engine data parser.

Builds the objects ``engine_data.Dict.frombytes`` builds from the same bytes,
or gives up (returns None) on anything else, leaving the Python parser to
read the data and raise what it raises. Elements are created the way their
attrs ``__init__`` would leave them, without the per-element budget calls;
the counts come back to the caller to charge at once.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.float cimport PyFloat_FromString
from cpython.list cimport PyList_Append
from cpython.long cimport PyLong_FromLongLong
from cpython.object cimport PyObject_GenericSetAttr, PyObject_SetItem, PyTypeObject
from cpython.unicode cimport PyUnicode_DecodeASCII

cdef extern from "Python.h":
    object PyType_GenericAlloc(PyTypeObject *cls, Py_ssize_t nitems)
    void PyObject_GC_UnTrack(void *op)


cdef inline object _new(object cls):
    """``object.__new__(cls)`` without the call (what it allocates for these classes)."""
    return PyType_GenericAlloc(<PyTypeObject *>cls, 0)


cdef tuple _ESCAPES = ((b"\\\\", b"\\"), (b"\\(", b"("), (b"\\)", b")"))
cdef object _int = int
cdef object _OrderedDict = None
cdef object _Dict = None
cdef object _List = None
cdef object _String = None
cdef object _Bool = None
cdef object _Integer = None
cdef object _Float = None
cdef object _Property = None
cdef object _Tag = None
# One Property per key name (frozen, so shared); long names and those past the cap get their own.
cdef dict _KEYS = {}
cdef Py_ssize_t _KEYS_CAP = 4096
cdef Py_ssize_t _KEY_LEN = 64

cdef enum:
    T_STOP
    T_BAIL
    T_ARRAY_END
    T_ARRAY_START
    T_BOOLEAN
    T_DICT_END
    T_DICT_START
    T_NUMBER
    T_DECIMAL
    T_PROPERTY
    T_STRING
    T_TAG


def setup(Dict, List, String, Bool, Integer, Float, Property, Tag, OrderedDict):
    """Set the element classes the tokens map to."""
    global _Dict, _List, _String, _Bool, _Integer, _Float, _Property, _Tag
    global _OrderedDict
    _Dict, _List, _String, _Bool = Dict, List, String, Bool
    _Integer, _Float, _Property, _Tag = Integer, Float, Property, Tag
    _OrderedDict = OrderedDict
    _KEYS.clear()  # no keys of an older Property class


cdef inline bint _divider(unsigned char c) noexcept:
    return c == 32 or c == 10 or c == 9


cdef inline bint _digit(unsigned char c) noexcept:
    return 48 <= c <= 57


cdef inline bint _alnum(unsigned char c) noexcept:
    return _digit(c) or 65 <= c <= 90 or 97 <= c <= 122


cdef class _Parser:
    cdef object data
    cdef const unsigned char[::1] view
    cdef const unsigned char *b
    cdef Py_ssize_t n, i, ts, te
    cdef long long nbytes, nobjects, maxread
    cdef long depth, depth_cap
    cdef bint bail

    cdef int next_token(self):
        """Advance as ``Tokenizer.__next__`` does; the token is [ts, te)."""
        cdef const unsigned char *b = self.b
        cdef Py_ssize_t n = self.n
        cdef Py_ssize_t i, p
        while True:
            i = self.i
            if i >= n:
                return T_STOP
            if i + 3 <= n and b[i] == 40 and b[i + 1] == 0xFE and b[i + 2] == 0xFF:
                # UTF16_END: the first ")" past the start not after a backslash.
                p = i + 1
                while p < n and not (b[p] == 41 and b[p - 1] != 92):
                    p += 1
                if p >= n:
                    return T_BAIL
                self.ts, self.te, self.i = i, p + 1, p + 1
                break
            p = i
            while p < n and not _divider(b[p]):
                p += 1
            self.ts, self.te = i, p
            while p < n and _divider(b[p]):
                p += 1
            self.i = p
            if self.te > self.ts:
                break
        return self.classify()

    cdef int classify(self):
        """The token type, as the first matching ``EngineToken`` pattern names it."""
        cdef const unsigned char *b = self.b
        cdef Py_ssize_t s = self.ts, e = self.te
        cdef Py_ssize_t size = e - s, k
        cdef unsigned char c = b[s]
        if c == 93:  # ]
            return T_ARRAY_END if size == 1 else T_BAIL
        if c == 91:  # [
            return T_ARRAY_START if size == 1 else T_BAIL
        if c == 60:  # <
            return T_DICT_START if size == 2 and b[s + 1] == 60 else T_BAIL
        if c == 62:  # >>, then NULs
            if size < 2 or b[s + 1] != 62:
                return T_BAIL
            for k in range(s + 2, e):
                if b[k] != 0:
                    return T_BAIL
            return T_DICT_END
        if c == 47:  # /
            if size < 2:
                return T_BAIL
            for k in range(s + 1, e):
                if not (_alnum(b[k]) or b[k] == 95):
                    return T_BAIL
            return T_PROPERTY
        if c == 116:  # true
            if (size == 4 and b[s + 1] == 114 and b[s + 2] == 117
                    and b[s + 3] == 101):
                return T_BOOLEAN
            return T_BAIL
        if c == 102:  # false
            if (size == 5 and b[s + 1] == 97 and b[s + 2] == 108
                    and b[s + 3] == 115 and b[s + 4] == 101):
                return T_BOOLEAN
            return T_BAIL
        if c == 40:  # (
            if b[e - 1] != 41 or size < 2:
                return T_BAIL
            if size >= 4 and b[s + 1] == 0xFE and b[s + 2] == 0xFF:
                # Every ")" inside is escaped by the byte before it.
                for k in range(s + 3, e - 1):
                    if b[k] == 41 and (k - 1 < s + 3 or b[k - 1] != 92):
                        break
                else:
                    return T_STRING
            for k in range(s + 1, e - 1):
                if not _alnum(b[k]):
                    return T_BAIL
            return T_TAG
        if c == 45 or c == 46 or _digit(c):  # -, ., digits
            return self.classify_number(s, e)
        return T_BAIL

    cdef int classify_number(self, Py_ssize_t s, Py_ssize_t e):
        cdef const unsigned char *b = self.b
        cdef Py_ssize_t k = s
        cdef Py_ssize_t digits = 0, fraction = 0
        if b[k] == 45:
            k += 1
        while k < e and _digit(b[k]):
            k += 1
            digits += 1
        if k == e:
            if digits:
                return T_NUMBER
        elif b[k] == 46:
            k += 1
            while k < e and _digit(b[k]):
                k += 1
                fraction += 1
            if k == e and fraction:
                return T_DECIMAL
        # UNKNOWN_TAG2
        if (e - s == 6 and b[s] == 45 and b[s + 1] == 45 and b[s + 2] == 40
                and b[s + 3] == 46 and b[s + 4] == 45 and b[s + 5] == 48):
            return T_TAG
        return T_BAIL

    cdef inline void charge(self, Py_ssize_t size):
        self.nbytes += size
        if size > self.maxread:
            self.maxread = size
        self.nobjects += 1

    cdef object token(self):
        return PyBytes_FromStringAndSize(
            <const char *>(self.b + self.ts), self.te - self.ts
        )

    cdef object value(self, int kind):
        """The element of a scalar token, as its class's ``frombytes`` makes it."""
        cdef Py_ssize_t s = self.ts, e = self.te
        cdef Py_ssize_t size = e - s, k
        cdef long long number = 0
        cdef bint negative
        cdef object obj, raw, v, escaped, plain
        if kind == T_PROPERTY:
            v = PyUnicode_DecodeASCII(<const char *>(self.b + s + 1), size - 1, NULL)
            obj = _KEYS.get(v)
            if obj is not None:
                self.charge(size)
                return obj
            obj = _new(_Property)
            PyObject_GenericSetAttr(obj, "value", v)
            PyObject_GC_UnTrack(<void *>obj)
            if size <= _KEY_LEN and len(_KEYS) < _KEYS_CAP:
                _KEYS[v] = obj
            self.charge(size)
            return obj
        elif kind == T_STRING:
            raw = PyBytes_FromStringAndSize(<const char *>(self.b + s + 1), size - 2)
            if b"\\" in raw:  # String._ESCAPED_CHARS, undone in its order
                for escaped, plain in _ESCAPES:
                    raw = raw.replace(escaped, plain)
            v = raw.decode("utf-16")
            obj = _new(_String)
        elif kind == T_NUMBER:
            negative = self.b[s] == 45
            k = s + 1 if negative else s
            if e - k <= 18:
                while k < e:
                    number = number * 10 + (self.b[k] - 48)
                    k += 1
                v = PyLong_FromLongLong(-number if negative else number)
            else:
                v = _int(self.token())
            obj = _new(_Integer)
        elif kind == T_DECIMAL:
            v = PyFloat_FromString(self.token())
            obj = _new(_Float)
        elif kind == T_BOOLEAN:
            v = size == 4
            obj = _new(_Bool)
        else:  # T_TAG: Tag.frombytes charges its bytes twice
            v = self.token()
            obj = _new(_Tag)
            self.nbytes += size
        PyObject_GenericSetAttr(obj, "value", v)
        PyObject_GC_UnTrack(<void *>obj)  # atomic: in no cycle, as CPython untracks such tuples
        self.charge(size)
        return obj

    cdef object parse_dict(self, object cls):
        """``Dict.frombytes`` on the tokens left; None to give up."""
        cdef int kind
        cdef object obj, items, key, value
        self.depth += 1
        if self.depth > self.depth_cap:
            self.bail = True
            return None
        obj = _new(cls)
        items = _OrderedDict()
        PyObject_GenericSetAttr(obj, "_items", items)
        self.nobjects += 1
        while True:
            kind = self.next_token()
            if kind == T_PROPERTY:
                key = self.value(T_PROPERTY)
                kind = self.next_token()
                if kind == T_DICT_START:
                    value = self.parse_dict(_Dict)
                elif kind == T_ARRAY_START:
                    value = self.parse_list()
                elif kind >= T_BOOLEAN and kind != T_DICT_END:
                    value = self.value(kind)
                else:
                    self.bail = True
                if self.bail:
                    return None
                PyObject_SetItem(items, key, value)
            elif kind == T_DICT_END or kind == T_STOP:
                self.depth -= 1
                PyObject_GC_UnTrack(<void *>items)
                PyObject_GC_UnTrack(<void *>obj)
                return obj
            elif kind == T_BAIL:
                self.bail = True
                return None

    cdef object parse_list(self):
        """``List.frombytes`` on the tokens left; None to give up."""
        cdef int kind
        cdef object obj, items, value
        self.depth += 1
        if self.depth > self.depth_cap:
            self.bail = True
            return None
        obj = _new(_List)
        items = []
        PyObject_GenericSetAttr(obj, "_items", items)
        self.nobjects += 1
        while True:
            kind = self.next_token()
            if kind == T_ARRAY_END or kind == T_STOP:
                self.depth -= 1
                PyObject_GC_UnTrack(<void *>items)
                PyObject_GC_UnTrack(<void *>obj)
                return obj
            if kind == T_DICT_START:
                value = self.parse_dict(_Dict)
            elif kind == T_ARRAY_START:
                value = self.parse_list()
            elif kind >= T_BOOLEAN and kind != T_DICT_END:
                value = self.value(kind)
            else:
                self.bail = True
            if self.bail:
                return None
            PyList_Append(items, value)


def parse(cls, bytes data, long depth_cap):
    """Parse ``data`` into a ``cls`` element.

    :param depth_cap: nesting levels allowed below the root.
    :return: ``(element, bytes, objects, largest read)`` to charge to the
        budget, or None when the Python parser has to decide.
    """
    cdef _Parser parser = _Parser()
    cdef object obj
    if not data:
        return None
    parser.data = data
    parser.view = data
    parser.b = &parser.view[0]
    parser.n = len(data)
    parser.depth = -1  # the root's level is charged by the caller
    parser.depth_cap = depth_cap
    obj = parser.parse_dict(cls)
    if parser.bail:
        return None
    return obj, parser.nbytes, parser.nobjects, parser.maxread


# ------------------------------ writer ------------------------------
# Builds the bytes ``Dict.write``/``List.write`` build for the exact element
# classes, or gives up (None) on anything else, leaving Python to write.

from cpython.bytes cimport PyBytes_AsString, PyBytes_Size
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from libc.math cimport fabs, signbit
from cpython.long cimport PyLong_AsLongLongAndOverflow
from cpython.dict cimport PyDict_Next
from cpython.object cimport PyObject
from libc.string cimport memcpy

cdef extern from "Python.h":
    char *PyOS_double_to_string(double val, char format_code, int precision, int flags, int *type) except NULL
    Py_ssize_t PyUnicode_GetLength(object o) except -1

cdef object _EngineData = None
cdef object _EngineData2 = None
cdef long _WRITE_DEPTH = 100  # nesting the writer takes; deeper goes to Python


def setup_writer(EngineData, EngineData2):
    """Set the root classes the writer takes besides ``Dict``."""
    global _EngineData, _EngineData2
    _EngineData, _EngineData2 = EngineData, EngineData2


cdef class _Out:
    cdef char *buf
    cdef Py_ssize_t n, cap

    def __cinit__(self):
        self.cap = 1 << 16
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
        memcpy(self.buf + self.n, s, k)
        self.n += k
        return 0

    cdef inline int putc(self, char c) except -1:
        if self.n + 1 > self.cap:
            self.grow(1)
        self.buf[self.n] = c
        self.n += 1
        return 0

    cdef int tabs(self, long k) except -1:
        if self.n + k > self.cap:
            self.grow(k)
        cdef long i
        for i in range(k):
            self.buf[self.n + i] = 9
        self.n += k
        return 0


class _Bail(Exception):
    pass


cdef int _w_name(_Out out, object value) except -1:
    """``b"/" + value.encode("macroman")`` for a str value."""
    cdef Py_ssize_t k
    cdef const char *s
    cdef bytes data
    out.putc(47)
    s = PyUnicode_AsUTF8AndSize(value, &k)
    if k == PyUnicode_GetLength(value):  # ASCII: macroman's bytes too
        out.put(s, k)
    else:
        data = value.encode("macroman")
        out.put(PyBytes_AsString(data), PyBytes_Size(data))
    return 0


cdef int _w_int(_Out out, long long n) except -1:
    """``b"%d" % n``."""
    cdef char tmp[24]
    cdef int k = 24
    cdef unsigned long long u = <unsigned long long>(-(n + 1)) + 1 if n < 0 else <unsigned long long>n
    while True:
        k -= 1
        tmp[k] = 48 + <char>(u % 10)
        u //= 10
        if u == 0:
            break
    if n < 0:
        k -= 1
        tmp[k] = 45
    out.put(tmp + k, 24 - k)
    return 0


cdef int _w_scalar(_Out out, object v) except -1:
    cdef object t = type(v)
    cdef long long n
    cdef int overflow = 0
    cdef object x
    cdef bytes data
    cdef const unsigned char *s
    cdef Py_ssize_t k, i, end
    cdef char *f
    cdef double d
    if t is _Integer:
        x = v.value
        if type(x) is not int:
            raise _Bail()
        n = PyLong_AsLongLongAndOverflow(x, &overflow)
        if overflow:
            data = b"%d" % x
            out.put(PyBytes_AsString(data), PyBytes_Size(data))
        else:
            _w_int(out, n)
    elif t is _Float:
        x = v.value
        if type(x) is not float:
            raise _Bail()
        d = x
        if d == d and fabs(d) < 1e15 and d == <double>(<long long>d):
            # whole: b"%.8f" stripped is the integer and ".0"
            if d == 0.0:
                if signbit(d):
                    out.put(b"-0.0", 4)
                else:
                    out.put(b"0.0", 3)
            else:
                _w_int(out, <long long>d)
                out.put(b".0", 2)
            return 0
        f = PyOS_double_to_string(d, 102, 8, 0, NULL)  # b"%.8f"
        try:
            k = 0
            while f[k] != 0:
                k += 1
            end = k
            while end > 0 and f[end - 1] == 48:  # rstrip(b"0")
                end -= 1
            i = 0
            if fabs(d) > 0.0 and fabs(d) < 1.0:  # replace(b"0.", b".")
                if f[0] == 48 and end > 1 and f[1] == 46:
                    i = 1
                elif f[0] == 45 and end > 2 and f[1] == 48 and f[2] == 46:
                    out.putc(45)
                    i = 2
            out.put(f + i, end - i)
            if end > 0 and f[end - 1] == 46:  # b"." -> b".0"
                out.putc(48)
        finally:
            PyMem_Free(f)
    elif t is _Bool:
        x = v.value
        if x is True:
            out.put(b"true", 4)
        elif x is False:
            out.put(b"false", 5)
        else:
            raise _Bail()
    elif t is _String:
        x = v.value
        if type(x) is not str:
            raise _Bail()
        data = x.encode("utf-16-be")
        s = <const unsigned char *>PyBytes_AsString(data)
        k = PyBytes_Size(data)
        out.put(b"(\xfe\xff", 3)
        if out.n + 2 * k + 1 > out.cap:
            out.grow(2 * k + 1)
        for i in range(k):
            if s[i] == 92 or s[i] == 40 or s[i] == 41:
                out.buf[out.n] = 92
                out.n += 1
            out.buf[out.n] = s[i]
            out.n += 1
        out.putc(41)
    elif t is _Property:
        x = v.value
        if type(x) is not str:
            raise _Bail()
        _w_name(out, x)
    elif t is _Tag:
        x = v.value
        if type(x) is not bytes:
            raise _Bail()
        out.put(PyBytes_AsString(x), PyBytes_Size(x))
    else:
        raise _Bail()
    return 0


cdef int _w_value(_Out out, object value, long inner, bint has_indent, long depth) except -1:
    """A dict entry's value, after its key."""
    cdef object t = type(value), first, ft
    if t is _Dict or t is _EngineData:
        _w_dict(out, value, inner, has_indent, True, depth)
        return 0
    out.putc(32)
    if t is _List:
        items = value._items
        nested = False
        if len(items) > 0:
            first = items[0]
            ft = type(first)
            if ft is _Dict or ft is _EngineData:
                nested = True
            elif not (ft is _List or ft is _Integer or ft is _Float or ft is _Bool
                      or ft is _String or ft is _Property or ft is _Tag):
                raise _Bail()  # isinstance(first, Dict) is Python's to decide
        if nested:
            _w_list(out, value, inner, has_indent, depth)
        else:
            _w_list(out, value, 0, False, depth)
    else:
        _w_scalar(out, value)
    return 0


cdef int _w_key(_Out out, object key) except -1:
    if type(key) is _Property:
        v = key.value
        if type(v) is str:
            _w_name(out, v)
            return 0
    raise _Bail()


cdef int _w_dict(_Out out, object d, long indent, bint has_indent, bint container, long depth) except -1:
    cdef long inner = indent + 1
    cdef object key, value, items
    cdef Py_ssize_t pos = 0
    cdef PyObject *k
    cdef PyObject *v
    cdef bint lockstep = True
    if depth > _WRITE_DEPTH:
        raise _Bail()  # deeper: Python's, with its recursion limit
    if container:
        if not has_indent:
            out.put(b" <<", 3)
        else:
            if indent == 0:
                out.putc(10)
            out.putc(10)
            out.tabs(indent)
            out.put(b"<<\n", 3)
    items = d._items
    if type(items) is not _OrderedDict:
        raise _Bail()
    for key in items:  # the OrderedDict's order; values from the dict while it agrees, unhashed
        if lockstep and PyDict_Next(items, &pos, &k, &v) and <object>k is key:
            value = <object>v
        else:
            lockstep = False
            value = items[key]
        if has_indent:
            out.tabs(inner)
        else:
            out.putc(32)
        _w_key(out, key)
        _w_value(out, value, inner, has_indent, depth + 1)
        if has_indent:
            out.putc(10)
    if container:
        if not has_indent:
            out.put(b" >>", 3)
        else:
            out.tabs(indent)
            out.put(b">>", 2)
    return 0


cdef int _w_list(_Out out, object lst, long indent, bint has_indent, long depth) except -1:
    cdef object item, t
    if depth > _WRITE_DEPTH:
        raise _Bail()
    depth += 1
    out.putc(91)
    if not has_indent:
        for item in lst._items:
            t = type(item)
            if t is _Dict or t is _EngineData:
                _w_dict(out, item, 0, False, True, depth)
            else:
                out.putc(32)
                if t is _List:
                    _w_list(out, item, 0, False, depth)
                else:
                    _w_scalar(out, item)
        out.putc(32)
    else:
        for item in lst._items:
            t = type(item)
            if t is _Dict or t is _EngineData:
                _w_dict(out, item, indent, True, True, depth)
            elif t is _List:
                _w_list(out, item, indent, True, depth)
            else:
                raise _Bail()
        out.putc(10)
        out.tabs(indent)
    out.putc(93)
    return 0


def dumps(obj, indent, bint container):
    """``Dict.write``'s (or ``List.write``'s) bytes for exact classes; None to leave it to Python."""
    cdef _Out out
    cdef object t = type(obj)
    cdef bint has_indent = indent is not None
    cdef long level = 0
    if _Dict is None or _EngineData is None:
        return None
    if has_indent:
        if type(indent) is not int or not 0 <= indent <= 1 << 20:
            return None
        level = indent
    out = _Out()
    try:
        if t is _Dict or t is _EngineData or t is _EngineData2:
            _w_dict(out, obj, level, has_indent, container, 0)
        elif t is _List:
            _w_list(out, obj, level, has_indent, 0)
        else:
            return None
    except Exception:  # _Bail, or what Python raises the same way
        return None
    return PyBytes_FromStringAndSize(out.buf, out.n)
