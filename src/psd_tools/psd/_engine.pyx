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
from cpython.object cimport PyObject_GenericSetAttr, PyObject_SetItem
from cpython.unicode cimport PyUnicode_DecodeASCII

cdef object _new = object.__new__
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
            obj = _new(_Property)
            v = PyUnicode_DecodeASCII(<const char *>(self.b + s + 1), size - 1, NULL)
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
