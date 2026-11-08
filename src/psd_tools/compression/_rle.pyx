# distutils: language=c++
# cython: wraparound=False, binding=False

from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize
from libc.stdint cimport uint64_t
from libc.string cimport memcpy, memset
from libcpp.string cimport string
from libcpp.algorithm cimport copy_n, fill_n

cdef void _decode_into(
    const unsigned char* data, Py_ssize_t length, char* out, Py_ssize_t size
) noexcept nogil:
    """PackBits-decode ``length`` bytes into ``size`` zeroed bytes at ``out``."""
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t actual, available
    cdef unsigned char bit

    if length == 1:
        # Single byte: either a no-op (128) or a stray header — return zeros
        return

    while i < length and j < size:
        i, bit = i+1, data[i]
        if bit > 128:
            bit = 256 - bit
            if i >= length:  # lone repeat header at end of stream — stop
                break
            actual = min(1+bit, size-j)  # clip at remaining output space
            fill_n(out+j, actual, <char>data[i])
            j += actual
            i += 1
        elif bit < 128:
            if i >= length:  # copy header is the last byte; nothing to copy
                break
            available = min(length-i, 1+bit)
            actual = min(available, size-j)  # clip to input and output
            copy_n(&data[i], actual, out+j)
            j += actual
            i += available  # advance by declared amount or to end
        # bit == 128: no-op


cdef Py_ssize_t _decode_fast(
    const unsigned char* data, Py_ssize_t length, Py_ssize_t readable,
    char* out, Py_ssize_t size, Py_ssize_t writable
) noexcept nogil:
    """_decode_into, but a packet of 16 bytes or less is two 8-byte stores that
    may write ahead into bytes written later; ``readable`` and ``writable``
    bound the reads and writes. Returns the bytes decoded, unzeroed."""
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t actual, available
    cdef unsigned char bit
    cdef uint64_t word

    if length == 1:
        return 0

    while i < length and j < size:
        i, bit = i+1, data[i]
        if bit > 128:
            bit = 256 - bit
            if i >= length:
                break
            actual = min(1+bit, size-j)
            if actual <= 16 and j + 16 <= writable:
                word = <uint64_t>data[i] * <uint64_t>0x0101010101010101
                memcpy(out+j, &word, 8)
                memcpy(out+j+8, &word, 8)
            else:
                memset(out+j, data[i], actual)
            j += actual
            i += 1
        elif bit < 128:
            if i >= length:
                break
            available = min(length-i, 1+bit)
            actual = min(available, size-j)
            if actual <= 16 and j + 16 <= writable and i + 16 <= readable:
                memcpy(out+j, &data[i], 16)
            else:
                memcpy(out+j, &data[i], actual)
            j += actual
            i += available
    return j


def decode(const unsigned char[:] data, Py_ssize_t size) -> string:
    """decode(data, size) -> bytes

    Apple PackBits RLE decoder.

    Tolerant implementation: runs that would exceed *size* are clipped at the
    row boundary, runs whose input is truncated copy what is available, and any
    remaining bytes are zero-padded (std::string::resize zero-initialises).
    The function always returns exactly *size* bytes without raising.
    """
    cdef string result
    result.resize(size)  # zero-initialised by std::string::resize
    if data.shape[0]:
        _decode_into(&data[0], data.shape[0], &result[0], size)
    return result


def decode_rows(
    const unsigned char[:] data, Py_ssize_t start, counts, Py_ssize_t size
):
    """decode_rows(data, start, counts, size) -> bytes

    :func:`decode` of each row in turn, joined: the rows follow one another
    from ``start``, each ``count`` bytes long, and a row the data runs out
    under decodes what is left, as reading them one at a time would.
    """
    cdef Py_ssize_t length = data.shape[0]
    cdef Py_ssize_t pos = min(max(start, 0), length)
    cdef Py_ssize_t row = 0, count, end, done
    cdef Py_ssize_t rows = len(counts)
    cdef Py_ssize_t total = rows * size
    result = PyBytes_FromStringAndSize(NULL, total)
    cdef char* buf = PyBytes_AsString(result)
    cdef char* out
    for count in counts:
        if row == rows:
            break
        end = min(pos + count, length)
        out = buf + row * size
        done = 0
        if end > pos:
            done = _decode_fast(
                &data[pos], end - pos, length - pos, out, size, total - row * size
            )
        memset(out + done, 0, size - done)  # tail, write-ahead included
        pos = end
        row += 1
    memset(buf + row * size, 0, (rows - row) * size)
    return result


def encode(const unsigned char[:] data) -> string:
    """encode(data) -> bytes

    Apple PackBits RLE encoder.
    """

    cdef unsigned char MAX_LEN = 0xFF >> 1
    cdef int length = data.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef string result

    if length == 0:
        return result
    if length == 1:
        result.push_back(0)
        result.push_back(data[0])
        return result

    while i < length:
        if j + 1 < length and data[j] == data[j+1]:
            while j < length:
                if j - i >= MAX_LEN:
                    break
                if j + 1 >= length or data[j] != data[j+1]:
                    break
                j += 1
            result.push_back(- (j - i))
            result.push_back(<char>data[i])
            i = j = j + 1
        else:
            while j < length:
                if j - i >= MAX_LEN:
                    break
                if j+1 < length and (data[j] != data[j+1]):
                    pass
                # NOTE: There's no space saved from encoding length 2 repetitions in this situation.
                #: For example:
                #  A  B  C  D  D  E  F  G  G  G  G  G  G  H  I  J  J  K
                #: could be encoded as either of the following:
                # +2  A  B  C -1  D +1  E  F -5  G +1  H  I -1  J +0  K
                # +6  A  B  C  D  D  E  F -5  G +4  H  I  J  J  K
                elif ((j+2 == length) or (MAX_LEN - (j - i) <= 2)) and not (j+1 == length) and (data[j] == data[j+1]):
                    break
                elif j+2 < length and (data[j] == data[j+1] == data[j+2]):
                    break
                j += 1
            result.push_back(j - i - 1)
            result.append(<char*>&data[i], j - i)
            i = j
    return result
