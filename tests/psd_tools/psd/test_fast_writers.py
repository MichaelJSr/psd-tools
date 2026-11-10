"""The descriptor and engine data writers' fast paths (compiled and Python) write what the plain ones do."""

import io
import os
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.psd import descriptor as D
from psd_tools.psd import engine_data as E
from psd_tools.terminology import Enum, Key, Klass, Type, Unit

from ..utils import TEST_ROOT, full_name


def _fast_and_plain(
    monkeypatch: pytest.MonkeyPatch, write: Callable[[], bytes]
) -> tuple[bytes, bytes, bytes]:
    compiled = write()
    with monkeypatch.context() as m:
        m.setattr(D, "_WRITER", False)
        m.setattr(E, "_WRITER", False)
        fast = write()
        m.setattr(D, "_FAST_BODIES", frozenset())
        m.setattr(E, "_FAST_WRITERS", frozenset())
        plain = write()
    return compiled, fast, plain


L = TypeVar("L", E.List, D.List)


def _list(kls: type[L], *items: Any) -> L:
    lst = kls()
    lst.extend(items)
    return lst


# Subclasses: the fast writers leave them to their own write.
class _Floats(E.List):
    pass


class _Int(E.Integer):
    pass


class _Key(E.Property):
    pass


# Floats past the easy cases: rounding to "1.0", ".0" and "-.0", signed zero, whole and huge.
_FLOATS = [-0.0, 4e-9, -4e-9, 0.999999999, -0.999999996, 1e16, 2.0**63, 0.1]
_FLOATS += [float("nan"), float("inf"), -float("inf"), 100.000000004]


def _exact(depth: int = 0) -> E.Dict:
    """Exact classes only: the compiled writer takes it unless deeper than 100."""
    with open(os.path.join(TEST_ROOT, "engine_data", "TySh_1.dat"), "rb") as f:
        data = E.EngineData.frombytes(f.read())
    data["EngineDict"]["Editor"]["Text"] = E.String("é (x) \\ 😀\r")
    data["Edges"] = _list(E.List, *(E.Float(x) for x in _FLOATS))
    data["Big"] = _list(E.List, E.Integer(2**70), E.Integer(-(2**63)))
    data["Été"] = E.Integer(1)
    data._items.move_to_end(next(iter(data._items)))  # dict order != its own
    deep = data
    for _ in range(depth):
        deep["D"] = E.Dict()
        deep = deep["D"]
    return data


def _engine_data() -> E.Dict:
    data = _exact(120)
    data["Floats"] = _list(_Floats, *(E.Float(x) for x in (0.5, -0.25, 1e-9, 12.0)))
    first = E.Dict()
    first["A"] = E.Integer(1)
    data["Mixed"] = _list(E.List, first, E.Float(0.75), E.Tag(b"(hwid)"))
    data[_Key("Sub")] = E.Integer(1)
    two = E.EngineData2()
    two["B"] = E.Bool(True)
    data["Two"] = two
    flat = _list(E.List, E.Float(-1.5), E.Float(0.0))
    data["Flat"] = _list(E.List, E.Integer(3), two, _Int(2), flat)
    return data


def _descriptor() -> D.Descriptor:
    with open(os.path.join(TEST_ROOT, "descriptors", "1.dat"), "rb") as f:
        d = D.Descriptor.frombytes(f.read())
    inner = D.Descriptor(name="ü", classID=b"longClassID")
    inner[b"Rd  "] = D.Double(0.5)
    d[Key.Enabled] = D.Bool(True)
    d[D._ZeroLengthKey(b"abcd")] = D.UnitFloat(unit=Unit.Pixels, value=2.5)
    d[b"enum"] = D.Enumerated(Type.BlendMode, Enum.Normal)
    d[b"list"] = _list(D.List, D.Integer(-3), D.String("a\U0001f600"), inner)
    d[b"big"] = D.LargeInteger(2**40)
    d[b"glob"] = D.GlobalObject(name="g", classID=Klass.Color.value)
    d[b"objc"] = D.ObjectArray(items_count=1, name="o")
    d[b"raw"] = D.RawData(b"\x00\x01\x02")
    d[b"engn"] = D.RawData(_engine_data())  # type: ignore[arg-type]
    d[b"ref "] = _list(D.Reference, D.Name("n", b"Lyr ", "v"), D.Identifier(7))
    d[b"ufls"] = D.UnitFloats(unit=Unit.Points, values=[1.0, 2.0])
    d[b"offs"] = D.Offset("x", b"Lyr ", 2)
    d[b"ints"] = _list(D.List, D.Integer(-(2**31)), D.LargeInteger(-(2**63)))
    d._items.move_to_end(next(iter(d._items)))
    deep = d
    for _ in range(120):
        deep[b"Nest"] = D.Descriptor()
        deep = deep[b"Nest"]
    return d


@pytest.mark.parametrize(
    "case",
    ["descriptor", "block", "block2", "engine", "engine2", "exact", "edited text"],
)
def test_fast_writers_match_the_plain_ones(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    def write() -> bytes:
        f = io.BytesIO()
        f.write(b"\x00" * 3)  # at an odd offset
        if case == "descriptor":
            _descriptor().write(f)
        elif case in ("block", "block2"):
            kls = D.DescriptorBlock if case == "block" else D.DescriptorBlock2
            block = kls(name="b", classID=b"null", items=_descriptor().items())
            block.write(f)
        elif case == "engine":
            _engine_data().write(f)
        elif case == "engine2":
            E.EngineData2(_engine_data().items()).write(f)
        elif case == "exact":
            _exact().write(f)
        else:
            psd = PSDImage.open(full_name("text-story.psd"))
            layer = [x for x in psd.descendants() if isinstance(x, TypeLayer)][-1]
            layer.set_text("(1CD) é \\ 1.5x (2U)")
            psd.save(f)
        return f.getvalue()

    compiled, fast, plain = _fast_and_plain(monkeypatch, write)
    assert compiled == fast == plain


def test_compiled_writer_takes_exact_trees() -> None:
    _engine = pytest.importorskip("psd_tools.psd._engine")
    assert E._WRITER and D._WRITER
    assert _engine.dumps(_exact(), 0, True) is not None
    assert _engine.dumps(_exact(120), 0, True) is None  # Python's to write
    assert _engine.dumps(_engine_data(), 0, True) is None  # subclasses
