"""The descriptor and engine data writers' fast paths write what the plain ones do."""

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
) -> tuple[bytes, bytes]:
    fast = write()
    with monkeypatch.context() as m:
        m.setattr(D, "_FAST_BODIES", frozenset())
        m.setattr(E, "_FAST_WRITERS", frozenset())
        plain = write()
    return fast, plain


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


def _engine_data() -> E.Dict:
    with open(os.path.join(TEST_ROOT, "engine_data", "TySh_1.dat"), "rb") as f:
        data = E.EngineData.frombytes(f.read())
    data["EngineDict"]["Editor"]["Text"] = E.String("é (x) \\ 😀\r")
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
    return d


@pytest.mark.parametrize(
    "case",
    ["descriptor", "block", "block2", "engine", "engine2", "edited text"],
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
        else:
            psd = PSDImage.open(full_name("text-story.psd"))
            layer = [x for x in psd.descendants() if isinstance(x, TypeLayer)][-1]
            layer.set_text("(1CD) é \\ 1.5x (2U)")
            psd.save(f)
        return f.getvalue()

    fast, plain = _fast_and_plain(monkeypatch, write)
    assert fast == plain
