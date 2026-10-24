"""The compiled engine data and descriptor readers against the Python ones.

Each case is read twice, with the compiled reader and with it switched off,
and the two must agree on the elements (their exact classes and fields), the
parse budget left behind, the log records and any exception.
"""

import io
import logging
import os
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import psd_tools.psd._descriptor as _descriptor  # type: ignore[import-not-found]
import psd_tools.psd._engine as _engine  # type: ignore[import-not-found]
import pytest

from psd_tools import PSDImage
from psd_tools.psd import descriptor, engine_data
from psd_tools.psd.descriptor import Descriptor, DescriptorBlock, UnitFloat
from psd_tools.psd.engine_data import EngineData, EngineData2
from psd_tools.psd.parse_limits import ParseLimits, current_budget, parse_context
from psd_tools.terminology import Enum, Key, Klass, Unit

from ..utils import TEST_ROOT, all_files

LIMITS = [
    None,
    ParseLimits(max_objects=6),
    ParseLimits(max_read_bytes=5),
    ParseLimits(max_total_bytes=40),
    ParseLimits(max_nesting_depth=2),
    ParseLimits(max_nesting_depth=None, max_objects=None),
]


def tree(value: Any) -> Any:
    """``value`` as nested tuples of exact classes, fields and reprs."""
    kind = type(value)
    fields = getattr(kind, "__attrs_attrs__", None)
    if fields is not None:
        extra = sorted(vars(value).items()) if getattr(value, "__dict__", None) else []
        return (
            kind,
            tuple((f.name, tree(getattr(value, f.name))) for f in fields),
            tree(extra),
        )
    if isinstance(value, dict):
        return (kind, tuple((tree(k), tree(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (kind, tuple(tree(v) for v in value))
    return (kind, repr(value))


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.rows: list[tuple[Any, ...]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.rows.append(
            (record.name, record.levelno, record.getMessage(), record.funcName)
        )


def outcome(read: Callable[[], Any], limits: ParseLimits | None) -> Any:
    """What ``read()`` gives in a fresh budget: result or error, budget, logs."""
    records = _Records()
    logger = logging.getLogger("psd_tools")
    level = logger.level
    logger.addHandler(records)
    logger.setLevel(logging.DEBUG)
    result: Any = None
    state: Any = None
    try:
        with parse_context(limits):
            budget = current_budget()
            assert budget is not None
            try:
                result = ("ok", tree(read()))
            except Exception as e:
                result = ("error", type(e), str(e))
            error = None if budget.error is None else str(budget.error)
            state = (budget.total_bytes, budget.objects, budget.depth, error)
        closed = None
    except Exception as e:
        closed = (type(e), str(e))
    finally:
        logger.removeHandler(records)
        logger.setLevel(level)
    return result, state, closed, records.rows


def both(
    monkeypatch: pytest.MonkeyPatch,
    read: Callable[[], Any],
    limits: ParseLimits | None = None,
) -> tuple[Any, Any]:
    compiled = outcome(read, limits)
    with monkeypatch.context() as m:
        m.setattr(engine_data, "_engine", None)
        m.setattr(descriptor, "_descriptor", None)
        python = outcome(read, limits)
    return compiled, python


ENGINE_CASES = [
    b"",
    b"<< >>",
    b"<< /A 1 >>",
    b"\n\n<<\n\t/A 1\n\t/B -2.5\n\t/C true\n\t/D (\xfe\xff\x00a)\n\t/E [ 1 2 .5 ]"
    b"\n\t/F << /G /H >>\n>>",
    b"<< /T (\xfe\xff\x00\\(\x00\\)\x00\\\\) >>",
    b"<< /T (\xfe\xff\x00\\\\) /U 1 >> >>",
    b"<< /T (\xfe\xff\x00a\n\x00b) >>",
    b"<< /T (\xfe\xff) >>",
    b"<< /T (\xfe\xff\x00) >>",
    b"<< /T (\xfe\xff\xd8\x00) >>",
    b"<< /A (\xfe\xff\x00a",
    b"<< /A (hwid) /B --(.-0 /C (a1) >>",
    b"<< /A 1 >>\x00\x00",
    b"<< /A 1 >>\x00x",
    b"<< /A 1x >>",
    b"<< /A @ >>",
    b"<< /A 1. >>",
    b"<< /A - >>",
    b"<< /A",
    b"<< /A ] >>",
    b"<< /A >> >>",
    b"<< /A [ 1 >> ] >>",
    b"<< /A [ 1 [ 2 [ ] ] << /B 3 >> ] >>",
    b"<< /A [ 1",
    b"<< 1 (\xfe\xff) [ << /A 1 >> /B 2 >>",
    b"<< /A 123456789012345678901234567890 /B -9223372036854775809 >>",
    b"<< /A 007 /B -0 /C -.0 /D 00.10 /E 999999999999999999 >>",
    b"<< /A 0.1 /B 1.7976931348623157 /C 123456789012345678.5 /D .000001 >>",
    b"<< /A_1 /B_2 /C true /D false /E truex >>",
    b"<< /A 1 /B 2 /A 3 >>",
    b"<< /A 1\r >>",
    b"<<\t/A\t1\n/B\n2 >> /C 3",
    b"<< /A [ " + b"<< /B " * 70 + b"1" + b" >>" * 70 + b" ] >>",
    b"<< /A " * 150 + b"1" + b" >>" * 150,
]


def _engine_files() -> list[str]:
    root = os.path.join(TEST_ROOT, "engine_data")
    return [os.path.join(root, name) for name in sorted(os.listdir(root))]


@pytest.mark.parametrize("limits", LIMITS)
@pytest.mark.parametrize("data", ENGINE_CASES)
@pytest.mark.parametrize("kind", [EngineData, EngineData2, engine_data.Dict])
def test_engine_data_cases(
    monkeypatch: pytest.MonkeyPatch,
    kind: type[engine_data.Dict],
    data: bytes,
    limits: ParseLimits | None,
) -> None:
    compiled, python = both(monkeypatch, lambda: kind.frombytes(data), limits)
    assert compiled == python


@pytest.mark.parametrize("filename", _engine_files())
def test_engine_data_files(monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    with open(filename, "rb") as f:
        data = f.read()
    kind = EngineData2 if os.path.basename(filename).startswith("Txt2") else EngineData
    compiled, python = both(monkeypatch, lambda: kind.frombytes(data))
    assert compiled == python
    assert compiled[0][0] == "ok"


def _cut(data: bytes, index: int) -> Iterator[bytes]:
    yield data[:index]
    for byte in (0x00, 0x20, 0x29, 0x2F, 0x3E, 0x5C, 0xFF):
        yield data[:index] + bytes([byte]) + data[index + 1 :]


ENGINE_MUTATED = (
    b"\n\n<<\n\t/A 1\n\t/B -2.5\n\t/C [ (\xfe\xff\x00a\\)) true << /D .5 >> ]"
    b"\n\t/E << /F (hwid) /G [ 1 2 ] >>\n>>\x00"
)


@pytest.mark.parametrize("index", range(len(ENGINE_MUTATED)))
def test_engine_data_mutations(monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    for data in _cut(ENGINE_MUTATED, index):
        for limits in (None, ParseLimits(max_objects=8)):
            compiled, python = both(
                monkeypatch, partial(EngineData.frombytes, data), limits
            )
            assert compiled == python, data


def test_every_element_is_built_compiled() -> None:
    assert {
        engine_data.Dict,
        EngineData,
        EngineData2,
    } == engine_data._COMPILED_ROOTS
    assert {ostype.value for ostype in descriptor.TYPES} == descriptor._COMPILED_OSTYPES


def test_engine_data_is_read_compiled(monkeypatch: pytest.MonkeyPatch) -> None:
    with open(_engine_files()[0], "rb") as f:
        data = f.read()
    with parse_context():
        assert _engine.parse(EngineData2, data, 64) is not None

    def refuse(self: Any) -> Any:
        raise AssertionError("the Python tokenizer ran")

    monkeypatch.setattr(engine_data.Tokenizer, "__next__", refuse)
    assert len(EngineData2.frombytes(data)) > 0


def build(kind: type[Any], items: list[tuple[Any, Any]], **fields: Any) -> Any:
    """A descriptor of ``kind`` holding ``items``."""
    element = kind(**fields)
    for key, value in items:
        element[key] = value
    return element


def listing(kind: type[Any], items: list[Any]) -> Any:
    element = kind()
    element.extend(items)
    return element


EVERY = build(
    Descriptor,
    [
        (Key.Name, descriptor.String("n\u00e9")),
        (b"kind", descriptor.Enumerated(Klass.Color.value, Enum.Red.value)),
        (b"gs99", descriptor.Bool(True)),
        (Key.Opacity, UnitFloat(unit=Unit.Percent, value=55.5)),
        (b"flts", descriptor.UnitFloats(unit=Unit.Pixels, values=[1.0, -2.5])),
        (b"dbl ", descriptor.Double(0.1)),
        (b"int ", descriptor.Integer(-7)),
        (b"long", descriptor.LargeInteger(-(2**40))),
        (b"raw ", descriptor.RawData(b"\x00\x01")),
        (b"alis", descriptor.Alias(b"alias")),
        (b"path", descriptor.Path(b"")),
        (b"clss", descriptor.Class1("c", Klass.Layer.value)),
        (b"cls2", descriptor.Class2("", b"longer class")),
        (b"cls3", descriptor.Class3("", Klass.Layer.value)),
        (b"prop", descriptor.Property("p", Klass.Layer.value, Key.Name.value)),
        (b"enmr", descriptor.EnumeratedReference("", b"Lyr ", b"Ordn", b"Trgt")),
        (b"rele", descriptor.Offset("", Klass.Layer.value, 3)),
        (b"idnt", descriptor.Identifier(5)),
        (b"indx", descriptor.Index(6)),
        (b"name", descriptor.Name("", Klass.Layer.value, "v")),
        (
            b"list",
            listing(descriptor.List, [descriptor.Integer(1), Descriptor(name="in")]),
        ),
        (b"obj ", listing(descriptor.Reference, [descriptor.Index(2)])),
        (b"GlbO", build(descriptor.GlobalObject, [(b"a", descriptor.Bool(False))])),
        (
            b"ObAr",
            build(
                descriptor.ObjectArray,
                [(b"x", descriptor.UnitFloats(Unit.Pixels, [1.0, 2.0]))],
                items_count=2,
            ),
        ),
    ],
)


def _descriptor_bytes() -> list[bytes]:
    root = os.path.join(TEST_ROOT, "descriptors")
    datas = []
    for name in sorted(os.listdir(root)):
        with open(os.path.join(root, name), "rb") as f:
            datas.append(f.read())
    return [*datas, EVERY.tobytes()]


DESCRIPTORS = _descriptor_bytes()


@pytest.mark.parametrize("limits", LIMITS)
@pytest.mark.parametrize("data", DESCRIPTORS)
def test_descriptor_files(
    monkeypatch: pytest.MonkeyPatch, data: bytes, limits: ParseLimits | None
) -> None:
    compiled, python = both(monkeypatch, lambda: Descriptor.frombytes(data), limits)
    assert compiled == python
    if limits is None:
        assert compiled[0][0] == "ok"


@pytest.mark.parametrize("data", DESCRIPTORS)
def test_descriptor_mutations(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    for index in range(len(data)):
        for cut in _cut(data, index):
            compiled, python = both(monkeypatch, partial(Descriptor.frombytes, cut))
            assert compiled == python, (index, cut[index : index + 1])


def test_descriptor_unknown_term_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    key = descriptor._ZeroLengthKey(b"Gcls")
    body = build(Descriptor, [(key, descriptor.Integer(1))]).tobytes()
    compiled, python = both(monkeypatch, lambda: Descriptor.frombytes(body))
    assert compiled == python
    assert ("psd_tools.psd.descriptor", logging.DEBUG, "Unknown term: b'Gcls'") in [
        row[:3] for row in compiled[3]
    ]


def test_descriptor_unit_from_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        build(Descriptor, [(b"x", UnitFloat(unit=Unit.Pixels, value=1.0))])
        .tobytes()
        .replace(Unit.Pixels.value, Enum.RulerCm.value)
    )
    compiled, python = both(monkeypatch, lambda: Descriptor.frombytes(body))
    assert compiled == python
    assert any("Using Enum for Unit field" in row[2] for row in compiled[3])


def test_descriptor_block_in_a_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    data = build(DescriptorBlock, [(b"a", descriptor.Integer(3))]).tobytes()

    def read() -> Any:
        fp = io.BytesIO(b"pad!" + data + b"tail")
        fp.seek(4)
        block = DescriptorBlock.read(fp)
        return block, fp.tell(), fp.read()

    compiled, python = both(monkeypatch, read)
    assert compiled == python
    assert compiled[0][0] == "ok"


def test_descriptor_is_read_compiled(monkeypatch: pytest.MonkeyPatch) -> None:
    data = DESCRIPTORS[-1]
    with parse_context():
        assert _descriptor.read(data, 0, 64) is not None
    monkeypatch.setattr(descriptor, "read_length_and_key", None)
    assert len(Descriptor.frombytes(data)) == len(EVERY)


@pytest.mark.parametrize(
    "filename", [f for f in all_files() if os.path.getsize(f) < 4_000_000]
)
def test_documents(monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    def read() -> Any:
        with open(filename, "rb") as f:
            return PSDImage.open(f, lazy=True)._record

    compiled, python = both(monkeypatch, read)
    assert compiled == python
