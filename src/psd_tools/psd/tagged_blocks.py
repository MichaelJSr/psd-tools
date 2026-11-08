"""
Tagged block data structure.

.. todo:: Support the following tagged blocks: ``Tag.PATTERN_DATA``,
    ``Tag.TYPE_TOOL_INFO``, ``Tag.LAYER``,
    ``Tag.ALPHA``

"""

import io
import logging
import struct
from typing import IO, Any, ClassVar, Self

from attrs import converters, define, field, validators

from psd_tools.constants import (
    BlendMode,
    PlacedLayerType,
    SectionDivider,
    SheetColorType,
    Tag,
)
from psd_tools.psd.adjustments import ADJUSTMENT_TYPES
from psd_tools.psd.base import (
    BaseElement,
    ByteElement,
    DictElement,
    EmptyElement,
    IntegerElement,
    ListElement,
    StringElement,
    ValueElement,
)
from psd_tools.psd.bin_utils import (
    bounded_reader,
    is_readable,
    pad,
    read_exact,
    read_fmt,
    read_length_block,
    read_padding,
    read_pascal_string,
    trimmed_repr,
    write_bytes,
    write_fmt,
    write_length_block,
    write_padding,
    write_pascal_string,
)
from psd_tools.psd.color import Color
from psd_tools.psd.descriptor import DescriptorBlock, DescriptorBlock2
from psd_tools.psd.effects_layer import EffectsLayer
from psd_tools.psd.engine_data import EngineData, EngineData2
from psd_tools.psd.filter_effects import FilterEffects
from psd_tools.psd.lazy import Span, defer
from psd_tools.psd.linked_layer import LinkedLayers
from psd_tools.psd.parse_limits import (
    ParseLimitError,
    consume_bytes,
    current_budget,
    parse_context,
    resume_budget,
)
from psd_tools.psd.patterns import Patterns
from psd_tools.psd.vector import VectorMaskSetting, VectorStrokeContentSetting
from psd_tools.registry import new_registry
from psd_tools.validators import in_

logger = logging.getLogger(__name__)


TYPES, register = new_registry()

# Tag(key) without the enum call, for a known key.
_TAGS: dict[bytes, Tag] = {member.value: member for member in Tag}

# Layer effects blocks: Photoshop writes them at version 0 (set_data).
_EFFECT_BLOCKS = frozenset(
    {
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO,
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0,
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1,
    }
)

TYPES.update(ADJUSTMENT_TYPES)
TYPES.update(
    {
        Tag.ANIMATION_EFFECTS: DescriptorBlock,
        Tag.ARTBOARD_DATA1: DescriptorBlock,
        Tag.ARTBOARD_DATA2: DescriptorBlock,
        Tag.ARTBOARD_DATA3: DescriptorBlock,
        Tag.BLEND_CLIPPING_ELEMENTS: ByteElement,
        Tag.BLEND_FILL_OPACITY: ByteElement,
        Tag.BLEND_INTERIOR_ELEMENTS: ByteElement,
        Tag.COMPOSITOR_INFO: DescriptorBlock,
        Tag.CONTENT_GENERATOR_EXTRA_DATA: DescriptorBlock,
        Tag.EFFECTS_LAYER: EffectsLayer,
        Tag.EXPORT_SETTING1: DescriptorBlock,
        Tag.EXPORT_SETTING2: DescriptorBlock,
        Tag.FILTER_EFFECTS1: FilterEffects,
        Tag.FILTER_EFFECTS2: FilterEffects,
        Tag.FILTER_EFFECTS3: FilterEffects,
        Tag.FRAMED_GROUP: DescriptorBlock,
        Tag.KNOCKOUT_SETTING: ByteElement,
        Tag.LINKED_LAYER1: LinkedLayers,
        Tag.LINKED_LAYER2: LinkedLayers,
        Tag.LINKED_LAYER3: LinkedLayers,
        Tag.LINKED_LAYER_EXTERNAL: LinkedLayers,
        Tag.LAYER_ID: IntegerElement,
        Tag.LAYER_MASK_AS_GLOBAL_MASK: ByteElement,
        Tag.UNICODE_LAYER_NAME: StringElement,
        Tag.LAYER_VERSION: IntegerElement,
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO: DescriptorBlock2,
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0: DescriptorBlock2,
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1: DescriptorBlock2,
        Tag.PATTERNS1: Patterns,
        Tag.PATTERNS2: Patterns,
        Tag.PATTERNS3: Patterns,
        Tag.PATT: EmptyElement,
        Tag.PIXEL_SOURCE_DATA1: DescriptorBlock,
        Tag.SAVING_MERGED_TRANSPARENCY: EmptyElement,
        Tag.SAVING_MERGED_TRANSPARENCY16: EmptyElement,
        Tag.SAVING_MERGED_TRANSPARENCY32: EmptyElement,
        Tag.TEXT_ENGINE_DATA: EngineData2,
        Tag.TRANSPARENCY_SHAPES_LAYER: ByteElement,
        Tag.UNICODE_PATH_NAME: DescriptorBlock,
        Tag.USING_ALIGNED_RENDERING: IntegerElement,
        Tag.VECTOR_MASK_AS_GLOBAL_MASK: ByteElement,
        Tag.VECTOR_MASK_SETTING1: VectorMaskSetting,
        Tag.VECTOR_MASK_SETTING2: VectorMaskSetting,
        Tag.VECTOR_ORIGINATION_DATA: DescriptorBlock2,
        Tag.VECTOR_ORIGINATION_UNKNOWN: IntegerElement,
        Tag.VECTOR_STROKE_DATA: DescriptorBlock,
        Tag.VECTOR_STROKE_CONTENT_DATA: VectorStrokeContentSetting,
        # Unknown tags.
        # Tag.CAI: DescriptorBlock2,
        Tag.GENI: DescriptorBlock,
        Tag.OCIO: DescriptorBlock,
    }
)


@define(repr=False)
class TaggedBlocks(DictElement):
    """
    Dict of tagged block items.

    See :py:class:`~psd_tools.constants.Tag` for available keys.

    Example::

        from psd_tools.constants import Tag

        # Iterate over fields
        for key in tagged_blocks:
            print(key)

        # Get a field
        value = tagged_blocks.get_data(Tag.TYPE_TOOL_OBJECT_SETTING)
    """

    def get_data(self, key: Any, default: Any = None) -> Any:
        """
        Get data from the tagged blocks.

        Shortcut for the following::

            if key in tagged_blocks:
                value = tagged_blocks[key].data

        A block that has a handler but failed to parse keeps its raw bytes
        (see ``tagged_blocks[key].data``) and reads as missing here.
        """
        if key in self:
            value = self[key].data
            if isinstance(value, bytes) and key in TYPES:
                return default
            if isinstance(value, ValueElement):
                return value.value
            else:
                return value
        return default

    def set_data(self, key: Any, *args: Any, **kwargs: Any) -> None:
        """
        Set data for the given key.

        Shortut for the following::

            key = getattr(Tag, key)
            kls = TYPES.get(key)
            self[key] = TaggedBlocks(key=key, data=kls(value))

        A value that already is the key's type is stored as it is, not
        copied (its version and name kept); a new effects block (``lfx2``, ``lmfx``,
        ``lfxs``) takes Photoshop's version 0 and name, as Photoshop 2023
        refuses to read a layer's effects at any other version.
        """
        key = self._key_converter(key)
        kls = TYPES.get(key)
        if kls is None:
            return
        if len(args) == 1 and not kwargs and type(args[0]) is kls:
            data = args[0]
        else:
            if key in _EFFECT_BLOCKS:
                kwargs.setdefault("version", 0)
                kwargs.setdefault("name", "\x00")
            data = kls(*args, **kwargs)
        signature = self[key].signature if key in self else b"8BIM"
        self[key] = TaggedBlock(signature=signature, key=key, data=data)

    @classmethod
    def read(
        cls,
        fp: IO[bytes],
        version: int = 1,
        padding: int = 1,
        end_pos: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Read the structure from a file-like object."""
        if end_pos is not None:
            with bounded_reader(fp, end_pos - fp.tell()) as body:
                return cls.read(body, version=version, padding=padding, **kwargs)
        items: list[tuple[Any, TaggedBlock]] = []
        if version == 1 and type(fp) is io.BytesIO and not _hold(fp, items, padding):
            return cls(items)  # type: ignore[arg-type]
        while is_readable(fp, 8):  # len(signature) + len(key) = 8
            block = TaggedBlock.read(fp, version, padding)
            if block is None:
                break
            items.append((block.key, block))
        return cls(items)  # type: ignore[arg-type]

    @classmethod
    def _key_converter(cls, key: Any) -> Any:
        return getattr(key, "value", key)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            return

        with p.group(2, "{", "}"):
            p.breakable("")
            for idx, key in enumerate(self._items):
                if idx:
                    p.text(",")
                    p.breakable()
                value = self._items[key]
                try:
                    p.text(Tag(key).name)
                except ValueError:
                    p.pretty(key)
                p.text(": ")
                if isinstance(value.data, bytes):
                    p.text(trimmed_repr(value.data))
                else:
                    p.pretty(value.data)
            p.breakable("")


@define(repr=False, eq=False)
class TaggedBlock(BaseElement):
    """
    Layer tagged block with extra info.

    .. py:attribute:: key

        4-character code. See :py:class:`~psd_tools.constants.Tag`

    .. py:attribute:: data

        Data. A lazy read (:py:mod:`psd_tools.psd.lazy`) leaves the payload
        blocks (``_LAZY_KEYS``: embedded smart object files, smart filter
        pixels) in the file and reads and parses one on first access. A
        layer's blocks are kept as bytes and parsed on first access; one
        never read is written back as it was read.
    """

    _SIGNATURES = (b"8BIM", b"8B64")
    _BIG_KEYS: ClassVar[set[Tag]] = {
        Tag.USER_MASK,
        Tag.LAYER_16,
        Tag.LAYER_32,
        Tag.LAYER,
        Tag.SAVING_MERGED_TRANSPARENCY16,
        Tag.SAVING_MERGED_TRANSPARENCY32,
        Tag.SAVING_MERGED_TRANSPARENCY,
        Tag.SAVING_MERGED_TRANSPARENCY16,
        Tag.ALPHA,
        Tag.FILTER_MASK,
        Tag.LINKED_LAYER2,
        Tag.LINKED_LAYER3,
        Tag.LINKED_LAYER_EXTERNAL,
        Tag.FILTER_EFFECTS1,
        Tag.FILTER_EFFECTS2,
        Tag.FILTER_EFFECTS3,
        Tag.PIXEL_SOURCE_DATA2,
        Tag.UNICODE_PATH_NAME,
        Tag.EXPORT_SETTING1,
        Tag.EXPORT_SETTING2,
        Tag.COMPOSITOR_INFO,
        Tag.ARTBOARD_DATA2,
    }

    # The layer tree cannot be built without these, so a parse failure raises
    # instead of falling back to raw bytes.
    _STRUCTURAL: ClassVar[set[Tag]] = {
        Tag.LAYER_16,
        Tag.LAYER_32,
        Tag.SECTION_DIVIDER_SETTING,
        Tag.NESTED_SECTION_DIVIDER_SETTING,
    }

    # Payloads nothing reads while a document opens: read on first access.
    _LAZY_KEYS: ClassVar[set[Tag]] = {
        Tag.LINKED_LAYER1,
        Tag.LINKED_LAYER2,
        Tag.LINKED_LAYER3,
        Tag.LINKED_LAYER_EXTERNAL,
        Tag.FILTER_EFFECTS1,
        Tag.FILTER_EFFECTS2,
        Tag.FILTER_EFFECTS3,
    }

    # Keys whose payload can dominate the file: an embedded smart object is
    # read from a bounded substream instead of a materialized copy of the
    # whole block (#899). The set is deliberately not _BIG_KEYS, which
    # selects the length-field width and covers descriptor-backed tags this
    # has no business streaming.
    _STREAMED_KEYS: ClassVar[set[Tag]] = {
        Tag.LINKED_LAYER1,
        Tag.LINKED_LAYER2,
        Tag.LINKED_LAYER3,
        Tag.LINKED_LAYER_EXTERNAL,
    }

    signature: bytes = field(default=b"8BIM", repr=False, validator=in_(_SIGNATURES))
    key: bytes = b""
    _data: Any = field(default=b"", repr=True)
    _span: "tuple[Span, int] | None" = field(default=None, init=False)
    _held: "tuple[bytes, int, Any] | None" = field(default=None, init=False, repr=False)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if self._held is not None:
            self._load()  # show the data, as an eager read does
        super()._repr_pretty_(p, cycle)

    def _load(self) -> None:
        if self._held is not None:
            raw, _, budget = self._held
            with resume_budget(budget):
                self._data = self._parse(self.key, raw, 1)
            self._held = None
        elif self._span is not None:
            span, version = self._span
            self._data, self._span = self._parse(self.key, span.read(), version), None

    @property
    def data(self) -> Any:
        """The parsed block, or its bytes."""
        self._load()
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data, self._span, self._held = value, None, None

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, TaggedBlock)
        return (self.signature, self.key, self.data) == (
            other.signature,
            other.key,
            other.data,
        )

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def read(
        cls,
        fp: IO[bytes],
        version: int = 1,
        padding: int = 1,
        **kwargs: Any,
    ) -> Self:  # type: ignore[return]
        """Read the structure from a file-like object."""
        signature = read_fmt("4s", fp)[0]
        if signature not in cls._SIGNATURES:
            logger.warning(f"Invalid signature ({signature!r})")
            fp.seek(-4, 1)
            return None  # type: ignore[return-value]

        key = read_fmt("4s", fp)[0]
        try:
            key = _TAGS[key] if key in _TAGS else Tag(key)
        except ValueError:
            message = f"Unknown key: {key!r}"
            logger.warning(message)

        fmt = cls._length_format(key, version)
        if key in cls._LAZY_KEYS:
            marker = fp.tell()
            length = read_fmt(fmt, fp)[0]
            span = defer(fp, length)
            if span is not None:
                read_padding(fp, length, padding)
                self = cls(signature, key)
                self._span = (span, version)
                return self
            kls = TYPES.get(key)
            if kls is not None and key in cls._STREAMED_KEYS:
                fp.seek(marker)
                data = cls._read_streamed(fp, key, kls, fmt, version, padding)
                return cls(signature, key, data)
            raw_data = read_exact(fp, length)
            read_padding(fp, length, padding)
            if kls is not None and version == 1:
                return cls._unparsed(signature, key, raw_data, padding)
        else:
            raw_data = read_length_block(fp, fmt=fmt, padding=padding)
            if key is Tag.TEXT_ENGINE_DATA and version == 1:
                # Only text edits read the document's text engine data.
                return cls._unparsed(signature, key, raw_data, padding)
        return cls(signature, key, cls._parse(key, raw_data, version))

    @classmethod
    def _unparsed(cls, signature: bytes, key: Any, raw: bytes, padding: int) -> Self:
        """Keep ``raw`` as it is until read, then parse it on the open's budget."""
        self = cls(signature, key)
        self._held = (raw, padding, current_budget())
        return self

    @classmethod
    def _parse(cls, key: Any, raw_data: bytes, version: int) -> Any:
        """Parse the bytes with the key's handler; keep the bytes without one."""
        kls = TYPES.get(key)
        if kls:
            try:
                return kls.frombytes(raw_data, version=version)
            except ParseLimitError:
                raise
            except (OSError, ValueError) as e:
                if key in cls._STRUCTURAL:
                    raise
                # Fallback to raw data.
                message = f"Failed to read tagged block {key!r}: {e}"
                logger.error(message)
                return raw_data
        message = f"Unknown tagged block: {key!r}, {trimmed_repr(raw_data)}"
        logger.debug(message)
        return raw_data

    @classmethod
    def _read_streamed(
        cls,
        fp: IO[bytes],
        key: Any,
        kls: type[BaseElement],
        fmt: str,
        version: int,
        padding: int,
    ) -> Any:
        """Read a streamed key's payload without materializing it first.

        The handler parses from a bounded substream over the live stream, so
        the only full-size copy alive is the one it retains (#899). A parse
        that fails, or that does not consume the window exactly, falls back
        to the raw bytes like any other block; a truncated payload raises
        before anything is read.
        """
        marker = fp.tell()
        length = read_fmt(fmt, fp)[0]
        payload = fp.tell()
        if not is_readable(fp, length):
            raise OSError("Invalid data section size: %d" % length)
        try:
            with parse_context():
                consume_bytes(length)
                with bounded_reader(fp, length) as body:
                    data = kls.read(body, version=version)
                    consumed = body.tell() - payload
                if consumed != length:
                    del data  # free the payload before the raw fallback reads it
                    raise ValueError("consumed %d of %d bytes" % (consumed, length))
        except ParseLimitError:
            # Leave the stream where the whole-block read would have.
            fp.seek(payload + length)
            if is_readable(fp, pad(length, padding) - length):
                read_padding(fp, length, padding)
            raise
        except (OSError, ValueError) as e:
            if key in cls._STRUCTURAL:
                raise
            message = f"Failed to read tagged block {key!r}: {e}"
            logger.error(message)
            fp.seek(marker)
            return read_length_block(fp, fmt=fmt, padding=padding)
        read_padding(fp, length, padding)
        return data

    def write(
        self, fp: IO[bytes], version: int = 1, padding: int = 1, **kwargs: Any
    ) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        key = self.key if isinstance(self.key, bytes) else self.key.value
        written = write_fmt(fp, "4s4s", self.signature, key)

        def writer(f: IO[bytes]) -> int:
            held = self._held
            if held is not None and version == 1 and held[1] == padding:
                return write_bytes(f, held[0])
            if hasattr(self.data, "write"):
                # It seems padding size applies at the block level here.
                inner_padding = 1 if padding == 4 else 4
                return self.data.write(f, padding=inner_padding, version=version)
            return write_bytes(f, self.data)

        fmt = self._length_format(self.key, version)
        written += write_length_block(fp, writer, fmt=fmt, padding=padding)
        return written

    @classmethod
    def _length_format(cls, key: Any, version: int) -> str:
        return ("I", "Q")[int(version == 2 and key in cls._BIG_KEYS)]


@register(Tag.ANNOTATIONS)
@define(repr=False)
class Annotations(ListElement):
    """
    List of Annotation, see :py:class: `.Annotation`.

    .. py:attribute:: major_version
    .. py:attribute:: minor_version
    """

    major_version: int = 2
    minor_version: int = 1

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "Annotations":
        """Read the structure from a file-like object."""
        major_version, minor_version, count = read_fmt("2HI", fp)
        items = []
        for _ in range(count):
            length = read_fmt("I", fp)[0] - 4
            if length < 0:
                raise OSError("Invalid annotation length: %d" % (length + 4))
            if length > 0:
                with io.BytesIO(read_exact(fp, length)) as f:
                    items.append(Annotation.read(f))
        return cls(
            major_version=major_version,
            minor_version=minor_version,
            items=items,  # type: ignore[arg-type]
        )

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(
            fp, "2HI", self.major_version, self.minor_version, len(self)
        )
        for item in self:
            data = item.tobytes()
            written += write_fmt(fp, "I", len(data) + 4)
            written += write_bytes(fp, data)
        written += write_padding(fp, written, 4)
        return written


@define(repr=False)
class Annotation(BaseElement):
    """
    Annotation structure.

    .. py:attribute:: kind
    .. py:attribute:: is_open
    """

    kind: bytes = field(default=b"txtA", validator=in_((b"txtA", b"sndM")))
    is_open: int = 0
    flags: int = 0
    optional_blocks: int = 1
    icon_location: list[int] = field(factory=lambda: [0, 0, 0, 0], converter=list)
    popup_location: list[int] = field(factory=lambda: [0, 0, 0, 0], converter=list)
    color: Color = field(factory=Color)
    author: str = ""
    name: str = ""
    mod_date: str = ""
    marker: bytes = field(default=b"txtC", validator=in_((b"txtC", b"sndM")))
    data: bytes = b""

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "Annotation":
        """Read the structure from a file-like object."""
        kind, is_open, flags, optional_blocks = read_fmt("4s2BH", fp)
        icon_location = read_fmt("4i", fp)
        popup_location = read_fmt("4i", fp)
        color = Color.read(fp)
        author = read_pascal_string(fp, "macroman", padding=2)
        name = read_pascal_string(fp, "macroman", padding=2)
        mod_date = read_pascal_string(fp, "macroman", padding=2)
        _length, marker = read_fmt("I4s", fp)
        data = read_length_block(fp)
        return cls(
            kind,
            is_open,
            flags,
            optional_blocks,
            icon_location,
            popup_location,
            color,
            author,
            name,
            mod_date,
            marker,
            data,
        )

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(
            fp, "4s2BH", self.kind, self.is_open, self.flags, self.optional_blocks
        )
        written += write_fmt(fp, "4i", *self.icon_location)
        written += write_fmt(fp, "4i", *self.popup_location)
        written += self.color.write(fp)
        written += write_pascal_string(fp, self.author, "macroman", padding=2)
        written += write_pascal_string(fp, self.name, "macroman", padding=2)
        written += write_pascal_string(fp, self.mod_date, "macroman", padding=2)
        written += write_fmt(fp, "I4s", len(self.data) + 12, self.marker)
        written += write_length_block(fp, lambda f: write_bytes(f, self.data))
        return written


@register(Tag.FOREIGN_EFFECT_ID)
@register(Tag.LAYER_NAME_SOURCE_SETTING)
@define(repr=False, eq=False, order=False)
class Bytes(ValueElement):
    """
    Bytes structure.

    .. py:attribute:: value
    """

    value: bytes = b"\x00\x00\x00\x00"

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "Bytes":
        """Read the structure from a file-like object."""
        return cls(read_exact(fp, 4))

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        return write_bytes(fp, self.value)


@register(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING)
@define(repr=False)
class ChannelBlendingRestrictionsSetting(ListElement):
    """
    ChannelBlendingRestrictionsSetting structure.

    List of restricted channel numbers (int).
    """

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "ChannelBlendingRestrictionsSetting":
        """Read the structure from a file-like object."""
        items = []
        while is_readable(fp, 4):
            items.append(read_fmt("I", fp)[0])
        return cls(items)

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        return write_fmt(fp, "%dI" % len(self), *self._items)


@register(Tag.FILTER_MASK)
@define(repr=False)
class FilterMask(BaseElement):
    """
    FilterMask structure.

    .. py:attribute:: color
    .. py:attribute:: opacity
    """

    color: Color = field(factory=Color)
    opacity: int = 0

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "FilterMask":
        """Read the structure from a file-like object."""
        color = Color.read(fp)
        opacity = read_fmt("H", fp)[0]
        return cls(color, opacity)

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = self.color.write(fp)
        written += write_fmt(fp, "H", self.opacity)
        return written


@register(Tag.METADATA_SETTING)
class MetadataSettings(ListElement):
    """MetadataSettings structure."""

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "MetadataSettings":
        """Read the structure from a file-like object."""
        count = read_fmt("I", fp)[0]
        items = []
        for _ in range(count):
            items.append(MetadataSetting.read(fp))
        return cls(items)  # type: ignore[arg-type]

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "I", len(self))
        written += sum(item.write(fp) for item in self)
        return written


@define(repr=False)
class MetadataSetting(BaseElement):
    """MetadataSetting structure."""

    _KNOWN_KEYS: ClassVar[set[bytes]] = {
        b"cust",
        b"cmls",
        b"extn",
        b"mlst",
        b"tmln",
        b"sgrp",
    }
    _KNOWN_SIGNATURES = (b"8BIM", b"8ELE")
    signature: bytes = field(
        default=b"8BIM", repr=False, validator=in_(_KNOWN_SIGNATURES)
    )
    key: bytes = b""
    copy_on_sheet: bool = False
    data: bytes = b""

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "MetadataSetting":
        """Read the structure from a file-like object."""
        signature = read_fmt("4s", fp)[0]
        if signature not in cls._KNOWN_SIGNATURES:
            raise ValueError(f"Invalid signature {signature!r}")
        key, copy_on_sheet = read_fmt("4s?3x", fp)
        data: Any = read_length_block(fp)
        if key in (b"mdyn", b"sgrp"):
            with io.BytesIO(data) as f:
                data = read_fmt("I", f)[0]
        elif key in cls._KNOWN_KEYS:
            data = DescriptorBlock.frombytes(data, padding=4)
        else:
            message = f"Unknown metadata key {key!r}"
            logger.debug(message)
            data = data
        return cls(signature, key, copy_on_sheet, data)

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "4s4s?3x", self.signature, self.key, self.copy_on_sheet)

        def writer(f: IO[bytes]) -> int:
            if hasattr(self.data, "write"):
                return self.data.write(f, padding=4)
            elif isinstance(self.data, int):
                return write_fmt(fp, "I", self.data)
            return write_bytes(f, self.data)

        written += write_length_block(fp, writer)
        return written


@register(Tag.PIXEL_SOURCE_DATA2)
@define(repr=False)
class PixelSourceData2(ListElement):
    """PixelSourceData2 structure."""

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "PixelSourceData2":
        """Read the structure from a file-like object."""
        items = []
        while is_readable(fp, 8):
            items.append(read_length_block(fp, fmt="Q"))
        return cls(items)  # type: ignore[arg-type]

    def write(self, fp: IO[bytes], padding: int = 4, **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = 0
        for item in self:
            written += write_length_block(
                fp, lambda f, item=item: write_bytes(f, item), fmt="Q"
            )
        written += write_padding(fp, written, padding)
        return written


@register(Tag.PLACED_LAYER1)
@register(Tag.PLACED_LAYER2)
@define(repr=False)
class PlacedLayerData(BaseElement):
    """PlacedLayerData structure."""

    kind: bytes = b"plcL"
    version: int = field(default=3, validator=in_((3,)))
    uuid: bytes = b""
    page: int = 0
    total_pages: int = 0
    anti_alias: int = 0
    layer_type: PlacedLayerType = field(
        default=PlacedLayerType.UNKNOWN,
        converter=PlacedLayerType,
        validator=in_(tuple(PlacedLayerType)),
    )
    transform: tuple = (0.0,) * 8
    warp: object = None

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "PlacedLayerData":
        """Read the structure from a file-like object."""
        kind, version = read_fmt("4sI", fp)
        uuid_str = read_pascal_string(fp, "macroman", padding=1)
        page, total_pages, anti_alias, layer_type = read_fmt("4I", fp)
        transform = read_fmt("8d", fp)
        warp = DescriptorBlock2.read(fp, padding=1)
        return cls(
            kind,
            version,
            uuid_str.encode("macroman") if isinstance(uuid_str, str) else uuid_str,
            page,
            total_pages,
            anti_alias,
            layer_type,
            transform,
            warp,
        )

    def write(self, fp: IO[bytes], padding: int = 4, **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "4sI", self.kind, self.version)
        uuid_str = (
            self.uuid.decode("macroman") if isinstance(self.uuid, bytes) else self.uuid
        )
        written += write_pascal_string(fp, uuid_str, "macroman", padding=1)
        written += write_fmt(
            fp,
            "4I",
            self.page,
            self.total_pages,
            self.anti_alias,
            self.layer_type.value,
        )
        written += write_fmt(fp, "8d", *self.transform)
        written += self.warp.write(fp, padding=1)  # type: ignore[attr-defined]
        written += write_padding(fp, written, padding)
        return written


@register(Tag.PROTECTED_SETTING)
class ProtectedSetting(IntegerElement):
    """ProtectedSetting structure."""

    @property
    def transparency(self) -> bool:
        """Whether transparency is protected."""
        return bool(self.value & 0x01)

    @property
    def composite(self) -> bool:
        """Whether compositing is protected."""
        return bool(self.value & 0x02)

    @property
    def position(self) -> bool:
        """Whether position is protected."""
        return bool(self.value & 0x04)

    @property
    def nesting(self) -> bool:
        """Whether nesting is protected."""
        return bool(self.value & 0x08)

    @property
    def complete(self) -> bool:
        """Whether everything is protected."""
        return self.value == 2147483648

    def lock(self, lock_flags: Any) -> None:
        """Set the protection flags."""
        self.value = int(lock_flags)


@register(Tag.REFERENCE_POINT)
@define(repr=False)
class ReferencePoint(ListElement):
    """ReferencePoint structure."""

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "ReferencePoint":
        """Read the structure from a file-like object."""
        return cls(list(read_fmt("2d", fp)))

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        return write_fmt(fp, "2d", *self._items)


@register(Tag.SECTION_DIVIDER_SETTING)
@register(Tag.NESTED_SECTION_DIVIDER_SETTING)
@define(repr=False)
class SectionDividerSetting(BaseElement):
    """
    SectionDividerSetting structure.

    .. py:attribute:: kind
    .. py:attribute:: blend_mode
    .. py:attribute:: sub_type
    """

    kind: SectionDivider = field(
        default=SectionDivider.OTHER,
        converter=SectionDivider,
        validator=in_(tuple(SectionDivider)),
    )
    signature: bytes | None = field(default=None, repr=False)
    blend_mode: BlendMode | None = field(
        default=None,
        converter=converters.optional(BlendMode),
        validator=validators.optional(in_(tuple(BlendMode))),
    )
    sub_type: int | None = None

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "SectionDividerSetting":
        """Read the structure from a file-like object."""
        kind = SectionDivider(read_fmt("I", fp)[0])
        signature, blend_mode = None, None
        if is_readable(fp, 8):
            signature = read_fmt("4s", fp)[0]
            if signature != b"8BIM":
                raise ValueError(f"Invalid signature {signature!r}")
            blend_mode = BlendMode(read_fmt("4s", fp)[0])
        sub_type = None
        if is_readable(fp, 4):
            sub_type = read_fmt("I", fp)[0]
        return cls(kind, signature=signature, blend_mode=blend_mode, sub_type=sub_type)

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "I", self.kind.value)
        if self.blend_mode:
            if self.signature is None:
                logger.debug(
                    "Signature is missing in SectionDividerSetting, overriding"
                )
                self.signature = b"8BIM"
            written += write_fmt(fp, "4s4s", self.signature, self.blend_mode.value)
            if self.sub_type is not None:
                written += write_fmt(fp, "I", self.sub_type)
        elif self.sub_type is not None:
            logger.debug(
                "Blend mode is missing in SectionDividerSetting, ignoring sub_type"
            )
        return written


@register(Tag.SHEET_COLOR_SETTING)
@define(repr=False, eq=False, order=False)
class SheetColorSetting(ValueElement):
    """
    SheetColorSetting value.

    This setting represents color label in the layers panel in Photoshop UI.

    .. py:attribute:: value
    """

    value: SheetColorType = field(
        default=SheetColorType.NO_COLOR, converter=SheetColorType
    )

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "SheetColorSetting":
        """Read the structure from a file-like object."""
        return cls(SheetColorType(*read_fmt("H6x", fp)))

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        return write_fmt(fp, "H6x", self.value.value)


@register(Tag.SMART_OBJECT_LAYER_DATA1)
@register(Tag.SMART_OBJECT_LAYER_DATA2)
@define(repr=False)
class SmartObjectLayerData(BaseElement):
    """
    VersionedDescriptorBlock structure.

    .. py:attribute:: kind
    .. py:attribute:: version
    .. py:attribute:: data
    """

    kind: bytes = field(default=b"soLD", validator=in_((b"soLD",)))
    version: int = field(default=5, validator=in_((4, 5)))
    data: DescriptorBlock = field(factory=DescriptorBlock)

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "SmartObjectLayerData":
        """Read the structure from a file-like object."""
        kind, version = read_fmt("4sI", fp)
        data = DescriptorBlock.read(fp)
        return cls(kind, version, data)

    def write(self, fp: IO[bytes], padding: int = 4, **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "4sI", self.kind, self.version)
        written += self.data.write(fp, padding=1)
        written += write_padding(fp, written, padding)
        return written


@register(Tag.TYPE_TOOL_OBJECT_SETTING)
@define(repr=False)
class TypeToolObjectSetting(BaseElement):
    """
    TypeToolObjectSetting structure.

    .. py:attribute:: version
    .. py:attribute:: transform

        Tuple of affine transform parameters (xx, xy, yx, yy, tx, ty).

    .. py:attribute:: text_version
    .. py:attribute:: text_data
    .. py:attribute:: warp_version
    .. py:attribute:: warp
    .. py:attribute:: left
    .. py:attribute:: top
    .. py:attribute:: right
    .. py:attribute:: bottom
    """

    version: int = 1
    transform: tuple = (0.0,) * 6
    text_version: int = field(default=1, validator=in_((50,)))
    text_data: DescriptorBlock = field(factory=DescriptorBlock)
    warp_version: int = field(default=1, validator=in_((1,)))
    warp: DescriptorBlock = field(factory=DescriptorBlock)
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "TypeToolObjectSetting":
        """Read the structure from a file-like object."""
        version = read_fmt("H", fp)[0]
        transform = read_fmt("6d", fp)
        text_version = read_fmt("H", fp)[0]
        text_data = DescriptorBlock.read(fp)
        # Engine data.
        if b"EngineData" in text_data:
            try:
                engine_data = text_data[b"EngineData"].value
                engine_data = EngineData.frombytes(engine_data)
                text_data[b"EngineData"].value = engine_data
            except ParseLimitError:
                raise
            except Exception:
                logger.warning("Failed to read engine data")
        warp_version = read_fmt("H", fp)[0]
        warp = DescriptorBlock.read(fp)
        left, top, right, bottom = read_fmt("4i", fp)
        return cls(
            version,
            transform,
            text_version,
            text_data,
            warp_version,
            warp,
            left,
            top,
            right,
            bottom,
        )

    def write(self, fp: IO[bytes], padding: int = 4, **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = write_fmt(fp, "H6d", self.version, *self.transform)
        written += write_fmt(fp, "H", self.text_version)
        written += self.text_data.write(fp, padding=1)
        written += write_fmt(fp, "H", self.warp_version)
        written += self.warp.write(fp, padding=1)
        written += write_fmt(fp, "4i", self.left, self.top, self.right, self.bottom)
        written += write_padding(fp, written, padding)
        return written


@register(Tag.USER_MASK)
@define(repr=False)
class UserMask(BaseElement):
    """
    UserMask structure.

    .. py:attribute:: color
    .. py:attribute:: opacity
    .. py:attribute:: flag
    """

    color: Color = field(factory=Color)
    opacity: int = 0
    flag: int = 128

    @classmethod
    def read(cls, fp: IO[bytes], **kwargs: Any) -> "UserMask":
        """Read the structure from a file-like object."""
        color = Color.read(fp)
        opacity, flag = read_fmt("HBx", fp)
        return cls(color, opacity, flag)

    def write(self, fp: IO[bytes], **kwargs: Any) -> int:
        """Write the structure to a file-like object and return the bytes written."""
        written = self.color.write(fp)
        written += write_fmt(fp, "HBx", self.opacity, self.flag)
        return written


_HEAD = struct.Struct(">4s4sI")
_SIGNATURES = frozenset(TaggedBlock._SIGNATURES)
# Keys held as bytes until read: every handled key but the layer tree's own
# and the lazy payloads.
_HELD = frozenset(
    key.value
    for key in TYPES
    if key not in TaggedBlock._STRUCTURAL and key not in TaggedBlock._LAZY_KEYS
)


def _hold(fp: io.BytesIO, items: list[tuple[Any, TaggedBlock]], padding: int) -> bool:
    """Read the blocks of ``fp``, keeping the ``_HELD`` keys' bytes unparsed.

    Any other block, or one cut short, goes through :py:meth:`TaggedBlock.read`;
    False where that returns None.
    """
    data = fp.getbuffer()
    try:
        position, end = fp.tell(), len(data)
        budget = current_budget()
        while position + 12 <= end:
            signature, key, length = _HEAD.unpack_from(data, position)
            start = position + 12
            stop = start + length
            gap = -length % padding
            if signature not in _SIGNATURES or key not in _HELD or stop + gap > end:
                fp.seek(position)
                block = TaggedBlock.read(fp, 1, padding)
                if block is None:
                    return False
                items.append((block.key, block))
                position = fp.tell()
                continue
            if budget is not None:  # what TaggedBlock.read charges
                budget.consume(3, 12)
                budget.consume(0, length)
                if gap:
                    budget.consume(0, gap)
            tag = _TAGS[key]
            raw = bytes(data[start:stop])
            items.append((tag, TaggedBlock._unparsed(signature, tag, raw, padding)))
            position = stop + gap
    finally:
        data.release()
    fp.seek(position)
    return True
