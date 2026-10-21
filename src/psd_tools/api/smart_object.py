"""Smart object module."""

import contextlib
import io
import logging
import os
from collections.abc import Iterator
from typing import IO

import numpy as np

from psd_tools.api._descriptor import get_scalar
from psd_tools.api.protocols import LayerProtocol
from psd_tools.compression import decompress
from psd_tools.constants import Compression, Tag
from psd_tools.psd.descriptor import Descriptor
from psd_tools.psd.filter_effects import FilterEffect, FilterEffectExtra

logger = logging.getLogger(__name__)

# (prefix_bytes, filetype_string) — ordered so longer/more-specific prefixes
# come first (8BPS+version before bare 8BPS would collide; PSD/PSB checked
# as 6-byte prefixes including the version field).
_MAGIC_TABLE: tuple[tuple[bytes, str], ...] = (
    (b"8BPS\x00\x01", "8bps"),  # PSD  (version 1)
    (b"8BPS\x00\x02", "8bpb"),  # PSB  (version 2)
    (b"%PDF-", "pdf"),
    (b"\x89PNG", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"II*\x00", "tiff"),  # TIFF little-endian
    (b"MM\x00*", "tiff"),  # TIFF big-endian
    (b"PK\x03\x04", "zip"),  # ZIP / ZIP-based AI
    (b"BM", "bmp"),
)
_MAGIC_PREFIX_LEN = max(len(p) for p, _ in _MAGIC_TABLE)


def _detect_filetype_from_magic(data: bytes) -> str | None:
    """Return a filetype string inferred from *data* magic bytes, or ``None``."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    prefix = data[:_MAGIC_PREFIX_LEN]
    for magic, result in _MAGIC_TABLE:
        if prefix[: len(magic)] == magic:
            return result
    return None


def _is_inside(parent: str, child: str) -> bool:
    """Return True if child is inside parent (or equal to it)."""
    parent = os.path.normcase(parent)
    child = os.path.normcase(child)
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:
        # commonpath raises ValueError on Windows when paths span different drives
        return False


class SmartObject:
    """
    Smart object that represents embedded or external file.

    Smart objects are attached to
    :py:class:`~psd_tools.api.layers.SmartObjectLayer`.
    """

    def __init__(self, layer: LayerProtocol):
        self._layer = layer
        self._config = None
        for key in (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2):
            self._config = layer.tagged_blocks.get_data(key)
            if self._config is not None:
                break

        self._placed_layer = None
        for key in (Tag.PLACED_LAYER1, Tag.PLACED_LAYER2):
            self._placed_layer = layer.tagged_blocks.get_data(key)
            if self._placed_layer is not None:
                break

        self._data = None
        unique_id = self._read_unique_id()
        if unique_id is not None and layer._psd.tagged_blocks is not None:
            for key in (
                Tag.LINKED_LAYER1,
                Tag.LINKED_LAYER2,
                Tag.LINKED_LAYER3,
                Tag.LINKED_LAYER_EXTERNAL,
            ):
                data = layer._psd.tagged_blocks.get_data(key)
                if data is not None:
                    for item in data:
                        if item.uuid == unique_id:
                            self._data = item
                            break
                    if self._data:
                        break

    @property
    def kind(self) -> str:
        """Kind of the link, 'data', 'alias', or 'external'."""
        if self._data is None:
            raise ValueError("Smart object data not found")
        return self._data.kind.name.lower()

    @property
    def filename(self) -> str:
        """Original file name of the object."""
        if self._data is None:
            raise ValueError("Smart object data not found")
        return self._data.filename.strip("\x00")

    @contextlib.contextmanager
    def open(
        self,
        external_dir: str | os.PathLike | None = None,
        *,
        trust_full_path: bool = False,
    ) -> Iterator[IO[bytes]]:
        """
        Open the smart object as binary IO.

        For external smart objects the linked file is read from disk. By default
        the read is confined to a directory: ``external_dir`` when given,
        otherwise the current working directory. A ``fullPath`` stored in the PSD
        is honoured only when it resolves inside that directory; otherwise it is
        ignored and ``relPath`` (resolved inside the directory) is tried instead.
        A ``relPath`` that escapes the directory raises :py:exc:`ValueError`. This
        default prevents a crafted PSD from reading arbitrary files via an
        absolute ``fullPath``. Note that the confinement directory itself is
        trusted: a ``relPath`` resolving to a file *inside* it is still read, so
        point ``external_dir`` at a location that holds only linked assets when
        processing untrusted files.

        :param external_dir: Directory to resolve and confine external paths
            against. Defaults to the current working directory. Strongly
            recommended to set explicitly when processing untrusted PSD files.
        :param trust_full_path: When ``True``, disable confinement entirely and
            trust ``fullPath``/``relPath`` from the PSD verbatim (the historical
            behaviour). Only use this for fully trusted files. Cannot be combined
            with ``external_dir``.
        :raises ValueError: If ``trust_full_path`` is combined with
            ``external_dir``, or a ``relPath`` escapes the confinement directory.

        Example::

            with layer.smart_object.open() as f:
                data = f.read()
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if trust_full_path and external_dir is not None:
            raise ValueError(
                "external_dir cannot be combined with trust_full_path=True; "
                "trust_full_path disables path confinement entirely"
            )
        if self.kind == "data":
            with io.BytesIO(self._data.data) as f:
                yield f
        elif self.kind == "external":
            # safe_dir is the confinement boundary; None means confinement is
            # disabled (trust_full_path=True) and paths are used verbatim.
            safe_dir = (
                None
                if trust_full_path
                else os.path.realpath(
                    external_dir if external_dir is not None else os.getcwd()
                )
            )
            linked_file = self._data.linked_file
            filepath = get_scalar(linked_file, b"fullPath", str, "")
            filepath = filepath.replace("\x00", "").replace("file://", "")
            if safe_dir is not None and filepath:
                resolved = os.path.realpath(filepath)
                if _is_inside(safe_dir, resolved):
                    # Open the validated, symlink-resolved path rather than the
                    # raw value, matching the relPath branch below.
                    filepath = resolved
                else:
                    # fullPath escapes safe_dir — fall through to relPath
                    filepath = ""
            if not filepath or not os.path.exists(filepath):
                relpath = get_scalar(linked_file, b"relPath", str, "")
                relpath = relpath.replace("\x00", "")
                if not relpath:
                    raise ValueError("External smart object has no readable path")
                if safe_dir is not None:
                    filepath = os.path.realpath(os.path.join(safe_dir, relpath))
                    if not _is_inside(safe_dir, filepath):
                        raise ValueError(
                            f"External smart object path escapes external_dir: {relpath!r}"
                        )
                else:
                    filepath = relpath
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Smart object file not found: {filepath}")
            with open(filepath, "rb") as f:
                yield f
        else:
            raise NotImplementedError("alias is not supported.")

    @property
    def data(self) -> bytes:
        """Embedded file content as bytes.

        For ``kind == 'data'`` this returns the bytes stored in the PSD without
        any filesystem access.  For ``kind == 'external'`` this calls
        :py:meth:`open` with default confinement, so the read is restricted to
        the current working directory and an absolute ``fullPath`` pointing
        outside it is ignored.  Use :py:meth:`open` with an explicit
        ``external_dir`` to read linked files from a known location, or
        ``open(trust_full_path=True)`` to restore the historical unconfined
        behaviour for fully trusted files.
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind == "data":
            return self._data.data
        else:
            with self.open() as f:
                return f.read()

    def _read_unique_id(self) -> str | None:
        if self._config is None:
            return None
        value = get_scalar(self._config.data, b"Idnt", str, None)
        return None if value is None else value.strip("\x00")

    @property
    def unique_id(self) -> str:
        """UUID of the object."""
        if self._config is None:
            raise ValueError("Smart object config not found")
        unique_id = self._read_unique_id()
        if unique_id is None:
            raise ValueError("Smart object identifier not found")
        return unique_id

    @property
    def filesize(self) -> int:
        """File size of the object."""
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind == "data":
            return len(self._data.data)
        return self._data.filesize

    @property
    def filetype(self) -> str | None:
        """4-byte file-type code stored in the PSD, lowercased and stripped.

        Returns ``None`` when the field is absent or blank (e.g. some
        Adobe Illustrator smart objects store four spaces instead of a real
        type code).  Use :py:attr:`detected_filetype` for a best-effort
        guess that falls back to the filename extension and magic bytes.
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        raw = self._data.filetype.strip(b" \x00").lower().decode("ascii")
        return raw if raw else None

    @property
    def detected_filetype(self) -> str | None:
        """Best-effort file type, falling back when :py:attr:`filetype` is ``None``.

        Resolution order:

        1. :py:attr:`filetype` — the value stored in the PSD.
        2. Extension taken from :py:attr:`filename` (lower-cased, dot stripped).
        3. Magic-byte sniffing of the first few embedded bytes
           (only for ``kind == 'data'``; external objects are not read from
           disk to keep this property side-effect-free).

        .. warning::
            Steps 2 and 3 are heuristic.  Do not use the result for
            security-sensitive decisions (e.g. deciding whether to execute
            content or trust a type boundary).
        """
        ft = self.filetype
        if ft is not None:
            return ft

        # Extension from the embedded filename (safe: no filesystem access)
        clean = self._data.filename.replace("\x00", "").strip() if self._data else ""
        _, ext = os.path.splitext(clean)
        suffix = ext.lstrip(".").lower()
        if suffix:
            return suffix

        # Magic bytes — only for embedded (data-kind) objects
        if self._data is not None and self.kind == "data":
            raw_data = self._data.data
            if raw_data:
                return _detect_filetype_from_magic(raw_data)

        return None

    def is_psd(self) -> bool:
        """Return True if the file is embedded PSD/PSB.

        Checks the stored :py:attr:`filetype` first; if that is ``None``
        (blank field), falls back to inspecting the first four magic bytes of
        the embedded data.  The filename extension is intentionally *not* used
        as a fallback to avoid misclassifying non-PSD files named ``*.psd``.
        """
        ft = self.filetype
        if ft in ("8bpb", "8bps"):
            return True
        if self._data is not None and self.kind == "data":
            raw_data = self._data.data
            if raw_data and len(raw_data) >= 4:
                return raw_data[:4] == b"8BPS"
        return False

    @property
    def warp(self) -> object | None:
        """Warp parameters."""
        if self._config is None:
            raise ValueError("Smart object config not found")
        return self._config.data.get(b"warp")

    @property
    def resolution(self) -> float | None:
        """Resolution of the object, or `None` if the file does not record one."""
        if self._config is None:
            raise ValueError("Smart object config not found")
        return get_scalar(self._config.data, b"Rslt", float, None)

    @property
    def transform_box(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float] | None:
        """
        A tuple representing the coordinates of the smart objects's transformed box. This box is the result of one or more transformations such as scaling, rotation, translation, or skewing to the original bounding box of the smart object.

        The format of the tuple is as follows:
        (x1, y1, x2, y2, x3, y3, x4, y4)

        Where 1 is top left corner, 2 is top right, 3 is bottom right and 4 is bottom left.
        """
        if self._placed_layer and hasattr(self._placed_layer, "transform"):
            return self._placed_layer.transform
        else:
            return None

    # -- Smart filters -----------------------------------------------------

    @property
    def filters(self) -> Descriptor | None:
        """The smart filter descriptor (``filterFX``), or ``None``."""
        if self._config is None:
            return None
        value = self._config.data.get(b"filterFX")
        return value if isinstance(value, Descriptor) else None

    @property
    def has_enabled_filters(self) -> bool:
        """Whether an enabled smart filter is applied to the object."""
        fx = self.filters
        if fx is None or not _flag(fx, b"enab", True):
            return False
        items = fx.get(b"filterFXList") or []
        return any(_flag(item, b"enab", True) for item in items)

    @property
    def filter_mask_enabled(self) -> bool:
        """Whether the smart filters' mask is on."""
        return _flag(self.filters, b"filterMaskEnable", False)

    @property
    def filter_mask_extends_white(self) -> bool:
        """Mask value outside its stored rectangle: white (filtered) or black."""
        return _flag(self.filters, b"filterMaskExtendWithWhite", True)

    def filter_effect(self) -> FilterEffect | None:
        """Return the document's stored pre-filter raster and mask for this layer.

        Photoshop keeps, for each smart object carrying smart filters, the
        rendered object *before* the filters (colour channels plus
        transparency) and the painted filter mask in the document-level
        ``FEid`` block, while the layer's own channels hold the filtered
        result; it composites ``lerp(source, filtered, mask)``. The block
        lists one item per filtered layer in layer order, and the item's
        rectangle contains the layer's bbox; both are checked here.
        """
        if self.filters is None:
            return None
        psd = getattr(self._layer, "_psd", None)
        blocks = getattr(psd, "tagged_blocks", None)
        if psd is None or blocks is None:
            return None
        items = None
        for key in (Tag.FILTER_EFFECTS1, Tag.FILTER_EFFECTS2, Tag.FILTER_EFFECTS3):
            items = blocks.get_data(key)
            if items is not None:
                break
        if not items:
            return None
        bbox = self._layer.bbox

        def contains(item: FilterEffect) -> bool:
            if item.rectangle is None or bbox is None:
                return False
            top, left, bottom, right = item.rectangle
            return (
                left <= bbox[0]
                and top <= bbox[1]
                and right >= bbox[2]
                and bottom >= bbox[3]
            )

        index = 0
        for layer in psd.descendants():
            if layer is self._layer:
                break
            if layer.kind == "smartobject" and layer.smart_object.filters is not None:
                index += 1
        if index < len(items) and contains(items[index]):
            return items[index]
        for item in items:
            if contains(item):
                return item
        return None

    def filter_source(
        self,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray | None] | None:
        """``(bbox, color uint8 (h, w, 3), alpha uint8 (h, w) | None)`` of the
        object as rendered before its smart filters, or ``None``.
        """
        item = self.filter_effect()
        if (
            item is None
            or item.depth != 8
            or not item.channels
            or item.rectangle is None
            or item.max_channels is None
        ):
            return None
        top, left, bottom, right = item.rectangle
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            return None
        planes = []
        for channel in item.channels[:3]:
            if not channel.is_written or channel.compression is None:
                return None
            planes.append(_decode_plane(channel.data, channel.compression, w, h))
        if len(planes) == 1:
            planes = planes * 3
        color = np.stack(planes[:3], axis=-1)
        alpha = None
        index = item.max_channels + 1
        transparency = item.channels[index] if len(item.channels) > index else None
        if (
            transparency is not None
            and transparency.is_written
            and transparency.compression is not None
        ):
            alpha = _decode_plane(transparency.data, transparency.compression, w, h)
        return (left, top, right, bottom), color, alpha

    def filter_mask(self) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """``(bbox, mask uint8 (h, w))`` of the smart filters' mask (255 =
        filtered), or ``None`` when no mask pixels are stored.
        """
        item = self.filter_effect()
        extra = item.extra if item is not None else None
        if (
            not isinstance(extra, FilterEffectExtra)
            or not extra.is_written
            or not extra.data
        ):
            return None
        top, left, bottom, right = extra.rectangle
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            return None
        return (left, top, right, bottom), _decode_plane(
            extra.data, extra.compression, w, h
        )

    def save(
        self,
        filename: str | os.PathLike | None = None,
        directory: str | os.PathLike | None = None,
        external_dir: str | os.PathLike | None = None,
        *,
        trust_full_path: bool = False,
    ) -> None:
        """
        Save the smart object to a file.

        :param filename: Explicit destination path. When provided it is used
            as-is and ``directory`` is ignored.
        :param directory: Output directory used when ``filename`` is ``None``.
            Defaults to the current working directory. The embedded basename is
            written inside this directory; path-traversal sequences in the
            embedded name are stripped automatically.
        :param external_dir: Passed to :py:meth:`open` when the smart object
            kind is ``'external'``. Constrains which paths on disk may be read.
            Strongly recommended when processing untrusted PSD files.
        :param trust_full_path: Passed to :py:meth:`open` when the smart object
            kind is ``'external'``. When ``True``, disable read confinement and
            trust the PSD paths verbatim. Cannot be combined with
            ``external_dir``.
        :raises ValueError: If the embedded name contains no safe basename,
            resolves outside ``directory``, ``trust_full_path`` is combined with
            ``external_dir``, or (for external kind) the source path escapes
            ``external_dir``.
        """
        if trust_full_path and external_dir is not None:
            raise ValueError(
                "external_dir cannot be combined with trust_full_path=True; "
                "trust_full_path disables path confinement entirely"
            )
        if filename is None:
            basename = os.path.basename(self.filename)
            if not basename or basename == ".":
                raise ValueError(
                    f"Embedded smart object filename has no safe basename: {self.filename!r}"
                )
            outdir = os.path.realpath(
                directory if directory is not None else os.getcwd()
            )
            resolved = os.path.realpath(os.path.join(outdir, basename))
            if not _is_inside(outdir, resolved):
                raise ValueError(
                    f"Embedded filename resolves outside target directory: {self.filename!r}"
                )
            filename = resolved
        if self.kind == "external":
            with self.open(
                external_dir=external_dir, trust_full_path=trust_full_path
            ) as f:
                content = f.read()
        else:
            content = self.data
        with open(filename, "wb") as f:
            f.write(content)

    def __repr__(self) -> str:
        return f"SmartObject({self.filename!r} kind={self.kind!r} type={self.filetype!r} detected={self.detected_filetype!r} size={self.filesize})"


def _flag(desc: Descriptor | None, key: bytes, default: bool) -> bool:
    """Return a boolean descriptor field, or *default* when absent."""
    if desc is None:
        return default
    value = desc.get(key)
    return default if value is None else bool(getattr(value, "value", value))


def _decode_plane(data: bytes, compression: int, width: int, height: int) -> np.ndarray:
    """One 8-bit ``FEid`` plane as ``uint8 (height, width)``.

    The rows carry 4-byte RLE counts (the PSB layout) in PSD files too: the
    4-byte table sums to the payload length, the 2-byte one does not.
    """
    raw = decompress(data, Compression(compression), width, height, 8, 2)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
