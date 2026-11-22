"""Smart object module."""

import logging
import posixpath
from collections.abc import Callable, Sequence
from copy import deepcopy

import numpy as np

from psd_tools.api import links
from psd_tools.api._descriptor import get_scalar
from psd_tools.api.numpy_io import _DECOMPRESS_PEAK
from psd_tools.api.protocols import LayerProtocol, PSDProtocol
from psd_tools.api.utils import check_pixel_size, get_color_channels
from psd_tools.compression import _decompress, decompress_row_peak_bytes
from psd_tools.constants import Compression, LinkedLayerType, Tag
from psd_tools.psd.descriptor import Descriptor, Double
from psd_tools.psd.filter_effects import (
    FilterEffect,
    FilterEffectChannel,
    FilterEffectExtra,
)
from psd_tools.psd.linked_layer import LinkedLayer, LinkedLayers
from psd_tools.psd.tagged_blocks import (
    SmartObjectLayerData,
    TaggedBlock,
    TaggedBlocks,
)

logger = logging.getLogger(__name__)

_FILTER_TAGS = (Tag.FILTER_EFFECTS1, Tag.FILTER_EFFECTS2, Tag.FILTER_EFFECTS3)


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

    @property
    def data(self) -> bytes:
        """Embedded file content as bytes.

        For ``kind == 'data'`` this returns the bytes stored in the PSD without
        any filesystem access. A linked file is never read.

        :raises ValueError: for a linked object, naming the repository path it
            links (see :py:meth:`link_target`).
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind == "data":
            return self._data.data
        if self.kind != "external":
            raise NotImplementedError("alias is not supported.")
        path = self.link_target()
        if path is None:
            stored = get_scalar(self._data.linked_file, b"relPath", str, "")
            raise ValueError(f"Linked file {stored.strip(chr(0))!r} places nowhere")
        raise ValueError(f"Smart object links {path!r}")

    def link_target(self, psd_path: str | None = None) -> str | None:
        """Return the repository path of the file this object links.

        Read off the record's strings, never the file system: ``relPath`` from
        the PSD's folder (it may not climb above the root), then ``fullPath``
        and ``originalPath``, an absolute path or ``file://`` URL cut after its
        last ``/TTSCardMaker/``: the first repository path a string names.

        :param psd_path: the PSD's repository path; default the document's
            ``repo_path``.
        :return: the path, or ``None`` when it places nowhere or the object is
            embedded.
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind != "external":
            return None
        return links.place(self._data.linked_file, self._psd_path(psd_path))

    @property
    def content_id(self) -> str | None:
        """The link record's ``contentID``, or ``None`` without one."""
        if self._data is None or self._data.content_descriptor is None:
            return None
        value = get_scalar(self._data.content_descriptor, b"contentID", str, None)
        return None if value is None else value.strip("\x00") or None

    def link(
        self,
        path: str,
        size: int,
        *,
        blob: str,
        psd_path: str | None = None,
        pixels: tuple[float, float] | None = None,
        corners: "Corners | None" = None,
    ) -> None:
        """Make this embedded object a link to repository file *path*.

        Written as Photoshop 2023 writes a link (``lnkE`` version 8) with the
        same uuid, so every layer sharing the object follows: their ``SoLd``
        becomes ``SoLE``, their ``PlLd`` goes and the embedded item is removed.
        The layers keep their pixels. ``relPath`` runs from the PSD's folder,
        ``fullPath`` and ``originalPath`` are ``LINK_ROOT`` + *path*, and the
        ``contentID`` is :py:func:`~psd_tools.api.links.link_content_id` of
        *blob*, the PSD path and the uuid.

        :param path: the linked file's repository path.
        :param size: its size in bytes.
        :param blob: its git blob sha.
        :param psd_path: the PSD's repository path; default the document's
            ``repo_path``.
        :param pixels: the file's (width, height) when it is not the embedded
            file's, written as each layer's ``Sz``.
        :param corners: the new placement (``Trnf``): 8 numbers, or a callable
            taking a layer's current 8 and returning its new ones, for an
            object several layers share.
        :raises ValueError: if the object is not embedded, has no uuid, the
            PSD path is unknown, or a corner sequence is given for an object
            several layers share.
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind != "data":
            raise ValueError("Only an embedded smart object can be linked")
        psd_path = self._known_path(psd_path)
        item = self._data
        users = self._users()
        placed = _placements(users, corners)
        name = posixpath.basename(links.normal_path(path))
        external = LinkedLayer(
            kind=LinkedLayerType.EXTERNAL,
            version=8,
            uuid=item.uuid,
            filename=name + "\x00",
            filetype=_filetype(path, item.filetype),
            creator=b"\x00\x00\x00\x00",
            filesize=size,
            open_file=deepcopy(item.open_file)
            if item.open_file is not None
            else links.comp_info(),
            linked_file=links.linked_file(path, psd_path, name),
            timestamp=links.LINK_DATE,
            data=b"",
            child_id="\x00",
            mod_time=0.0,
            lock_state=0,
            content_descriptor=links.content_descriptor(
                links.link_content_id(blob, psd_path, item.uuid)
            ),
        )
        _swap_item(self._layer._psd, item, external)
        for layer, new in zip(users, placed, strict=True):
            _set_placement(layer, pixels, new, embedded=False)
        self._refresh(users, external)

    def relink(
        self,
        path: str | None = None,
        size: int | None = None,
        *,
        blob: str | None = None,
        psd_path: str | None = None,
        pixels: tuple[float, float] | None = None,
        corners: "Corners | None" = None,
    ) -> None:
        """Rewrite this link canonically, as :py:meth:`link` writes one.

        The record's paths are rebuilt for *path* (default the file it links
        now) from the PSD's repository path. *size*, *blob* (a new
        ``contentID``), *pixels* and *corners* are written
        when given; the raster is never changed.

        :raises ValueError: if the object is not linked, the PSD path is
            unknown, or its link places nowhere and no *path* is given.
        """
        if self._data is None:
            raise ValueError("Smart object data not found")
        if self.kind != "external":
            raise ValueError("Only a linked smart object can be relinked")
        psd_path = self._known_path(psd_path)
        item = self._data
        if path is None:
            path = self.link_target(psd_path)
            if path is None:
                raise ValueError("The link places nowhere: give its path")
        users = self._users()
        placed = _placements(users, corners)
        name = posixpath.basename(links.normal_path(path))
        item.filename = name + "\x00"
        item.filetype = _filetype(path, item.filetype)
        item.linked_file = links.linked_file(path, psd_path, name)
        item.timestamp = links.LINK_DATE
        if size is not None:
            item.filesize = size
        if blob is not None:
            if item.version < 8:
                item.child_id = item.child_id if item.child_id is not None else "\x00"
                item.mod_time = item.mod_time if item.mod_time is not None else 0.0
                item.lock_state = item.lock_state if item.lock_state is not None else 0
                item.version = 8
            item.content_descriptor = links.content_descriptor(
                links.link_content_id(blob, psd_path, item.uuid)
            )
        for layer, new in zip(users, placed, strict=True):
            _set_placement(layer, pixels, new, embedded=False)
        self._refresh(users, item)

    def _psd_path(self, psd_path: str | None) -> str | None:
        if psd_path is not None:
            return psd_path
        return getattr(getattr(self._layer, "_psd", None), "repo_path", None)

    def _known_path(self, psd_path: str | None) -> str:
        found = self._psd_path(psd_path)
        if found is None:
            raise ValueError("The PSD's repository path is unknown: set repo_path")
        return links.normal_path(found)

    def _users(self) -> list[LayerProtocol]:
        """Every smart object layer of the document naming this object's item."""
        unique_id = self.unique_id
        psd = self._layer._psd
        found = [
            layer
            for layer in psd.descendants()
            if layer.kind == "smartobject" and _unique_id(layer) == unique_id
        ]
        return found or [self._layer]

    def _refresh(self, users: Sequence[LayerProtocol], item: LinkedLayer) -> None:
        self._data = item
        self._placed_layer = None
        for layer in users:
            if getattr(layer, "_smart_object", None) is not self:
                layer.__dict__.pop("_smart_object", None)

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
    def warp(self) -> object | None:
        """Warp parameters."""
        if self._config is None:
            raise ValueError("Smart object config not found")
        return self._config.data.get(b"warp")

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
        corners = None if self._config is None else self._config.data.get(b"Trnf")
        if corners is None or len(corners) != 8:
            return None
        return tuple(float(getattr(v, "value", v)) for v in corners)  # type: ignore[return-value]

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
        if fx is None or not get_scalar(fx, b"enab", bool, True):
            return False
        items = fx.get(b"filterFXList") or []
        return any(
            not isinstance(item, Descriptor) or get_scalar(item, b"enab", bool, True)
            for item in items
        )

    @property
    def filter_mask_enabled(self) -> bool:
        """Whether the smart filters' mask is on."""
        return get_scalar(self.filters, b"filterMaskEnable", bool, False)

    @property
    def filter_mask_extends_white(self) -> bool:
        """Mask value outside its stored rectangle: white (filtered) or black."""
        return get_scalar(self.filters, b"filterMaskExtendWithWhite", bool, True)

    @property
    def placed_id(self) -> str | None:
        """The placed instance's ID (``placed``), NULs stripped; ``None`` without one.

        Each placed copy of one embedded object has its own.
        """
        if self._config is None:
            return None
        value = get_scalar(self._config.data, b"placed", str, None)
        return None if value is None else value.strip("\x00") or None

    @property
    def filter_item(self) -> FilterEffect | None:
        """The document's ``FEid`` item named after this layer's placed ID.

        Photoshop names each item after the placed instance it belongs to, so
        a copy has its own, and moving a layer moves its item's rectangle (the
        planes stay). ``None`` without smart filters or such an item.
        """
        if self.filters is None:
            return None
        placed = self.placed_id
        if placed is None:
            return None
        return next(
            (
                item
                for item in _filter_items(self._layer)
                if (item.uuid or "").strip("\x00") == placed
            ),
            None,
        )

    def _paired_effect(self) -> tuple[FilterEffect, int, int] | None:
        """Return the document's stored filter input and mask for this layer.

        Photoshop keeps, for each smart object carrying smart filters, the
        object rendered *before* the filters and the painted filter mask in the
        document's ``FEid`` block, one item per filtered layer in file order;
        the layer's own channels hold the filtered result. The item named after
        the layer's placed ID (:py:attr:`filter_item`) is its own, where it
        lies. Else an item belongs to this layer only if its rectangle contains
        the layer's bbox as the document was opened: the item at the layer's
        position among the filtered layers then if it does, else the first one
        that does. A duplicate pairs as its source does; a layer the document
        was not opened with has none. With the item, how far the layer has
        moved since.
        """
        if self.filters is None:
            return None
        own = self.filter_item
        if own is not None:  # moved with its layer, as Photoshop does
            return own, 0, 0
        psd = getattr(self._layer, "_psd", None)
        if psd is None or getattr(psd, "tagged_blocks", None) is None:
            return None
        opened = psd._opened_records.get(id(self._layer._record))
        if opened is None:
            return None
        items = _filter_items(self._layer)
        if not items:
            return None
        _, index, left, top = opened
        bbox = (left, top, left + self._layer.width, top + self._layer.height)
        if index is not None and index < len(items) and _contains(items[index], bbox):
            item = items[index]
        else:
            item = next((item for item in items if _contains(item, bbox)), None)
        if item is None:
            return None
        return item, self._layer.left - left, self._layer.top - top

    def filter_source(
        self,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray | None] | None:
        """Return the object as rendered before its smart filters.

        ``(bbox, color, alpha)``: ``bbox`` moves with the layer, ``color`` is
        ``uint8 (h, w, channels)``, the document's colour channels as stored,
        and ``alpha`` is ``uint8 (h, w)`` or ``None``. ``None`` when the
        document stores no 8-bit source.

        :raises ValueError: if the planes are over the allocation budget, or a
            plane does not decode or its RLE row counts do not match its
            payload.
        """
        return self._filter_source()

    def _filter_source(
        self, held: int = 0
    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray | None] | None:
        """:py:meth:`filter_source`; *held* bytes stay live across the decode."""
        paired = self._paired_effect()
        if paired is None:
            return None
        item, dx, dy = paired
        if (
            item.depth != 8
            or item.rectangle is None
            or item.max_channels is None
            or not item.channels
        ):
            return None
        psd = self._layer._psd
        top, left, bottom, right = item.rectangle
        width, height = right - left, bottom - top
        count = get_color_channels(psd)
        colour = item.channels[:count]
        if width <= 0 or height <= 0 or len(colour) < count:
            return None
        if not all(_is_stored(channel) for channel in colour):
            return None
        index = item.max_channels + 1
        transparency = item.channels[index] if index < len(item.channels) else None
        if not _is_stored(transparency):
            transparency = None
        planes = colour + ([transparency] if transparency is not None else [])
        _check_planes(psd, width, height, planes, held + width * height * count)
        color = np.empty((height, width, count), dtype=np.uint8)
        for i, channel in enumerate(colour):
            color[..., i] = _decode_plane(channel, width, height)
        alpha = None
        if transparency is not None:
            alpha = _decode_plane(transparency, width, height)
        return (left + dx, top + dy, right + dx, bottom + dy), color, alpha

    def filter_mask(self) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Return the smart filters' mask, or ``None`` when none is stored.

        ``(bbox, mask uint8 (h, w))``; 255 = filtered, 0 = unfiltered. The
        bbox moves with the layer.

        :raises ValueError: as :py:meth:`filter_source` does.
        """
        return self._filter_mask()

    def _filter_mask(
        self, held: int = 0
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """:py:meth:`filter_mask`; *held* bytes stay live across the decode."""
        paired = self._paired_effect()
        if paired is None:
            return None
        item, dx, dy = paired
        if item.depth != 8:
            return None
        extra = item.extra
        if not isinstance(extra, FilterEffectExtra) or not _is_stored(extra):
            return None
        top, left, bottom, right = extra.rectangle
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return None
        _check_planes(self._layer._psd, width, height, [extra], held)
        bbox = (left + dx, top + dy, right + dx, bottom + dy)
        return bbox, _decode_plane(extra, width, height)

    def can_draw_filters(self) -> bool:
        """Return whether :py:meth:`draw_filters` draws the smart filters on.

        True with none on, or one Puppet Warp, Invert, Gaussian Blur, proven
        Motion Blur, master-only Hue/Saturation, untinted Black & White or
        Relative Selective Color of the six hues, Normal at 100%, in an RGB
        document; a filter mask does not refuse (the composite applies it).
        """
        from psd_tools.composite import smart_filters  # noqa: PLC0415

        return smart_filters.drawable(self)

    def draw_filters(
        self,
        rgba: np.ndarray,
        box: tuple[int, int, int, int],
        corners: Sequence[float] | None = None,
        sampling: str | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Draw the smart filters over a new picture of the object.

        For a redraw from a changed file: *rgba* (straight float RGBA in 0..1)
        is the new picture drawn on the quad *corners* (default
        :py:attr:`transform_box`) over canvas *box*. Call it before the
        layer's pixels or filter source change. See
        :py:func:`psd_tools.composite.smart_filters.draw`.

        :return: ``(straight float32 RGBA, canvas box)``
        :raises ValueError: when :py:meth:`can_draw_filters` is False
        """
        from psd_tools.composite import smart_filters  # noqa: PLC0415

        quad = None if corners is None else tuple(float(v) for v in corners)
        return smart_filters.draw(self, rgba, box, quad, sampling)  # type: ignore[arg-type]

    def can_draw_warp(self) -> bool:
        """Return whether :py:meth:`draw_warp` draws the object's warp.

        True for a custom warp of 4 x 4 points over its whole content (no
        split grid, no preset bend) on a parallelogram quad.
        """
        from psd_tools.composite import placed_warp  # noqa: PLC0415

        return placed_warp.drawable(self)

    def draw_warp(
        self, rgba: np.ndarray, corners: Sequence[float] | None = None
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Draw the object's custom warp on a new picture of its file.

        For a redraw from a changed file of the same size: *rgba* (straight
        RGBA, uint8 or float in 0..1) is the file's pixels, *corners* the
        layer's quad (default :py:attr:`transform_box`). See
        :py:func:`psd_tools.composite.placed_warp.draw`.

        :return: ``(straight float32 RGBA, canvas box)``
        :raises ValueError: when :py:meth:`can_draw_warp` is False or the
            picture is another size
        """
        from psd_tools.composite import placed_warp  # noqa: PLC0415

        quad = None if corners is None else tuple(float(v) for v in corners)
        return placed_warp.draw(self, rgba, quad)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"SmartObject({self.filename!r} kind={self.kind!r} size={self.filesize})"


#: A placement: 8 corner numbers, or a layer's current 8 -> its new ones.
Corners = Sequence[float] | Callable[[tuple[float, ...]], Sequence[float]]

_LIST_TAGS = (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3)
_PLACED_TAGS = (Tag.PLACED_LAYER1, Tag.PLACED_LAYER2)


def _unique_id(layer: LayerProtocol) -> str | None:
    for key in (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2):
        config = layer.tagged_blocks.get_data(key)
        if config is not None:
            value = get_scalar(config.data, b"Idnt", str, None)
            return None if value is None else value.strip("\x00")
    return None


def _filetype(path: str, current: bytes) -> bytes:
    ext = posixpath.splitext(path)[1].lstrip(".").lower()
    return links.FILETYPES.get(ext, current)


def _placements(
    users: Sequence[LayerProtocol], corners: Corners | None
) -> list[tuple[float, ...] | None]:
    """Each user's new corners, checked before anything is written."""
    if corners is None:
        return [None] * len(users)
    if not callable(corners):
        if len(users) > 1:
            raise ValueError("Several layers share this object: give a callable")
        out = [tuple(float(v) for v in corners)]
    else:
        out = []
        for layer in users:
            current = _config(layer).data.get(b"Trnf")
            old = tuple(float(getattr(v, "value", v)) for v in current or ())
            out.append(tuple(float(v) for v in corners(old)))
    if any(len(c) != 8 for c in out):
        raise ValueError("A placement has 8 corner numbers")
    return out  # type: ignore[return-value]


def _config(layer: LayerProtocol) -> SmartObjectLayerData:
    for key in (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2):
        config = layer.tagged_blocks.get_data(key)
        if config is not None:
            return config
    raise ValueError("Smart object config not found")


def _set_placement(
    layer: LayerProtocol,
    pixels: tuple[float, float] | None,
    corners: tuple[float, ...] | None,
    embedded: bool,
) -> None:
    """``SoLE`` without ``PlLd`` (unless *embedded*), with the values given."""
    blocks = layer.tagged_blocks
    if not embedded and Tag.SMART_OBJECT_LAYER_DATA1 in blocks:
        items = list(blocks.items())
        blocks.clear()
        for key, block in items:
            if key in _PLACED_TAGS:
                continue
            if key == Tag.SMART_OBJECT_LAYER_DATA1:
                key = Tag.SMART_OBJECT_LAYER_DATA2
                block = TaggedBlock(signature=block.signature, key=key, data=block.data)
            blocks[key] = block
    descriptor = _config(layer).data
    if pixels is not None:
        size = descriptor.get(b"Sz  ")
        if not isinstance(size, Descriptor):
            size = Descriptor(classID=b"Pnt ")
            descriptor[b"Sz  "] = size
        size[b"Wdth"] = Double(float(pixels[0]))
        size[b"Hght"] = Double(float(pixels[1]))
    if corners is not None:
        for key in (b"Trnf", b"nonAffineTransform"):
            values = descriptor.get(key)
            if values is not None and len(values) == 8:
                for i, value in enumerate(corners):
                    values[i] = Double(value)
        for key in _PLACED_TAGS:
            placed = blocks.get_data(key)
            if placed is not None:
                placed.transform = tuple(corners)


def _swap_item(psd: PSDProtocol, item: LinkedLayer, external: LinkedLayer) -> None:
    """Put *external* in the document's ``lnkE`` in place of embedded *item*.

    An emptied embedded block goes; a new ``lnkE`` follows the last embedded
    block, where Photoshop writes it.
    """
    blocks = psd.tagged_blocks
    if blocks is None:
        raise ValueError("The document holds no smart object data")
    rebuilt: list[tuple[object, TaggedBlock]] = []
    after, signature = None, b"8BIM"
    for key, block in blocks.items():
        if key in _LIST_TAGS:
            signature = block.signature
            kept = [it for it in block.data if it is not item]
            if len(kept) < len(block.data):
                block = TaggedBlock(
                    signature=signature, key=key, data=LinkedLayers(kept)
                )
            if kept:
                rebuilt.append((key, block))
            after = len(rebuilt)
        else:
            rebuilt.append((key, block))
    existing = next(
        (
            b
            for k, b in rebuilt
            if k == Tag.LINKED_LAYER_EXTERNAL and b.data is not None
        ),
        None,
    )
    if existing is not None:
        existing.data.append(external)
    else:
        rebuilt = [(k, b) for k, b in rebuilt if k != Tag.LINKED_LAYER_EXTERNAL]
        items = LinkedLayers()
        items.append(external)
        block = TaggedBlock(
            signature=signature, key=Tag.LINKED_LAYER_EXTERNAL, data=items
        )
        at = len(rebuilt) if after is None else min(after, len(rebuilt))
        rebuilt.insert(at, (block.key, block))
    blocks.clear()
    for key, block in rebuilt:
        blocks[key] = block


def _has_filters(blocks: TaggedBlocks) -> bool:
    """Whether a layer's smart object data carries smart filters."""
    keys = [
        key
        for key in (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2)
        if key in blocks
    ]
    if len(keys) == 1:
        held = blocks[keys[0]]._held
        if held is not None and b"filterFX" not in held:
            return False  # unparsed, and no such key in its bytes
    for key in keys:
        config = blocks.get_data(key)
        if config is not None:
            return isinstance(config.data.get(b"filterFX"), Descriptor)
    return False


def _filter_items(layer: LayerProtocol) -> Sequence[FilterEffect]:
    """Return the items of *layer*'s document's ``FEid`` block; empty without one."""
    blocks = getattr(getattr(layer, "_psd", None), "tagged_blocks", None)
    if blocks is None:
        return []
    for key in _FILTER_TAGS:
        items = blocks.get_data(key)
        if items is not None:
            return items
    return []


def _contains(item: FilterEffect, bbox: tuple[int, int, int, int]) -> bool:
    """Whether the item's rectangle contains *bbox*."""
    if item.rectangle is None:
        return False
    top, left, bottom, right = item.rectangle
    return left <= bbox[0] and top <= bbox[1] and right >= bbox[2] and bottom >= bbox[3]


def _is_stored(plane: FilterEffectChannel | FilterEffectExtra | None) -> bool:
    """Whether *plane* carries compressed pixel data."""
    return (
        plane is not None
        and bool(plane.is_written)
        and plane.compression is not None
        and len(plane.data) > 0
    )


def _check_planes(
    psd: PSDProtocol,
    width: int,
    height: int,
    planes: Sequence[FilterEffectChannel | FilterEffectExtra],
    held: int,
) -> None:
    """Check decoding *planes* one at a time against the allocation budget.

    *held* is what stays live across the decodes.
    """
    pixels = width * height
    peak = 0
    for plane in planes:
        compression = _compression(plane)
        peak = max(
            peak,
            pixels * _DECOMPRESS_PEAK[compression]
            + decompress_row_peak_bytes(compression, height),
        )
    check_pixel_size(
        width,
        height,
        len(planes),
        max_alloc_bytes=psd._max_alloc_bytes,
        estimated_bytes=held + peak,
        warn=False,
    )


def _decode_plane(
    plane: FilterEffectChannel | FilterEffectExtra, width: int, height: int
) -> np.ndarray:
    """One 8-bit ``FEid`` plane as ``uint8 (height, width)``.

    :raises ValueError: if the plane does not decode.
    """
    compression = _compression(plane)
    version = _rle_version(plane.data, height) if compression == Compression.RLE else 1
    raw = _decompress(plane.data, compression, width, height, 8, version, strict=True)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def _compression(plane: FilterEffectChannel | FilterEffectExtra) -> Compression:
    """Return the plane's compression method."""
    if plane.compression is None:
        raise ValueError("FEid plane without a compression method")
    return Compression(plane.compression)


def _rle_version(data: bytes, height: int) -> int:
    """Return an RLE plane's row count width: version 2 (4 bytes) or 1 (2 bytes).

    Photoshop writes 4-byte counts, the PSB layout, in PSD files too, so the
    width is read off the payload: the right table sums to its length.
    """
    for version, size in ((2, 4), (1, 2)):
        if len(data) >= size * height:
            counts = np.frombuffer(data, dtype=f">u{size}", count=height)
            if size * height + int(counts.sum(dtype=np.int64)) == len(data):
                return version
    raise ValueError("RLE row counts do not match the payload length")
