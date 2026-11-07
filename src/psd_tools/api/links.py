"""External links as repository paths.

A linked smart object names its file three ways: ``relPath`` from the PSD's
own folder, ``fullPath`` (a ``file://`` URL) and ``originalPath`` (a native
path). Here they are read as strings and placed in a repository by path,
never through the file system, so a PSD opened from bytes, a temporary file
or a partial checkout places its links alike.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar
from urllib.parse import unquote

from psd_tools.api._descriptor import get_scalar
from psd_tools.psd.descriptor import Descriptor, DescriptorBlock, Integer, String

#: The date every link written here carries. Photoshop flags a link as
#: modified when its file's date or size differs from the record.
LINK_DATE = (2000, 1, 1, 0, 0, 0.0)
#: The folder ``fullPath`` and ``originalPath`` name a repository file under.
LINK_ROOT = "/TTSCardMaker"

_CUT = posixpath.basename(LINK_ROOT).lower()
_D = TypeVar("_D", bound=Descriptor)
#: The record keys that name a link's file, in the order Photoshop tries them.
PATH_KEYS = (b"relPath", b"fullPath", b"originalPath")
_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|/)", re.IGNORECASE)
_CONTENT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_OID, "psd-tools.link.contentID")
#: Photoshop's 4-byte file type code by extension.
FILETYPES = {
    "png": b"png ",
    "jpg": b"JPEG",
    "jpeg": b"JPEG",
    "psb": b"8BPB",
    "psd": b"8BPS",
    "webp": b"WEBP",
    "gif": b"GIFf",
    "tif": b"TIFF",
    "tiff": b"TIFF",
}


def _fold(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


class RepoTree:
    """A repository's file listing, matched blind to case as Photoshop is.

    :param paths: repository paths, ``/``-separated, relative to its root.
    """

    def __init__(self, paths: Iterable[str]) -> None:
        self._paths: dict[str, str] = {}
        for path in paths:
            self._paths.setdefault(_fold(path), path)

    def find(self, path: str) -> str | None:
        """Return the listed spelling of *path*, or ``None`` when it is not listed."""
        return self._paths.get(_fold(path))

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and self.find(path) is not None

    def __len__(self) -> int:
        return len(self._paths)


def as_tree(tree: RepoTree | Iterable[str] | None) -> RepoTree | None:
    """*tree* as a :class:`RepoTree`; ``None`` stays ``None``."""
    if tree is None or isinstance(tree, RepoTree):
        return tree
    return RepoTree(tree)


def normal_path(path: str) -> str:
    """Return *path* with ``/`` separators and no leading ``./`` or ``/``."""
    return path.replace("\\", "/").lstrip("/").removeprefix("./")


def folder_of(psd_path: str) -> str:
    """Return the repository folder holding *psd_path*; ``""`` at the root."""
    return posixpath.dirname(normal_path(psd_path))


def _inside(path: str) -> str | None:
    """*path* normalised, or ``None`` when it leaves the repository."""
    path = posixpath.normpath(path)
    if path in ("", ".") or path == ".." or path.startswith(("../", "/")):
        return None
    return path


def is_relative(value: str) -> bool:
    """Whether a stored path is relative: no scheme, drive or leading slash."""
    value = value.replace("\x00", "").replace("\\", "/")
    return bool(value) and not _ABSOLUTE.match(value)


def candidates(value: str, folder: str | None) -> Iterator[str]:
    """Repository paths one stored path string can name, best first.

    A relative path joins *folder*, the PSD's repository folder, and may not
    climb above the root. An absolute path or ``file://`` URL is cut after its
    last ``/TTSCardMaker/``. A ``%``-escaped form is also tried unescaped.
    """
    value = value.replace("\x00", "").replace("\\", "/")
    if not value:
        return
    forms = [value]
    if "%" in value and unquote(value) != value:
        forms.append(unquote(value))
    for form in forms:
        if is_relative(form):
            if folder is not None:
                path = _inside(posixpath.join(folder, form))
                if path is not None:
                    yield path
            continue
        parts = form.split("/")
        cut = next(
            (i for i in range(len(parts) - 1, -1, -1) if parts[i].lower() == _CUT),
            None,
        )
        if cut is not None:
            path = _inside("/".join(parts[cut + 1 :]))
            if path is not None:
                yield path


def place(
    linked_file: Any,
    psd_path: str | None,
    tree: RepoTree | Iterable[str] | None = None,
) -> str | None:
    """Return the repository file an ``ExternalFileLink`` descriptor names.

    ``relPath`` is tried first, then ``fullPath`` and ``originalPath``, as
    Photoshop does. With *tree*, the first candidate it lists wins, in its
    spelling; without, the first candidate that is a repository path.

    :param linked_file: the link record's ``linked_file`` descriptor.
    :param psd_path: the linking PSD's repository path; ``None`` places
        absolute paths only.
    :param tree: the repository's files.
    :return: the repository path, or ``None`` when no candidate places.
    """
    folder = None if psd_path is None else folder_of(psd_path)
    finder = as_tree(tree)
    for key in PATH_KEYS:
        for path in candidates(get_scalar(linked_file, key, str, ""), folder):
            if finder is None:
                return path
            found = finder.find(path)
            if found is not None:
                return found
    return None


def link_content_id(blob: str, psd_path: str, unique_id: str) -> str:
    """Return the ``contentID`` a link written here carries.

    Repeatable for the same file blob, PSD path and item, and different for
    two items linking one file.

    :param blob: the linked file's git blob sha the raster was drawn from.
    :param psd_path: the linking PSD's repository path.
    :param unique_id: the link item's uuid.
    """
    key = "\n".join((blob, normal_path(psd_path), unique_id.strip("\x00")))
    return str(uuid.uuid5(_CONTENT_NAMESPACE, key))


def relative(path: str, psd_path: str) -> str:
    """Return ``relPath`` of repository file *path* from the PSD at *psd_path*."""
    target = normal_path(path).split("/")
    base = [part for part in folder_of(psd_path).split("/") if part]
    common = 0
    while common < min(len(target) - 1, len(base)) and target[common] == base[common]:
        common += 1
    return "/".join([".."] * (len(base) - common) + target[common:])


def _filled(descriptor: _D, pairs: Iterable[tuple[bytes, Any]]) -> _D:
    for key, value in pairs:
        descriptor[key] = value
    return descriptor


def linked_file(
    path: str, psd_path: str, name: str, root: str = LINK_ROOT
) -> DescriptorBlock:
    """Build an ``ExternalFileLink`` descriptor as Photoshop 2023 writes one."""
    full = f"{root.rstrip('/')}/{normal_path(path)}"
    return _filled(
        DescriptorBlock(name="\x00", classID=b"ExternalFileLink"),
        [
            (b"descVersion", Integer(2)),
            (b"Nm  ", String(name + "\x00")),
            (b"fullPath", String(full + "\x00")),
            (b"originalPath", String(full + "\x00")),
            (b"relPath", String(relative(path, psd_path) + "\x00")),
        ],
    )


def content_descriptor(content_id: str) -> DescriptorBlock:
    """Build a link record's ``contentID`` descriptor."""
    return _filled(
        DescriptorBlock(name="\x00"), [(b"contentID", String(content_id + "\x00"))]
    )


def comp_info() -> DescriptorBlock:
    """Build the ``compInfo`` an item records when its file has no layer comps."""
    info = _filled(
        Descriptor(), [(b"compID", Integer(-1)), (b"originalCompID", Integer(-1))]
    )
    return _filled(DescriptorBlock(name="\x00"), [(b"compInfo", info)])
