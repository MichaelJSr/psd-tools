"""A group's shadow, its knockout and its stroke share one composite of its alpha."""

import importlib

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage

from ..utils import full_name

pytest.importorskip("scipy")

composite = importlib.import_module("psd_tools.composite.composite")


def _render(viewport: tuple[int, int, int, int] | None, force: bool) -> bytes:
    psd = PSDImage.open(full_name("layer-edit.psd"))
    with np.errstate(all="ignore"):
        image = psd.composite(
            viewport=viewport, force=force, ignore_preview=True, apply_icc=False
        )
    return image.tobytes()


@pytest.mark.parametrize("viewport", [None, (4, 30, 44, 60), (10, 2, 30, 20)])
@pytest.mark.parametrize("force", [True, False])
def test_the_text_group_alpha_is_composited_once(
    monkeypatch: pytest.MonkeyPatch,
    viewport: tuple[int, int, int, int] | None,
    force: bool,
) -> None:
    """``Text`` casts a shadow that its stroke knocks out, from one alpha."""
    composited = []
    grow = composite.Compositor._group_alpha_box

    def counted(self, layer, *args):  # type: ignore[no-untyped-def]
        composited.append(layer.name)
        return grow(self, layer, *args)

    monkeypatch.setattr(composite.Compositor, "_group_alpha_box", counted)
    shared = _render(viewport, force)
    assert composited.count("Text") == 1

    asked = []

    def unshared(self, layer, viewport, inner, passthrough):  # type: ignore[no-untyped-def]
        asked.append(layer.name)
        return inner

    monkeypatch.setattr(composite.Compositor, "_group_alpha_box", unshared)
    monkeypatch.setattr(
        composite.Compositor,
        "_held_group_alpha",
        lambda self, layer, inner: (inner, None),
    )
    assert _render(viewport, force) == shared
    assert asked.count("Text") > 1
