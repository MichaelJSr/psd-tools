"""A document saved without its merged image, as Photoshop with Maximize Compatibility off."""

import io

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.constants import Compression

from ..utils import full_name


def test_a_stripped_document_saves_stripped_after_an_edit() -> None:
    pytest.importorskip("aggdraw")
    pytest.importorskip("scipy")
    psd = PSDImage.open(full_name("clipping-mask4.psd"))
    header = psd._record.header
    alpha = psd._record.image_data.get_data(header)[3]
    drawn = np.asarray(psd.composite(ignore_preview=True))
    psd.strip_merged()
    assert not psd.has_preview() and psd.is_stripped()
    assert psd._record.image_data.compression == Compression.RLE
    planes = psd._record.image_data.get_data(header)
    assert planes[:3] == [b"\xff" * (header.width * header.height)] * 3
    assert planes[3] == alpha  # extra channels keep their planes
    assert np.array_equal(np.asarray(psd.composite()), drawn)

    stripped = io.BytesIO()
    psd.save(stripped, update_preview=False)
    back = PSDImage.open(io.BytesIO(stripped.getvalue()))
    assert back.is_stripped()  # read off the white planes
    back.mark_updated()  # an edit
    edited = io.BytesIO()
    back.save(edited)
    assert PSDImage.open(io.BytesIO(edited.getvalue())).is_stripped()
    assert len(edited.getvalue()) - len(stripped.getvalue()) < 16

    back.save(io.BytesIO(), preview=back.composite())
    assert not back.is_stripped()
