"""A gradient's colour lookup is ``interp1d``'s, without importing scipy.interpolate."""

import subprocess
import sys

import numpy as np
import pytest

from psd_tools.composite.paint import _RAMP_SAMPLES, _Lookup

from ..utils import full_name

interpolate = pytest.importorskip("scipy.interpolate")


@pytest.mark.parametrize("samples", [2, 3, 17, _RAMP_SAMPLES])
@pytest.mark.parametrize("columns", [None, 1, 3, 4])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_the_lookup_is_interp1ds(samples: int, columns: int | None) -> None:
    rng = np.random.default_rng(samples)
    x = np.linspace(0.0, 1.0, samples)
    size = (samples,) if columns is None else (samples, columns)
    table = rng.random(size).astype(np.float32)
    expected = interpolate.interp1d(
        x, table, axis=0, bounds_error=False, fill_value=(table[0], table[-1])
    )
    lookup = _Lookup(x, table)
    special = [0.0, -0.0, 1.0, np.nan, np.inf, -np.inf, x[1], x[-2], -0.2, 1.2]
    positions: list[object] = [
        rng.random((40, 30)) * 1.4 - 0.2,
        (rng.random((20, 10)) * 1.4 - 0.2).astype(np.float32),
        np.array(special),
        np.zeros((0,)),
        np.float64(0.37),
        rng.integers(0, 2, (5, 6)),
    ]
    for at in positions:
        want, got = expected(at), lookup(at)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()


def test_a_gradient_render_leaves_scipy_interpolate_unimported() -> None:
    script = (
        "import sys, numpy as np\n"
        "from psd_tools import PSDImage\n"
        "with np.errstate(all='ignore'):\n"
        "    PSDImage.open(sys.argv[1]).composite(force=True)\n"
        "print('scipy.interpolate' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, full_name("gradient-fill.psd")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-1] == "False"
