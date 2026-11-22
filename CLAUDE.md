# CLAUDE.md

See also [the contributor documentation](docs/contributing.rst) for setup, test and docs commands.

## Development Notes

- Use `uv run python` / `uv run pytest` to use the development runtime.
- Rendering and compositing need the optional `composite` extra: `uv sync --extra composite`
  (`aggdraw`, `scipy`). It is optional because it is unavailable on some
  platforms. Tests needing it skip via `pytest.importorskip`.
- `uv run pytest --no-cov` skips coverage; `uv run pytest path::test_name` runs one test.
- Lint and types: `uv run ruff check`, `uv run ruff format`, `uv run mypy`.
- Work merges into `main` and is tagged `cards-YYYYMMDD` for AutoCroissant; no upstream PRs.

## Architecture Overview

### Two-Layer Design

**Low-level (`psd_tools.psd`)**: reads/writes raw PSD binary structures following Adobe's
specification. All classes use `attrs` and implement `read(fp)` and `write(fp)`.

**High-level (`psd_tools.api`)**: Pythonic interfaces. `PSDImage` wraps the low-level `PSD`
and reconstructs the layer tree from the flat layer record list.

Other subpackages: `psd_tools.composite` (rendering: blend modes, effects, vector
rasterization) and `psd_tools.compression` (Raw, RLE, ZIP; `_rle.pyx` is a Cython RLE codec
with a pure-Python fallback).

Import convention: internal code imports from the defining module
(`from psd_tools.psd.document import PSD`); public API uses the package
(`from psd_tools.psd import PSD`).

### Contracts

- **Read-only by design.** Mutating low-level structures under the API is out of contract.
  A stale cache after a *high-level* edit (`layer.name = ...`) is a real bug.
- **Compositor viewport and artboards** clip as described in
  [docs/architecture.rst](docs/architecture.rst); read it before changing either.
- **Unknown data round-trips.** Unrecognised tagged blocks and fields are preserved as bytes
  on write. Do not drop or reinterpret them.

### Layer Tree and Records

`PSDImage` rebuilds the layer tree from the flat record list. For 16- and 32-bit files
read records through `psd._record._get_layer_info()`, not `layer_info`. Details:
[docs/architecture.rst](docs/architecture.rst).

### BaseElement and attrs

All binary structures inherit from `BaseElement` and implement
`read(cls, fp, **kwargs) -> Self` and `write(self, fp, **kwargs) -> int`, so complex
structures compose recursively from simple ones. Data classes use `attrs`
(`@define(repr=False)`, `field(default=...)`); validators must match type hints.

### Tagged Blocks

Metadata lives in tagged blocks keyed by a 4-byte `Tag`; a registry maps tags to handler
classes via `@register(Tag.FOO)`.

```python
if Tag.UNICODE_LAYER_NAME in layer._record.tagged_blocks:
    name = layer._record.tagged_blocks.get_data(Tag.UNICODE_LAYER_NAME)
```

### Key Files

- `src/psd_tools/psd/document.py`: the `PSD` class (re-exported from `psd_tools.psd`).
- `src/psd_tools/psd/tagged_blocks.py`: tagged block registry and handlers.
- `src/psd_tools/psd/descriptor.py`: Adobe's descriptor format (key-value serialization).
- `src/psd_tools/terminology.py`: Adobe's 4-byte identifier mappings.

## Testing Conventions

- Tests mirror the package structure: `tests/psd_tools/psd/` for low-level,
  `tests/psd_tools/api/` for high-level. Fixture PSDs are in `tests/psd_files/`.
- Tests often parametrize over fixture files.
- Round-trip validation is common: read → modify → write → read → verify.

## Type Annotations

New code uses type hints on all signatures and `typing.Self` for methods
returning their own class.

`psd_tools.api` property getters return primitives, not descriptor wrappers, and `T | None`
for a missing key with no Photoshop default (#788); see
[docs/architecture.rst](docs/architecture.rst). Do not annotate `Any` for a value whose type
the low-level declaration knows.

## Code Comments and Docstrings

Keep them short, and document **current behavior only**. A comment earns its
length by explaining something the code cannot say itself — a non-obvious
constraint, a format quirk, why an obvious approach is wrong. Three things do
not belong in one:

- **History.** "used to", "was expected to fail at", "before #854". The
  before/after belongs in the commit message.
- **Measured figures.** MSE bounds, corpus statistics, error counts. They drift
  under the next change and leave a comment that reads as fact. A bound keeps
  its *reason* ("an aggdraw pen is not bit-stable across versions"), never its
  value.
- **The same explanation at three layers.** Say it once, where it belongs.

What stays: bare issue refs like `(#854)`, which point *into* that history;
fixture geometry such as "a 100x100 path", which is a property of the file; and
anything a test asserts.

Issue refs stay out of text a user reads: the `docs/` pages other than the
changelog, and docstrings, which render into the API reference. There `(#882)` is
plain text that links nowhere and means nothing to a reader. Use them in source
comments and the changelog.

To keep a number, name it in code rather than recite it in prose:

```python
stroke_color = 0.129  # PANTONE Black 3 C, red channel
assert color[row, column] == pytest.approx(share * stroke_color, abs=0.002)
```

That is checked on every run; the same number in a docstring rots silently.

Existing long comment blocks are the tree's current state, not a precedent to
match — see #891. Measure with `uv run python tools/prose_density.py`.

## Changelog and releases

`docs/changelog.rst` is history; changes add no entries. A release is a `cards-YYYYMMDD`
tag on `main`, which AutoCroissant's `requirements-cards.txt` pins.
