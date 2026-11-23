# Agent guidance

## Project map

- `imod/mf6/` contains MODFLOW 6 simulations, models, discretizations, and
  packages; `imod/msw/` contains MetaSWAP functionality. `imod/wq/` is
  sunset and is no longer under active development.
- `imod/prepare/` contains model preparation and allocation utilities.
  `imod/formats/` handles input/output formats; `imod/select/`,
  `imod/evaluate/`, and `imod/visualize/` provide selection, evaluation, and
  visualization functionality.
- Structured grids are represented with xarray; unstructured grids use
  xugrid. MODFLOW 6 support includes structured DIS and vertex-based DISV.
  Follow the established patterns in the relevant module and nearby
  tests rather than introducing a new abstraction.
- Tests are organized by area under `imod/tests/`. Shared fixtures are in
  `imod/tests/conftest.py` and `imod/tests/fixtures/`.
- `docs/` contains the Sphinx documentation. The API reference is under
  `docs/api/`; runnable Sphinx-Gallery examples are under `examples/`.

## Environment and commands

Use Pixi to run project commands from the repository root:

| Purpose | Command |
| --- | --- |
| Lint and check formatting | `pixi run lint` |
| Apply Ruff fixes and formatting | `pixi run format` |
| Type-check | `pixi run mypy_lint` |
| Unit tests (JIT-disabled and JIT-specific) | `pixi run unittests` |
| Unit tests and examples | `pixi run tests` |
| Documentation build | `pixi run docs` |

For a focused test, run `pixi run pytest imod/tests/<path> -k <expression>`.
Use the full Pixi task when checking the corresponding full suite. The test
markers are `example`, `user_acceptance`, `unittest_jit`, and `qgis_export`;
see Testing guidance below for how the tasks handle them.

The `user-acceptance` environment needs external DVC data and credentials;
don't fetch or run it unless the task calls for it. Keep any credentials in
local DVC configuration, never in tracked files. QGIS export is a separate,
optional workflow and is not part of the ordinary test run.

Pixi defines `py312`, `py313`, and `py314` environments. For a change that
affects Python-version compatibility, use the relevant environment(s) and
`test_import` task; don't infer the supported range solely from the default
environment.

## Public API

Treat a name as public when it is intentionally exposed from a user-facing
namespace, such as `imod` or `imod.mf6`, or is listed in the API reference or
user-facing documentation. This includes documented classes and their
documented methods, functions, and attributes. An implementation detail is
not automatically public just because it is importable from its defining
module; underscore-prefixed names and undocumented internal helpers are
generally private.

When changing a public API, preserve backward compatibility unless the task
explicitly calls for a breaking change. Update the API docs, relevant examples,
and changelog. If it's unclear whether a symbol is public, check how it is
exported and documented and treat it as public until clarified.

## Code conventions

- Follow the surrounding code's naming, structure, and established helpers;
  prefer consistency over adding a new pattern or dependency.
- Keep production functions typed. The configured mypy check is strict, though
  selected modules have explicit exceptions in `pyproject.toml`.
- Write docstrings using the NumPy convention configured for Ruff. Keep public
  API documentation useful to readers and consistent with the relevant
  `docs/api/` page.
- Documentation is reStructuredText. Examples use Sphinx-Gallery conventions,
  including `# %%` cell markers; don't edit generated gallery output.

## Definition of done

- Add or update a focused test for behavior changes and bug fixes.
- Run the relevant tests and `pixi run lint`; run `pixi run mypy_lint` for
  production Python changes. Use broader suites when practical or when the
  change affects shared behavior.
- Update user-facing documentation and examples when behavior or a public API
  changes.
- Add user-visible changes to the `[Unreleased]` section of
  `docs/api/changelog.rst`, using its existing categories and style.
- Keep the change focused; report any relevant validation that could not be
  run.

## Testing guidance

- Put tests beside the corresponding area under `imod/tests/` and reuse
  fixtures from `conftest.py` or `fixtures/` where appropriate. The Pixi
  environment includes `pytest-cases` and `hypothesis`; follow existing test
  patterns when using them.
- `pixi run unittests` runs both the JIT-disabled and JIT-specific test tasks.
  `unittests_njit` excludes `example`, `user_acceptance`, `unittest_jit`, and
  `qgis_export`; `unittests_jit` runs the JIT-specific tests. Run
  `pixi run examples` for example tests.
- For a focused test, use
  `pixi run pytest imod/tests/<path> -k <expression>`. Run `pixi run tests`
  for the broader functional suite when practical.
- The examples task sets `IMOD_DATA_DIR` to `imod/tests/.imod_data`. Prefer
  existing fixtures and locally available test data; don't add tests that
  unexpectedly download large datasets.

## Gotchas and boundaries

- Read the relevant pages under `docs/developing/` for contribution,
  documentation, CI, or testing details before changing those workflows.
- Preserve dependency constraints unless a task requires changing them.
  In particular, `pixi.toml` pins pytest below 9.1 for a pytest-cases
  compatibility issue and excludes xarray 2025.04.0 due to an HFB rendering
  issue; confirm the reason before changing either constraint.
- On Windows, use `pixi run docs`; this task invokes `make.bat html` from the
  `docs/` directory.
- Do not hand-edit `pixi.lock` or generated documentation, including
  `docs/api/generated/`; use the project's tools when they need updating.
- User-acceptance tests require external DVC data and credentials. Do not fetch
  or run them unless requested, and keep credentials in local DVC
  configuration, never in tracked files. QGIS export is a separate, optional
  workflow; use it only when relevant to the change.
- Do not add a dependency or make a breaking public API change without explicit
  authorization. Avoid unrelated cleanup so changes remain focused and
  reviewable.
