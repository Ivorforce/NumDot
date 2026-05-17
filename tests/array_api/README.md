# NumDot ↔ array-api-tests

Running the [`array-api-tests`](https://github.com/data-apis/array-api-tests)
conformance suite against NumDot via the `numdot_xp` adapter (which
calls through the test bridge to a long-lived Godot subprocess).

## Layout

- `../numdot_xp/` — adapter package that exposes NumDot through the Array API surface.
- `../bridge/` — TCP bridge to Godot (used by the adapter under the hood).
- `../array-api-tests/` — git submodule with the upstream test suite.
- `skips.txt` — test-IDs to never collect (test files we haven't tackled yet).
- `xfails.txt` — tests within the *enabled* files that are known to fail today.

## Running

The suite needs Godot **4.4+** (for `ClassDB.class_call_static`).

```bash
cd /Users/lukas/dev/godot/NumDot

# Steady-state run — everything not in xfails must pass.
PYTHONPATH=tests \
NUMDOT_GODOT=/Applications/Godot-4.4.app/Contents/MacOS/Godot \
ARRAY_API_TESTS_MODULE=numdot_xp \
ARRAY_API_TESTS_XFAIL_MARK=skip \
  tests/.venv/bin/python -m pytest tests/array-api-tests/array_api_tests/ \
    --skips-file tests/array_api/skips.txt \
    --xfails-file tests/array_api/xfails.txt \
    --hypothesis-disable-deadline
```

`ARRAY_API_TESTS_XFAIL_MARK=skip` makes Hypothesis skip xfailed tests
instead of running them (≈4-5× faster). For *triage* runs (looking for
xfails that have started passing), drop that env var so they run as
plain `xfail` and `XPASSED` shows up in the summary.

For fast iteration during development, lower the example count:

```bash
... pytest ... --max-examples 3
```

## Iteration loop

1. **Pick a category.** Either: (a) a whole file in `skips.txt` to bring
   online, or (b) a section in `xfails.txt` to drive to zero.
2. **Run focused.** Pass `tests/array-api-tests/array_api_tests/test_<file>.py`
   directly to pytest, with low `--max-examples` for quick feedback.
3. **Triage each failure.** Decide:
   - **Missing adapter shim** → add a function to `tests/numdot_xp/_funcs.py`.
   - **Signature mismatch** → adjust the shim (NumDot's positional args may
     not match Array API's keyword args).
   - **NumDot bug** → file an issue, leave xfailed with a TODO comment.
   - **Spec disagreement** → discuss before deciding to change NumDot.
4. **Move the line.** Successful tests come out of `xfails.txt`; new
   failures (with a clear root cause) get added with a section header.
5. **Run the full suite.** Confirm steady-state stays green; commit.

## Crash isolation

Hypothesis will sometimes generate inputs that crash NumDot. Because the
heavy work runs in the Godot subprocess (not the pytest process), a
NumDot segfault kills Godot but pytest sees only a `ConnectionError`.
The `BridgeClient` marks itself dead; `numdot_xp` respawns Godot before
the next test. The crashing test is reported as failed, then the run
continues. **Do not need `pytest-xdist` or process isolation** — that
isolation comes for free from the bridge architecture.

## Known limitations of the current adapter

- Reductions ignore `axis`/`keepdims`/`dtype` kwargs — we only call the
  no-axis path (`NotImplementedError` for the others).
- `meshgrid` doesn't exist in NumDot.
- Reflected operator forms (`scalar + array`) aren't wired; not exercised
  by the conformance suite today, easy to add when needed.

These are all addressable in follow-up slices.
