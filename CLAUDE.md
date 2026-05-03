# NumDot

NumDot is a numpy-like tensor math library for the Godot game engine.
NumDot imitates numpy's API where possible, using the `nd` class with static methods like `nd.add(a, b)`.

Find its documentation at `./docs/`. The source code is in `src/`.

## Testing & iteration

What follows is for working on the test infrastructure. Build setup and
project-wide conventions live in `Readme.md` / `CONTRIBUTING.md`.

### The two test surfaces

- **Legacy smoke**: `tests/run_tests.py` — generator-based; spawns Godot
  once, dumps `.npy` files, diffs against numpy. Works on Godot 4.3+.
- **TCP bridge stack**: `tests/bridge/` (transport + sanity tests),
  `tests/numdot_xp/` (Array API adapter), `tests/array-api-tests/`
  (upstream conformance suite, git submodule), `tests/array_api/`
  (skips/xfails/README). Requires Godot **4.4+**.

`tests/` is a Poetry project with an in-project `.venv`; run
`(cd tests && poetry install)` after pulling.

### Run commands

Both suites use `tests/.venv/bin/python` directly (poetry-managed). Set
the Godot path to a 4.4+ binary; many versions exist on this machine
under `/Applications/Godot-*.app/Contents/MacOS/Godot`.

Bridge sanity (fast, ~25s):

```bash
tests/.venv/bin/python -m pytest tests/bridge/ \
  --godot=/Applications/Godot-4.4.app/Contents/MacOS/Godot
```

Array-API steady state (must stay green; new failures = regressions):

```bash
PYTHONPATH=tests \
NUMDOT_GODOT=/Applications/Godot-4.4.app/Contents/MacOS/Godot \
ARRAY_API_TESTS_MODULE=numdot_xp \
ARRAY_API_TESTS_XFAIL_MARK=skip \
  tests/.venv/bin/python -m pytest tests/array-api-tests/array_api_tests/ \
    --ignore=tests/array-api-tests/array_api_tests/test_inspection_functions.py \
    --skips-file tests/array_api/skips.txt \
    --xfails-file tests/array_api/xfails.txt \
    --hypothesis-disable-deadline
```

Triage mode (lets `XPASSED` show up so you can de-xfail tests that have
started passing): drop the `ARRAY_API_TESTS_XFAIL_MARK=skip` line.
Iteration: add `--max-examples 3 -q --tb=no` for sub-second cycles on
one file.

### Iteration loop for conformance work

Goal: drive `xfails.txt` to zero and bring more files out of `skips.txt`.

1. Pick a target — either a file in `skips.txt` to enable, or a section
   in `xfails.txt` to clear.
2. Run that one file with low `--max-examples` for fast feedback.
3. For each failure, decide:
   - **Missing/wrong adapter shim** → edit `tests/numdot_xp/_funcs.py`.
   - **NumDot signature mismatch** → adjust the shim (NumDot is
     positional-only; Array API is keyword-heavy).
   - **NumDot bug or missing function** → file an issue, leave xfailed.
4. Move the line out of `xfails.txt` (or add it with a section header
   when introducing a new failure).
5. Re-run the steady-state command above; commit.

Always re-run the bridge sanity suite after touching `tests/bridge/` or
`demo/tests/bridge.gd`.

### Architecture facts that are easy to forget

- `nd` in GDScript is a **class with only static methods** — not an
  instance. `nd.has_method(...)` and `nd.callv(...)` are *parse errors*.
  Use `ClassDB.class_has_method("nd", name)` and
  `Callable(ClassDB, "class_call_static").callv(["nd", name, ...args])`.
  `class_call_static` is **Godot 4.4+ only**.
- GDScript's `JSON.parse_string` parses *every* number as `float`. The
  wire protocol uses explicit `$int` / `$ints` arg tags so int-ness
  survives — required for shape/axis args to `zeros`/`reshape`/etc.
- The bridge runs Godot in a child process. **NumDot segfaults kill
  Godot, not pytest.** `BridgeClient` detects the dead pipe, marks
  itself dead, and `numdot_xp` respawns transparently before the next
  test. No need for `pytest-xdist` for isolation.
- The bridge lives entirely in `demo/tests/bridge.gd` (pure GDScript).
  No bridge code is in the GDExtension — zero risk of leaking into
  shipped binaries.
- Wire format (v1.1): `[u32 LE header_len][JSON header][blob 0]...[blob N]`.
  Header carries optional `"blobs": [size, ...]` listing blob sizes in
  order. Zero-blob frames are byte-identical to the original v0 format.

### Probing what `nd` exposes

Cheaper than reading C++ source when you just need "does this method
exist?":

```bash
cat > /tmp/probe.gd <<'EOF'
extends SceneTree
func _init():
    for name in ["FUNC1", "FUNC2"]:
        print(name, ": ", ClassDB.class_has_method("nd", name))
    quit()
EOF
/Applications/Godot-4.4.app/Contents/MacOS/Godot \
  --path demo --headless --script /tmp/probe.gd
```

Use `ClassDB.class_get_method_list("nd")` to enumerate everything.
