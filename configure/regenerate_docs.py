"""Regenerate `docs/classes/*.rst` from `doc_classes/*.xml`.

Wraps godot's `make_rst.py`, which is fetched (along with its few helper
imports) into a temp directory at runtime. Without this wrapper:
  - `make_rst.py` lives in the godot tree under `doc/tools/` and imports
    `misc.utility.color` and `version`, so it can't be run standalone.
  - Its emitted `XML source: ...` URL comment is hardcoded to
    `github.com/godotengine/godot/tree/...` and built from
    `os.path.relpath(xml_file, root_directory)` — which depends on cwd, so
    every contributor's run produces a different (still-wrong) URL.

This script downloads the helpers, runs `make_rst.py`, then rewrites the
`XML source` line in each generated `.rst` to a stable URL pointing at
this repo. The result is byte-stable across machines and ready to commit.

Usage:
    tests/.venv/bin/python configure/regenerate_docs.py
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GODOT_RAW = "https://raw.githubusercontent.com/godotengine/godot/master"
HELPERS = [
	("doc/tools/make_rst.py", "make_rst.py"),
	("misc/utility/color.py", "misc/utility/color.py"),
	("version.py",            "version.py"),
]
# Stable URL prefix written into the .rst `XML source` comment. Points at the
# real repo; replaces make_rst.py's hardcoded github.com/godotengine/godot.
CANONICAL_URL_PREFIX = "https://github.com/Ivorforce/NumDot/tree/main"


def fetch_helpers(scratch: pathlib.Path) -> pathlib.Path:
	for remote, local in HELPERS:
		dest = scratch / local
		dest.parent.mkdir(parents=True, exist_ok=True)
		with urllib.request.urlopen(f"{GODOT_RAW}/{remote}") as r:
			dest.write_bytes(r.read())
	# Make the misc package importable.
	(scratch / "misc" / "__init__.py").touch()
	(scratch / "misc" / "utility" / "__init__.py").touch()
	return scratch / "make_rst.py"


def rewrite_urls(rst_dir: pathlib.Path) -> None:
	# Match the cwd-derived `XML source: <prefix>/.../doc_classes/X.xml` line
	# and replace the prefix with the canonical one.
	pattern = re.compile(
		r"^(\.\. XML source: )https://github\.com/godotengine/godot/tree/[^/]+/.*?(doc_classes/[^.]+\.xml\.)$",
		re.MULTILINE,
	)
	for rst in sorted(rst_dir.glob("*.rst")):
		text = rst.read_text()
		new = pattern.sub(rf"\g<1>{CANONICAL_URL_PREFIX}/\g<2>", text)
		if new != text:
			rst.write_text(new)


_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")
_NOISE_PATTERNS = (
	# Unresolved primitive types (Variant, bool, int, ...) — godot's full
	# class catalogue isn't shipped with NumDot, so these always trigger.
	re.compile(r"^ERROR: .*Unresolved type "),
	# The trailing "N errors were found" summary.
	re.compile(r"^\d+ errors were found in the class reference XML\."),
)


def _filter_noise(stream_bytes: bytes) -> bytes:
	out = []
	for line in stream_bytes.splitlines(keepends=True):
		stripped = _ANSI_RE.sub("", line.decode("utf-8", errors="replace"))
		if any(p.match(stripped) for p in _NOISE_PATTERNS):
			continue
		out.append(line)
	return b"".join(out)


def main() -> int:
	with tempfile.TemporaryDirectory(prefix="numdot_make_rst_") as tmp:
		scratch = pathlib.Path(tmp)
		make_rst = fetch_helpers(scratch)
		result = subprocess.run(
			[sys.executable, str(make_rst), "-o", "docs/classes", "-l", "en", "doc_classes"],
			cwd=REPO_ROOT,
			env={**__import__("os").environ, "PYTHONPATH": str(scratch)},
			capture_output=True,
		)
		sys.stdout.buffer.write(_filter_noise(result.stdout))
		sys.stderr.buffer.write(_filter_noise(result.stderr))
		# make_rst exits non-zero when it hits unresolved types but still
		# writes the .rst files. Treat that as OK.
		if result.returncode not in (0, 1):
			return result.returncode
	rewrite_urls(REPO_ROOT / "docs" / "classes")
	return 0


if __name__ == "__main__":
	sys.exit(main())
