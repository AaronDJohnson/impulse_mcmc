"""Execute every ``python fenced block in README.md and docs/**/*.md.

Each page's blocks run in ONE shared namespace, in order -- the way a reader
following that page top to bottom would. A page is independent of every other
page, so a snippet may only rely on setup shown earlier on the same page.

Run: python scripts/check_doc_snippets.py [repo_root]
Exits non-zero if any block raises. This is a docs test, not a numerical one:
it proves the code in the docs still imports and runs, not that it converges.

Snippets that need an OPTIONAL extra (``[flow]``) are skipped when that extra
is not installed, and the skips are reported. They are not silently passed:
a run that skipped something says so, so "0 failing" can never be mistaken for
"everything was checked". Install the extras to check those blocks too.
"""

import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FILES = [ROOT / "README.md"] + sorted(
    p for p in (ROOT / "docs").rglob("*.md") if "_build" not in str(p)
)
FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)

#: Optional third-party packages a snippet may legitimately require. A missing
#: one is a skip; anything else missing is a genuine failure (a real import
#: typo in the docs must not be excused).
OPTIONAL_DEPS = {"coppuccino", "matplotlib"}


def _missing_optional(exc: BaseException) -> str | None:
    """Name of the missing optional package this exception is about, if any.

    Matches both a bare ``ModuleNotFoundError`` and the package's own guarded
    ``ImportError`` (``flow_proposals`` raises a friendly message naming
    coppuccino rather than letting the import escape).
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name in OPTIONAL_DEPS:
        return exc.name
    if isinstance(exc, ImportError):
        text = str(exc)
        for dep in OPTIONAL_DEPS:
            if dep in text:
                return dep
    return None


fails = 0
skips: list[str] = []
for f in FILES:
    if not f.exists():
        continue
    text = f.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__main__"}
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            for m in FENCE.finditer(text):
                code, line = m.group(1), text[: m.start()].count("\n") + 1
                if code.lstrip().startswith(">>>"):
                    continue
                try:
                    exec(compile(code, f"{f}:{line}", "exec"), ns)
                except Exception as e:
                    dep = _missing_optional(e)
                    if dep is not None:
                        skips.append(f"{f}:{line} (needs {dep})")
                        continue
                    fails += 1
                    print(f"\n=== FAIL {f}:{line} ({type(e).__name__}: {e})")
                    print("\n".join(f"    {ln}" for ln in code.split("\n")[:12]))
        finally:
            os.chdir(cwd)

for s in skips:
    print(f"SKIP {s}")
print(f"\nfailing snippets: {fails}   skipped (optional deps missing): {len(skips)}")
sys.exit(1 if fails else 0)
