"""Execute every ``python fenced block in README.md and docs/**/*.md.

Each page's blocks run in ONE shared namespace, in order -- the way a reader
following that page top to bottom would. A page is independent of every other
page, so a snippet may only rely on setup shown earlier on the same page.

Run: python scripts/check_doc_snippets.py [repo_root]
Exits non-zero if any block raises. This is a docs test, not a numerical one:
it proves the code in the docs still imports and runs, not that it converges.
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

fails = 0
for f in FILES:
    if not f.exists():
        continue
    text = f.read_text()
    ns = {"__name__": "__main__"}
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
                    fails += 1
                    print(f"\n=== FAIL {f}:{line} ({type(e).__name__}: {e})")
                    print("\n".join(f"    {l}" for l in code.split("\n")[:12]))
        finally:
            os.chdir(cwd)
print(f"\nfailing snippets: {fails}")
sys.exit(1 if fails else 0)
