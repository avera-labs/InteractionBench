"""Package the docs/ showcase site into a single self-contained HTML file (manifest + all audio inlined as
data URIs), dropped into a new folder outside the repo so it is easy to zip and send to the team/boss:
double-click to open, no local server, no internet needed.

Read-only copy; the original docs/ files are left untouched.
"""
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"
OUT = REPO.parent / "interactionbench-demo"        # new folder outside the repo (read-only copy of docs/, repo untouched)
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((DOCS / "data" / "manifest.json").read_text(encoding="utf-8"))

# inline every audio clip as a data URI
n_audio = 0
for exs in manifest["examples"].values():
    for ex in exs:
        for c in ex["cells"]:
            p = DOCS / c["audio"]
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            c["audio"] = "data:audio/mpeg;base64," + b64
            n_audio += 1

html = (DOCS / "index.html").read_text(encoding="utf-8")

# turn the manifest into JS that is safe to embed in <script> (escape </ so it can't close the script tag early)
mjson = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
inject = "window.__MANIFEST__=" + mjson + ";\n"

# 1) drop the fetch, use the inlined manifest instead
html = html.replace(
    'fetch("data/manifest.json",{cache:"no-store"}).then(r=>r.json()).then(M=>{',
    'Promise.resolve(window.__MANIFEST__).then(M=>{')
# 2) inject the inlined manifest at the top of the main script body (before anything uses it)
html = html.replace("const TASK_LABEL = {", inject + "const TASK_LABEL = {", 1)

out_file = OUT / "index.html"
out_file.write_text(html, encoding="utf-8")
mb = out_file.stat().st_size / 1e6
print(f"inlined {n_audio} audio clips → {out_file}  ({mb:.1f} MB)")
print(f"double-click to open, no server needed. Send this file (or a zip of the {OUT} folder) straight to the team.")
