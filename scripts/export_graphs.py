"""Export every LangGraph in app.graph.builder as mermaid .md + .png.

    python scripts/export_graphs.py [outdir]   # default: docs/graphs

ponytail: PNG goes through mermaid.ink (needs internet). Offline? pip install
pygraphviz and swap draw_mermaid_png() for draw_png().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import builder  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/graphs")
out.mkdir(parents=True, exist_ok=True)

for name in sorted(n for n in dir(builder) if n.startswith("build_")):
    graph = getattr(builder, name)().get_graph()
    mermaid = graph.draw_mermaid()
    (out / f"{name}.md").write_text(f"# {name}\n\n```mermaid\n{mermaid}```\n", encoding="utf-8")
    try:
        (out / f"{name}.png").write_bytes(graph.draw_mermaid_png())
        print(f"{name}: md + png")
    except Exception as e:
        print(f"{name}: md only (png failed: {e})")
