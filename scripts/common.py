from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "dashboards"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, SLATE = "#2b6a99", "#d4701f", "#6f7c8a"
INK, PAPER, RULE = "#1d2733", "#f6f7f5", "#d5dae0"
FONT = '"Atkinson Hyperlegible", system-ui, sans-serif'

CSS = """<style>
@font-face{font-family:"Atkinson Hyperlegible";font-weight:400;src:url("../fonts/atkinson-hyperlegible-latin-400-normal.woff2") format("woff2")}
@font-face{font-family:"Atkinson Hyperlegible";font-weight:700;src:url("../fonts/atkinson-hyperlegible-latin-700-normal.woff2") format("woff2")}
body{margin:0;background:#f6f7f5}
.fx{font:400 15px/1.55 "Atkinson Hyperlegible",system-ui,sans-serif;color:#1d2733;padding:16px 20px 0;max-width:780px}
.fx p{font-weight:700;font-size:17px;margin:0 0 8px}
.fx ul{margin:0;padding-left:1.1em}
.fx li{margin:0 0 6px}
</style>"""


def save(fig, name, height, heading, findings, **layout):
    """Записывает dashboards/dashboard_<name>.html: блок выводов сверху, графики под ним."""
    opts = dict(
        height=height, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    opts.update(layout)
    fig.update_layout(**opts)
    fig.update_xaxes(gridcolor=RULE)
    fig.update_yaxes(gridcolor=RULE)

    path = OUT / f"dashboard_{name}.html"
    fig.write_html(path, include_plotlyjs="directory",
                   config={"displaylogo": False, "responsive": True})

    html = path.read_text(encoding="utf-8")
    items = "".join(f"<li>{t}</li>" for t in findings)
    box = f'<div class="fx"><p>{heading}</p><ul>{items}</ul></div>'
    html = html.replace("</head>", CSS + "</head>", 1).replace("<body>", "<body>" + box, 1)
    path.write_text(html, encoding="utf-8")
    print("saved:", path)