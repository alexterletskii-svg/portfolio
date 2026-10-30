import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import *

rng = np.random.default_rng(3)
# * = negatively worded in the questionnaire, reverse-scored below
ITEMS = ["The material was clear", "The pace suited me", "I did not feel lost*",
         "Feedback helped me improve", "I would recommend the course",
         "The platform was not frustrating*"]
REVERSED = [2, 5]
GROUPS = ["Undergraduate", "Postgraduate"]

# 1. DATA (simulated). For your own data: array `resp` (respondents x items, values 1-5, NaN = blank)
#    and array `group` with one label per respondent.
N = 240
group = rng.choice(GROUPS, N, p=[0.6, 0.4])
trait = rng.normal(0, 1, N) + np.where(group == "Postgraduate", 0.45, 0.0)
load = np.array([0.9, 0.7, -0.8, 0.8, 1.0, -0.6])
latent = trait[:, None] * load + rng.normal(0, 0.8, (N, 6))
cuts = np.array([-1.4, -0.5, 0.4, 1.3])
resp = 1.0 + (latent[:, :, None] > cuts).sum(axis=2)
careless = rng.choice(N, 14, replace=False)
resp[careless] = rng.integers(1, 6, (14, 1))        # same answer to every item
resp[rng.random(resp.shape) < 0.01] = np.nan        # blanks

# 2. CLEANING
straight = np.nanstd(resp, axis=1) == 0
blank = np.isnan(resp).any(axis=1)
keep = ~straight & ~blank
n_straight, n_blank = int(straight.sum()), int((~straight & blank).sum())
X, g = resp[keep], group[keep]
scored = X.copy()
scored[:, REVERSED] = 6 - scored[:, REVERSED]


# 3. ANALYSIS
def alpha(m):
    k = m.shape[1]
    return k / (k - 1) * (1 - m.var(axis=0, ddof=1).sum() / m.sum(axis=1).var(ddof=1))


a_ok, a_raw = alpha(scored), alpha(X)

pct = np.array([[(scored[:, j] == v).mean() * 100 for v in range(1, 6)] for j in range(6)])
fav = pct[:, 3] + pct[:, 4]
order = np.argsort(fav)                 # worst first, so the best item is drawn on top
labels = list(np.array(ITEMS)[order])
P = pct[order]
neu = P[:, 2] / 2

comp = scored.mean(axis=1)
ug, pg = comp[g == GROUPS[0]], comp[g == GROUPS[1]]
diff = pg.mean() - ug.mean()
bs = np.array([rng.choice(pg, len(pg)).mean() - rng.choice(ug, len(ug)).mean()
               for _ in range(5000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
sp = np.sqrt(((len(ug) - 1) * ug.var(ddof=1) + (len(pg) - 1) * pg.var(ddof=1))
             / (len(ug) + len(pg) - 2))
d = diff / sp
grp = {name: np.array([(scored[g == name][:, j] >= 4).mean() * 100 for j in range(6)])
       for name in GROUPS}

findings = [
    f"Of {N} respondents, {n_straight} gave the same answer to every item and {n_blank} left an item blank; "
    f"{int(keep.sum())} remain.",
    f"Two items are worded negatively (*) and were reverse-scored. Cronbach's alpha is {a_ok:.2f} "
    f"after that and {a_raw:.2f} without it.",
    f"Most favourable: “{labels[-1]}” ({fav[order][-1]:.0f}% agree). "
    f"Least favourable: “{labels[0]}” ({fav[order][0]:.0f}%).",
    f"Postgraduates score {diff:.2f} points higher on the 5-point scale "
    f"(95% bootstrap interval {lo:.2f} to {hi:.2f}, d = {d:.2f}).",
]

# 4. DASHBOARD
def bar(name, x, color, cd, legend=True):
    return go.Bar(y=labels, x=x, orientation="h", name=name, marker_color=color,
                  customdata=cd, showlegend=legend,
                  hovertemplate="%{y}<br>" + name + ": %{customdata:.0f}%<extra></extra>")


NEUTRAL = "#cfd5db"
fig = make_subplots(
    rows=2, cols=1, row_heights=[0.58, 0.42], vertical_spacing=0.16,
    subplot_titles=("Responses by item, all respondents (neutral split around zero)",
                    "Agree or strongly agree, by group"),
)
for t in [
    bar("Neutral", -neu, NEUTRAL, P[:, 2]),
    bar("Disagree", -P[:, 1], "#e5b384", P[:, 1]),
    bar("Strongly disagree", -P[:, 0], ORANGE, P[:, 0]),
    bar("Neutral", neu, NEUTRAL, P[:, 2], legend=False),
    bar("Agree", P[:, 3], "#7fa8c9", P[:, 3]),
    bar("Strongly agree", P[:, 4], BLUE, P[:, 4]),
]:
    fig.add_trace(t, row=1, col=1)

m = int(np.ceil(max((neu + P[:, 1] + P[:, 0]).max(), (neu + P[:, 3] + P[:, 4]).max()) / 10) * 10)
ticks = list(range(-m, m + 1, 10))
fig.update_xaxes(range=[-m - 5, m + 5], tickvals=ticks,
                 ticktext=[f"{abs(v)}%" for v in ticks], zeroline=True,
                 zerolinecolor=INK, row=1, col=1)

for j in order:
    fig.add_trace(go.Scatter(x=[grp[GROUPS[0]][j], grp[GROUPS[1]][j]], y=[ITEMS[j]] * 2,
                             mode="lines", line=dict(color=RULE, width=3),
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
for name, col in zip(GROUPS, [BLUE, ORANGE]):
    fig.add_trace(go.Scatter(x=grp[name][order], y=labels, mode="markers", name=name,
                             marker=dict(color=col, size=11)), row=2, col=1)
fig.update_xaxes(range=[0, 100], title_text="% agree or strongly agree", row=2, col=1)
fig.update_yaxes(categoryorder="array", categoryarray=labels, automargin=True)

save(fig, "survey", 900,
     "Course survey, 6 items on a 5-point scale. Simulated.", findings,
     margin=dict(l=250, r=20, t=50, b=90), barmode="relative",
     legend=dict(orientation="h", x=0, y=-0.1))