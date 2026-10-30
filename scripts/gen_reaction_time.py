import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import *

rng = np.random.default_rng(7)

# 1. DATA (simulated). For your own data: DataFrame with columns subject, group, trial, rt
parts = []
for group, shift, drift in [("Control", 0.0, -0.0005), ("Fatigue", 0.10, 0.0030)]:
    for p in range(30):
        trial = np.arange(60)
        log_rt = (np.log(420) + shift + rng.normal(0, 0.12)
                  + drift * trial + rng.normal(0, 0.22, 60))
        parts.append(pd.DataFrame({"subject": f"{group}-{p:02d}", "group": group,
                                   "trial": trial, "rt": np.exp(log_rt)}))
df = pd.concat(parts, ignore_index=True)
u = rng.random(len(df))
fast, slow = u < 0.01, (u >= 0.01) & (u < 0.02)
df.loc[fast, "rt"] = rng.uniform(60, 140, fast.sum())
df.loc[slow, "rt"] = rng.uniform(1600, 3000, slow.sum())

# 2. CLEANING
n0 = len(df)
df = df[(df.rt >= 150) & (df.rt <= 1500)].copy()
dropped = n0 - len(df)

# 3. ANALYSIS: one median per participant
per = df.groupby(["group", "subject"]).rt.median().reset_index()
a = per.loc[per.group == "Control", "rt"].to_numpy()
b = per.loc[per.group == "Fatigue", "rt"].to_numpy()
diff = b.mean() - a.mean()

boot = np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
                 for _ in range(10_000)])
lo, hi = np.percentile(boot, [2.5, 97.5])

pooled = np.concatenate([a, b])
perm = np.empty(10_000)
for i in range(len(perm)):
    q = rng.permutation(pooled)
    perm[i] = q[len(a):].mean() - q[:len(a)].mean()
p_val = ((np.abs(perm) >= abs(diff)).sum() + 1) / (len(perm) + 1)
p_txt = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
             / (len(a) + len(b) - 2))
d = diff / sp

df["block"] = df.trial // 10 + 1
pb = df.groupby(["group", "subject", "block"]).rt.median().reset_index()
stats = pb.groupby(["group", "block"]).rt.agg(["mean", "sem"]).reset_index()
slope = {g: np.polyfit(s["block"], s["mean"], 1)[0] for g, s in stats.groupby("group")}

findings = [
    f"Cleaning dropped {dropped} of {n0} trials ({dropped / n0 * 100:.1f}%): "
    f"anticipations under 150 ms and lapses over 1500 ms.",
    f"Participant medians differ by {diff:.0f} ms (95% bootstrap interval {lo:.0f} to {hi:.0f}); "
    f"permutation test {p_txt}; Cohen's d = {d:.1f}.",
    f"The fatigue group also slows down within the session: {slope['Fatigue']:+.0f} ms per block "
    f"of 10 trials, against {slope['Control']:+.0f} ms in the control group.",
    "One median per participant keeps trials from the same person from counting as independent evidence.",
]

# 4. DASHBOARD
colors = {"Control": BLUE, "Fatigue": ORANGE}
fig = make_subplots(
    rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]],
    row_heights=[0.5, 0.5], vertical_spacing=0.17, horizontal_spacing=0.1,
    subplot_titles=("Median reaction time per participant (ms)",
                    "Difference between groups, 10,000 bootstrap resamples (ms)",
                    "Mean of participant medians, by block of 10 trials (ms, ±1 SE)"),
)
for g, col in colors.items():
    sub = per[per.group == g]
    fig.add_trace(go.Box(y=sub.rt, name=g, boxpoints="all", jitter=0.45, pointpos=0,
                         marker=dict(color=col, size=6, opacity=0.65),
                         line=dict(color=col), fillcolor="rgba(0,0,0,0)",
                         showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=boot, nbinsx=50, marker_color=SLATE, showlegend=False),
              row=1, col=2)
fig.add_vrect(x0=lo, x1=hi, fillcolor=ORANGE, opacity=0.15, line_width=0, row=1, col=2)
fig.add_vline(x=0, line_dash="dot", line_color=INK, row=1, col=2)
for g, col in colors.items():
    s = stats[stats.group == g]
    fig.add_trace(go.Scatter(
        x=s["block"], y=s["mean"], mode="lines+markers", name=g,
        error_y=dict(type="data", array=s["sem"].to_numpy(), visible=True),
        line=dict(color=col, width=2.5), marker=dict(size=7)), row=2, col=1)
fig.update_xaxes(title_text="Block of 10 trials", dtick=1, row=2, col=1)

save(fig, "reaction-time", 800,
     "Reaction time in two groups. Simulated: the effect was built in, so this shows the method, not a finding.",
     findings, legend=dict(x=0.01, y=0.2, bgcolor="rgba(0,0,0,0)"))