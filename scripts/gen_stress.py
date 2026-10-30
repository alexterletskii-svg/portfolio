import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import *

rng = np.random.default_rng(42)
STATES = {"Baseline": (0, 0), "Cognitive stress": (8, -12), "Recovery": (3, -4)}  # shift in HR, RMSSD
N, W, MAX_RMSSD = 40, 10, 250

# 1. DATA (simulated). For your own data: DataFrame with columns participant, state, window, hr, rmssd
rows = []
for p in range(N):
    hr0, rm0 = rng.normal(70, 8), rng.normal(54, 12)
    for state, (dhr, drm) in STATES.items():
        for w in range(W):
            rows.append((p, state, w,
                         hr0 + dhr + rng.normal(0, 4),
                         max(rm0 + drm + rng.normal(0, 4), 5)))
df = pd.DataFrame(rows, columns=["participant", "state", "window", "hr", "rmssd"])
n0 = len(df)
df.loc[rng.random(n0) < 0.03, "hr"] = np.nan        # missing heart rate
df.loc[rng.random(n0) < 0.01, "rmssd"] = 400        # sensor artefacts

# 2. CLEANING
bad_na = df.hr.isna()
bad_art = ~bad_na & (df.rmssd > MAX_RMSSD)
clean = df[~bad_na & ~bad_art]
removed = n0 - len(clean)

# 3. ANALYSIS: one mean per participant and state
per = clean.groupby(["participant", "state"])[["hr", "rmssd"]].mean().reset_index()
rm = per.pivot(index="participant", columns="state", values="rmssd")[list(STATES)]
hr = per.pivot(index="participant", columns="state", values="hr")[list(STATES)]

change = (rm["Cognitive stress"] - rm["Baseline"]).dropna().to_numpy()
recov = (rm["Recovery"] - rm["Baseline"]).dropna().to_numpy()
hr_up = (hr["Cognitive stress"] - hr["Baseline"]).dropna().to_numpy()

boot = np.array([rng.choice(change, len(change)).mean() for _ in range(10_000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
dz = change.mean() / change.std(ddof=1)
means = rm.mean()

findings = [
    f"Cleaning removed {removed} of {n0} windows ({removed / n0 * 100:.1f}%): "
    f"{int(bad_na.sum())} with missing heart rate and {int(bad_art.sum())} with RMSSD above "
    f"{MAX_RMSSD} ms (sensor artefacts).",
    f"Mean RMSSD: {means['Baseline']:.1f} ms at baseline, {means['Cognitive stress']:.1f} ms under stress, "
    f"{means['Recovery']:.1f} ms in recovery.",
    f"Change from baseline to stress: {change.mean():+.1f} ms (95% bootstrap interval {lo:.1f} to {hi:.1f}, "
    f"n = {len(change)}, dz = {dz:.1f}). RMSSD fell in {(change < 0).sum()} of {len(change)} participants.",
    f"Heart rate rose by {hr_up.mean():.1f} bpm on average; in recovery RMSSD is still "
    f"{abs(recov.mean()):.1f} ms below baseline.",
]

# 4. DASHBOARD
line = {"Baseline": BLUE, "Cognitive stress": ORANGE, "Recovery": SLATE}
fill = {"Baseline": "rgba(43,106,153,.35)", "Cognitive stress": "rgba(212,112,31,.35)",
        "Recovery": "rgba(111,124,138,.35)"}

fig = make_subplots(
    rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]],
    row_heights=[0.45, 0.55], vertical_spacing=0.17, horizontal_spacing=0.1,
    subplot_titles=("Heart rate by state (bpm)", "RMSSD by state (ms)",
                    "RMSSD of each participant (ms), thick line is the mean"),
)
for col_idx, var in [(1, "hr"), (2, "rmssd")]:
    for s in STATES:
        fig.add_trace(go.Box(y=clean.loc[clean.state == s, var], name=s,
                             marker_color=line[s], line=dict(color=line[s]),
                             fillcolor=fill[s], boxpoints="outliers",
                             showlegend=False), row=1, col=col_idx)

for p, row in rm.iterrows():
    fig.add_trace(go.Scatter(x=list(STATES), y=row.to_numpy(), mode="lines",
                             line=dict(color=SLATE, width=1), opacity=0.35,
                             name=f"Participant {p + 1}", showlegend=False,
                             hovertemplate="%{x}: %{y:.0f} ms<extra>" + f"Participant {p + 1}" + "</extra>"),
                  row=2, col=1)
fig.add_trace(go.Scatter(x=list(STATES), y=means.to_numpy(), mode="lines+markers", name="Mean",
                         line=dict(color=ORANGE, width=4), marker=dict(size=9),
                         showlegend=False), row=2, col=1)

save(fig, "stress", 820,
     "Stress response in wearable data. Simulated: a 12 ms drop was built in, so this shows the method, not a finding.",
     findings)