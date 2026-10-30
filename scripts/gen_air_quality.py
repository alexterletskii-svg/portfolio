import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import *

LIMIT = 35  # reference level, ug/m3
rng = np.random.default_rng(11)

# 1. DATA (simulated). For your own data build `raw`: hourly DatetimeIndex, one column per station.
idx = pd.date_range("2025-01-01", periods=24 * 120, freq="h")
n = len(idx)
hour, dow = idx.hour.values, idx.dayofweek.values
rush = 10 * np.exp(-((hour - 8) ** 2) / 6) + 13 * np.exp(-((hour - 18.5) ** 2) / 8)
weekend = np.where(dow >= 5, 0.7, 1.0)
season = 6 * np.sin(np.linspace(0, np.pi, n))
smoke = np.zeros(n)
smoke[1900:1960] = 70 * np.hanning(60)

stations = {"City centre": (12, 1.0), "Ring road": (20, 1.3), "Park": (7, 0.5)}
raw = pd.DataFrame(
    {name: level + k * rush * weekend + season + smoke + rng.gamma(2, 2.5, n)
     for name, (level, k) in stations.items()},
    index=idx,
)
raw = raw.mask(rng.random(raw.shape) < 0.03)                  # gaps
raw.iloc[rng.choice(n, 10, replace=False), 1] = 400           # spikes, ring road
raw.iloc[3000:3072, 2] = 5.0                                  # park sensor stuck for 72 h


# 2. CLEANING
def flag_stuck(s, hours=12):
    flat = (s.rolling(hours).max() == s.rolling(hours).min()).to_numpy().astype(float)
    cover = pd.Series(flat[::-1]).rolling(hours, min_periods=1).max().to_numpy()[::-1] > 0
    return pd.Series(cover, index=s.index)


clean = raw.copy()
log = {"missing": 0, "spikes": 0, "stuck": 0}
for c in raw:
    s = raw[c]
    stuck = flag_stuck(s)
    r = s - s.rolling(25, center=True, min_periods=10).median()
    spike = (r.abs() > 6 * 1.4826 * r.abs().median()) & ~stuck
    clean[c] = s.mask(stuck | spike)
    log["missing"] += int(s.isna().sum())
    log["spikes"] += int(spike.sum())
    log["stuck"] += int(stuck.sum())

# 3. ANALYSIS
daily = clean.resample("D").mean()
net = daily.mean(axis=1)
hot = net[net > LIMIT].index
ep0, ep1 = (hot.min(), hot.max()) if len(hot) else (None, None)
typical, peak = net.median(), net.max()

if ep0 is not None:
    ep_days = pd.date_range(ep0 - pd.Timedelta(days=1), ep1 + pd.Timedelta(days=1))
    normal = clean[~clean.index.normalize().isin(ep_days)]
else:
    normal = clean

b = normal["Ring road"]
heat = b.groupby([b.index.dayofweek, b.index.hour]).mean().unstack()
wd = b[(b.index.dayofweek < 5) & (b.index.hour == 8)].mean()
we = b[(b.index.dayofweek >= 5) & (b.index.hour == 8)].mean()
share = (daily > LIMIT).mean() * 100

removed, total = sum(log.values()), raw.size
findings = [
    f"Cleaning removed {removed} of {total} readings ({removed / total * 100:.1f}%): "
    f"{log['missing']} gaps, {log['spikes']} sensor spikes and {log['stuck']} hours "
    f"from a stuck sensor (Park, found by a flat-line rule).",
]
if ep0 is not None:
    findings.append(
        f"One network-wide episode, {ep0:%d %b} to {ep1:%d %b}: the daily mean peaked at "
        f"{peak:.0f} µg/m³, {peak / typical:.1f} times a typical day.")
findings += [
    f"At 08:00 the ring road reads {wd:.0f} µg/m³ on weekdays and {we:.0f} on weekends, "
    f"{100 * (wd / we - 1):.0f}% higher (episode days excluded).",
    f"{share.idxmax()} is above {LIMIT} µg/m³ on {share.max():.0f}% of days; "
    f"the park on {share['Park']:.0f}%.",
]

# 4. DASHBOARD
colors = {"City centre": BLUE, "Ring road": ORANGE, "Park": SLATE}
fig = make_subplots(
    rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
    row_heights=[0.5, 0.5], vertical_spacing=0.17, horizontal_spacing=0.1,
    subplot_titles=("Daily mean PM2.5 by station", "Ring road, typical week (µg/m³)",
                    f"Days above {LIMIT} µg/m³, %"),
)
for c in daily:
    fig.add_trace(go.Scatter(x=daily.index, y=daily[c], name=c, mode="lines",
                             line=dict(color=colors[c], width=2)), row=1, col=1)
fig.add_hline(y=LIMIT, line_dash="dot", line_color=INK, line_width=1, row=1, col=1)
if ep0 is not None:
    fig.add_vrect(x0=ep0.strftime("%Y-%m-%d"),
                  x1=(ep1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                  fillcolor=ORANGE, opacity=0.12, line_width=0, row=1, col=1)

fig.add_trace(go.Heatmap(
    z=heat.values, x=heat.columns, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    colorscale=[[0, "#eef2f5"], [0.5, "#7fa8c9"], [1, "#1f4f75"]],
    colorbar=dict(thickness=10, len=0.38, y=0, yanchor="bottom", x=0.465, xanchor="left"),
    hovertemplate="%{y} %{x}:00, %{z:.0f} µg/m³<extra></extra>"), row=2, col=1)
fig.add_trace(go.Bar(x=list(share.index), y=share.values,
                     marker_color=[colors[c] for c in share.index],
                     text=[f"{v:.0f}%" for v in share.values], textposition="outside",
                     showlegend=False), row=2, col=2)
fig.update_xaxes(title_text="Hour of day", dtick=3, row=2, col=1)
fig.update_yaxes(autorange="reversed", row=2, col=1)
fig.update_yaxes(range=[0, share.max() * 1.3 + 1], row=2, col=2)
fig.update_yaxes(title_text="µg/m³", row=1, col=1)

save(fig, "air-quality", 820,
     "Air quality, four months of hourly data. Simulated.", findings,
     legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)"))