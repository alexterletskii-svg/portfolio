import io
import lzma
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common import *

LIMIT = 15      # WHO 2021 guideline for the 24-h mean of PM2.5, ug/m3
RADIUS_KM = 30  # sensors are picked within this distance of the best-covered sensor
DAYS = 120      # length of the window with the best coverage
N_SENSORS = 3

NEED = ["timestamp", "sensor_id", "lat", "lon", "p2"]
DEFAULT = Path(__file__).resolve().parent.parent / "data" / "2016_pm-sensors_mean30min.csv"
path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)


# 1. DATA. Sensor.Community (Luftdaten.info), 2016, 30-minute means.
def sep_of(head):
    line = head.decode("utf-8", "ignore").splitlines()[0]
    print("First line of the file:", line)
    return ";" if line.count(";") > line.count(",") else ","


def keep(c):
    return str(c).strip().lower() in NEED


def read_csv_bytes(name, data):
    if name.endswith(".xz"):
        data = lzma.decompress(data)
    return pd.read_csv(io.BytesIO(data), sep=sep_of(data[:4000]), usecols=keep)


def load(p):
    if tarfile.is_tarfile(p):
        frames = []
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if m.isfile() and ".csv" in m.name:
                    frames.append(read_csv_bytes(m.name, tf.extractfile(m).read()))
        return pd.concat(frames, ignore_index=True)
    if p.endswith(".xz"):
        with open(p, "rb") as f:
            return read_csv_bytes(p, f.read())
    with open(p, "rb") as f:
        head = f.read(4000)
    return pd.read_csv(p, sep=sep_of(head), usecols=keep)


df = load(path)
df.columns = [str(c).strip().lower() for c in df.columns]
print("Columns used:", list(df.columns))
missing_cols = [c for c in NEED if c not in df.columns]
if missing_cols:
    sys.exit(f"Missing columns {missing_cols}. Send me the lines above and I will adjust the names.")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=["timestamp", "sensor_id"])
df["p2"] = pd.to_numeric(df["p2"], errors="coerce")


def km(lat1, lon1, lat2, lon2):
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


cover = df.groupby("sensor_id")["p2"].count().sort_values(ascending=False)
pos = df.groupby("sensor_id")[["lat", "lon"]].median()
anchor = cover.index[0]
dist = km(pos.loc[anchor, "lat"], pos.loc[anchor, "lon"], pos["lat"], pos["lon"])
chosen = [s for s in cover.index if dist[s] <= RADIUS_KM][:N_SENSORS]
print("Chosen sensors:", chosen)

raw = (df[df["sensor_id"].isin(chosen)]
       .pivot_table(index="timestamp", columns="sensor_id", values="p2", aggfunc="mean")
       .resample("h").mean())
raw.columns = [f"Sensor {c}" for c in raw.columns]

# keep the DAYS-day window with the most readings
per_day = raw.notna().resample("D").sum().sum(axis=1)
if len(per_day) > DAYS:
    end = per_day.rolling(DAYS).sum().idxmax()
    raw = raw.loc[end - pd.Timedelta(days=DAYS - 1): end + pd.Timedelta(days=1)]
n_days = len(raw) // 24


# 2. CLEANING
def flag_stuck(s, hours=12):
    flat = (s.rolling(hours).max() == s.rolling(hours).min()).to_numpy().astype(float)
    cover_ = pd.Series(flat[::-1]).rolling(hours, min_periods=1).max().to_numpy()[::-1] > 0
    return pd.Series(cover_, index=s.index)


clean = raw.copy()
log = {"missing": 0, "impossible": 0, "spikes": 0, "stuck": 0}
for c in raw:
    s = raw[c]
    bad_range = (s < 0) | (s > 500)          # outside what the sensor can report
    s1 = s.mask(bad_range)
    stuck = flag_stuck(s1)
    r = s1 - s1.rolling(25, center=True, min_periods=10).median()
    spike = (r.abs() > 6 * 1.4826 * max(r.abs().median(), 1.0)) & ~stuck
    clean[c] = s1.mask(stuck | spike)
    log["missing"] += int(s.isna().sum())
    log["impossible"] += int(bad_range.sum())
    log["spikes"] += int(spike.sum())
    log["stuck"] += int(stuck.sum())

# 3. ANALYSIS
daily = clean.resample("D").mean()
net = daily.mean(axis=1)
mid = net.rolling(3, center=True).mean().idxmax()          # the worst 3-day stretch
ep0, ep1 = mid - pd.Timedelta(days=1), mid + pd.Timedelta(days=1)
typical, peak = net.median(), net.loc[ep0:ep1].max()

ep_days = pd.date_range(ep0 - pd.Timedelta(days=1), ep1 + pd.Timedelta(days=1))
normal = clean[~clean.index.normalize().isin(ep_days)]

top = clean.mean().idxmax()                                 # the dirtiest sensor
b = normal[top]
heat = b.groupby([b.index.dayofweek, b.index.hour]).mean().unstack()
wd = b[(b.index.dayofweek < 5) & (b.index.hour == 8)].mean()
we = b[(b.index.dayofweek >= 5) & (b.index.hour == 8)].mean()
share = (daily > LIMIT).mean() * 100

removed, total = sum(log.values()), raw.size
findings = [
    f"Cleaning removed {removed} of {total} hourly readings ({removed / total * 100:.1f}%): "
    f"{log['missing']} gaps, {log['impossible']} impossible values, {log['spikes']} spikes "
    f"and {log['stuck']} hours from flat-lined sensors.",
    f"The worst stretch was {ep0:%d %b} to {ep1:%d %b}: the network mean reached "
    f"{peak:.0f} µg/m³, {peak / typical:.1f} times a typical day.",
    f"At 08:00 {top} reads {wd:.0f} µg/m³ on weekdays and {we:.0f} on weekends "
    f"({100 * (wd / we - 1):+.0f}%, worst stretch excluded).",
    f"{share.idxmax()} is above the WHO guideline of {LIMIT} µg/m³ on {share.max():.0f}% of days; "
    f"{share.idxmin()} on {share.min():.0f}%.",
    "These are low-cost optical sensors, so absolute levels are less reliable than at reference "
    "stations; the cleaning and the patterns are the point of the example.",
]

# 4. DASHBOARD
colors = dict(zip(clean.columns, [BLUE, ORANGE, SLATE]))
fig = make_subplots(
    rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
    row_heights=[0.5, 0.5], vertical_spacing=0.17, horizontal_spacing=0.1,
    subplot_titles=("Daily mean PM2.5 by sensor", f"{top}, typical week (µg/m³)",
                    f"Days above {LIMIT} µg/m³, %"),
)
for c in daily:
    fig.add_trace(go.Scatter(x=daily.index, y=daily[c], name=c, mode="lines",
                             line=dict(color=colors[c], width=2)), row=1, col=1)
fig.add_hline(y=LIMIT, line_dash="dot", line_color=INK, line_width=1, row=1, col=1)
fig.add_vrect(x0=ep0.strftime("%Y-%m-%d"), x1=(ep1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
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

save(fig, "air-quality-real", 820,
     f"Air quality, {n_days} days of hourly PM2.5 from {len(chosen)} real sensors (2016). "
     "Data: Sensor.Community (Luftdaten.info), CC BY 4.0, doi:10.5281/zenodo.4740389.",
     findings, legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)"))