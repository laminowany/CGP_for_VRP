import pandas as pd
import re
import plotly.graph_objects as go

# --- helper: wyciąga float z "tensor(4.9219)"
def parse_tensor(x):
    if isinstance(x, str):
        m = re.search(r"tensor\(([-+]?\d*\.?\d+)\)", x)
        if m:
            return float(m.group(1))
    return float(x)

# --- wczytanie danych
df1 = pd.read_csv("/home/piotr/repos/magisterka/outputs/run_20260326T224320/epochs.csv")
df2 = pd.read_csv("/home/piotr/repos/magisterka/outputs/csv/vrp10_20_epochs_epoch_128000.csv")

# --- czyszczenie score
df1["score"] = df1["score"].apply(parse_tensor)
df2["score"] = df2["score"].apply(parse_tensor)

# --- oznaczamy źródło (żeby rozróżnić linie)
df1["source"] = "with_time"
df2["source"] = "no_time"

# --- łączymy
df = pd.concat([df1, df2], ignore_index=True)

# --- plot
fig = go.Figure()

for (osobnik, source), group in df.groupby(["osobnik", "source"]):
    fig.add_trace(go.Scatter(
        x=group["epoch"],
        y=group["score"],
        mode="lines+markers",
        name=f"osobnik {osobnik} ({source})"
    ))

fig.update_layout(
    title="Score vs Epoch (grouped by osobnik)",
    xaxis_title="Epoch",
    yaxis_title="Score"
)

fig.show()