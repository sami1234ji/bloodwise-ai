"""Plotly figures for the dashboard.

The data shaping lives in dashboard/results.py and is tested separately; these
functions only turn that data into figures.
"""

from __future__ import annotations

from .results import STATUS_STYLE, category_summary, range_bar_data, status_counts

PALETTE = {"low": "#c8860d", "high": "#b3261e", "normal": "#1e6b45",
           "unknown": "#8b98a3"}
GRID = "#dfe7ec"
INK = "#123a5f"


# Chart headings live here rather than inside the figures. A Plotly title and a
# horizontal legend both sit at the top of the plot and overlap each other, so
# the interface writes the heading above the chart instead.
TITLES = {
    "status_donut": "Results by status",
    "category_bars": "Results by panel",
    "range_chart": "Where each result sits in its interval",
    "anomaly_gauge": "How unusual this combination is",
    "contributor_bars": "What drove the result",
}


def _base(fig, height: int, legend: bool = True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=38 if legend else 14, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#1d2a33"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def status_donut(findings):
    """Share of results by status."""
    import plotly.graph_objects as go
    counts = status_counts(findings)
    labels, values, colors = [], [], []
    for key in ("normal", "low", "high", "unknown"):
        if counts.get(key):
            labels.append(STATUS_STYLE[key]["label"].title())
            values.append(counts[key])
            colors.append(PALETTE[key])
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
                           marker=dict(colors=colors, line=dict(color="white", width=2)),
                           textinfo="value", sort=False))
    fig.add_annotation(text=f"<b>{counts['total']}</b><br>results", showarrow=False,
                       font=dict(size=15, color=INK))
    return _base(fig, 268)


def category_bars(findings):
    """Normal against abnormal for each panel."""
    import plotly.graph_objects as go
    rows = category_summary(findings)
    cats = [r["category"] for r in rows]
    fig = go.Figure()
    fig.add_bar(y=cats, x=[r["normal"] for r in rows], name="Inside interval",
                orientation="h", marker_color=PALETTE["normal"])
    fig.add_bar(y=cats, x=[r["low"] for r in rows], name="Low",
                orientation="h", marker_color=PALETTE["low"])
    fig.add_bar(y=cats, x=[r["high"] for r in rows], name="High",
                orientation="h", marker_color=PALETTE["high"])
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="number of tests", dtick=1)
    return _base(fig, 64 + 46 * max(len(cats), 1))


def range_chart(findings):
    """Each value placed against its own interval, drawn on one scale.

    The interval is mapped to 0 to 1 for every parameter, so values from tests
    with very different units can be compared at a glance. The shaded band is
    the interval; a marker outside it is a result outside the interval.
    """
    import plotly.graph_objects as go
    data = range_bar_data(findings)
    if not data:
        return None
    tests = [d["test"] for d in data]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1, fillcolor="#1e6b45", opacity=0.11,
                  line=dict(color="#1e6b45", width=1), layer="below")
    fig.add_vline(x=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=1, line=dict(color=GRID, width=1))
    for status in ("normal", "low", "high"):
        pts = [d for d in data if d["status"] == status]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["position"] for p in pts], y=[p["test"] for p in pts],
            mode="markers", name=STATUS_STYLE[status]["label"].title(),
            marker=dict(size=12, color=PALETTE[status],
                        line=dict(color="white", width=1.5),
                        symbol=["x" if p["clipped"] else "circle" for p in pts]),
            hovertext=[p["label"] for p in pts], hoverinfo="text+y"))
    fig.update_xaxes(title="position within the reference interval",
                     tickvals=[0, 0.5, 1], ticktext=["lower limit", "mid", "upper limit"],
                     range=[-0.5, 1.5])
    fig.update_yaxes(categoryorder="array", categoryarray=tests[::-1], automargin=True)
    return _base(fig, 118 + 24 * len(tests))


def anomaly_gauge(anomaly):
    """The Mahalanobis distance against its flagging threshold."""
    import plotly.graph_objects as go
    if anomaly is None or not anomaly.available or anomaly.mahalanobis is None:
        return None
    top = max(anomaly.mahalanobis, anomaly.mahalanobis_threshold) * 1.35
    colour = PALETTE["high"] if anomaly.is_unusual else PALETTE["normal"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly.mahalanobis,
        number=dict(font=dict(size=26, color=colour), valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, top]),
            bar=dict(color=colour, thickness=0.7),
            bgcolor="#eef2f5",
            borderwidth=0,
            threshold=dict(line=dict(color=INK, width=3), thickness=0.85,
                           value=anomaly.mahalanobis_threshold)),
    ))
    return _base(fig, 220, legend=False)


def contributor_bars(anomaly):
    """Which parameters drove the unusual-combination result."""
    import plotly.graph_objects as go
    if anomaly is None or not anomaly.contributors:
        return None
    items = list(reversed(anomaly.contributors))
    names = [c["parameter"].replace("_", " ") for c in items]
    vals = [c["contribution"] for c in items]
    colors = [PALETTE["high"] if c["direction"] == "above" else PALETTE["low"]
              for c in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[c["direction"] for c in items], textposition="auto"))
    fig.update_xaxes(title="share of the distance")
    return _base(fig, 70 + 36 * len(items), legend=False)
