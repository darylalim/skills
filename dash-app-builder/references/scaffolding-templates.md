# Scaffolding templates

Function-split templates consumed by `SKILL.md` Step 3. Each template is a self-contained Python block, paste-ready into `dash_app.py` (T1+T2+T3+T4 in order) or `test_dash_app.py` (T5).

All templates assume:
- `DATASET_ID: str` is filled in at scaffold time from the user-supplied HF dataset URL slug.
- `MAX_ROWS: int` and `HF_TOKEN: str | None` are read from `.env`-loaded environment variables.
- `@lru_cache(maxsize=1)` from `functools` decorates `load_dataframe` so the dataset loads once per process.

## Template T1: Dataset loader + config

```python
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

DATASET_ID = "<org>/<dataset>"
MAX_ROWS = int(os.getenv("MAX_ROWS", "10000"))
HF_TOKEN = os.getenv("HF_TOKEN")

# Gated-dataset gate — emit only when dataset_info.gated is True
if not HF_TOKEN:
    raise SystemExit(
        f"Dataset `{DATASET_ID}` is gated. Set HF_TOKEN in .env "
        f"(get a token at https://huggingface.co/settings/tokens)."
    )


@lru_cache(maxsize=1)
def load_dataframe() -> pd.DataFrame:
    """Load the dataset's first MAX_ROWS rows as a pandas DataFrame."""
    ds = load_dataset(DATASET_ID, split=f"train[:{MAX_ROWS}]", token=HF_TOKEN)
    return ds.to_pandas()
```

The gated-gate block (the `if not HF_TOKEN: raise SystemExit(...)` lines) is omitted entirely when the dataset is not gated.

## Template T2: Filter-widget factory

```python
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc


def build_filter_for_column(df: pd.DataFrame, column: str):
    """Return a Dash filter widget (wrapped in dbc.Col) for the column, or None."""
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return dbc.Col([
            dbc.Label(column),
            dcc.RadioItems(
                id={"role": "filter", "col": column},
                options=[{"label": v, "value": v} for v in ["Any", "True", "False"]],
                value="Any",
                inline=True,
            ),
        ], md=3)
    if pd.api.types.is_numeric_dtype(series):
        lo, hi = float(series.min()), float(series.max())
        return dbc.Col([
            dbc.Label(column),
            dcc.RangeSlider(
                id={"role": "filter", "col": column},
                min=lo, max=hi, value=[lo, hi],
            ),
        ], md=4)
    if pd.api.types.is_datetime64_any_dtype(series):
        return dbc.Col([
            dbc.Label(column),
            dcc.DatePickerRange(id={"role": "filter", "col": column}),
        ], md=4)
    n_unique = series.nunique(dropna=True)
    if 1 < n_unique <= 20:
        opts = [{"label": str(v), "value": v} for v in sorted(series.dropna().unique())]
        return dbc.Col([
            dbc.Label(column),
            dcc.Dropdown(
                id={"role": "filter", "col": column},
                options=opts, multi=True,
            ),
        ], md=4)
    return None  # high-cardinality strings / unsupported dtypes get no filter
```

`is_bool_dtype` is checked **before** `is_numeric_dtype` because pandas treats bools as numeric.

## Template T3: Auto-chart heuristic

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def pick_chart(df: pd.DataFrame) -> go.Figure:
    """Pick a chart type from the DataFrame's columns. Filter-linked: caller passes
    the filtered slice."""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.columns
        if 2 <= df[c].nunique(dropna=True) <= 10 and c not in numeric_cols
    ]
    if numeric_cols:
        col = numeric_cols[0]
        if cat_cols:
            return px.histogram(df, x=col, color=cat_cols[0])
        return px.histogram(df, x=col)
    if cat_cols:
        col = cat_cols[0]
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        return px.bar(counts, x=col, y="count")
    return go.Figure().add_annotation(
        text="No chartable column", showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
```

## Template T4: Layout + callback wiring

<!-- skip-validate -->
```python
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = load_dataframe()
filter_widgets = [w for w in (build_filter_for_column(df, c) for c in df.columns) if w is not None]

app.layout = dbc.Container([
    html.H1(f"Dataset: {DATASET_ID}", className="my-3"),

    dbc.Row(filter_widgets, className="mb-3") if filter_widgets else html.Div(),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Rows", className="card-subtitle text-muted"),
            html.H4(id="kpi-rows"),
        ])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Columns", className="card-subtitle text-muted"),
            html.H4(id="kpi-cols"),
        ])), md=3),
    ], className="mb-3"),

    dash_table.DataTable(
        id="data-table",
        page_size=20,
        style_table={"overflowX": "auto"},
        data=df.head(50).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
    ),

    dcc.Graph(id="data-chart"),
], fluid=True)


@callback(
    Output("kpi-rows", "children"),
    Output("kpi-cols", "children"),
    Output("data-table", "data"),
    Output("data-chart", "figure"),
    Input({"role": "filter", "col": ALL}, "value"),
    State({"role": "filter", "col": ALL}, "id"),
)
def refresh(filter_values: list, filter_ids: list):
    """Apply each pattern-matched filter to the loaded DataFrame, then update KPIs,
    table, and chart. prevent_initial_call is left at the default False so the
    initial load renders the unfiltered view."""
    filtered = df.copy()
    for value, fid in zip(filter_values, filter_ids, strict=True):
        col = fid["col"]
        if value is None:
            continue
        series = filtered[col]
        if pd.api.types.is_numeric_dtype(series) and isinstance(value, list) and len(value) == 2:
            lo, hi = value
            filtered = filtered[(series >= lo) & (series <= hi)]
        elif pd.api.types.is_bool_dtype(series):
            if value == "True":
                filtered = filtered[series]
            elif value == "False":
                filtered = filtered[~series]
        elif isinstance(value, list) and value:
            filtered = filtered[series.isin(value)]
    return (
        f"{len(filtered):,}",
        f"{len(filtered.columns)}",
        filtered.head(50).to_dict("records"),
        pick_chart(filtered),
    )


if __name__ == "__main__":
    app.run(debug=True)
```

## Template T5: `test_dash_app.py` skeleton

```python
"""Unit tests for dash_app helper functions. Tests the underlying Python only —
no Dash callback or layout testing."""
import os
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

# Bypass T1's gated-gate (when emitted) by setting HF_TOKEN before dash_app is
# lazily imported inside each test function below.
os.environ.setdefault("HF_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def _stub_load_dataframe():
    """Prevent the real dataset load on first dash_app import. Each test
    passes its own DataFrame to the function under test, so the mock's
    return value is just a placeholder."""
    with patch("dash_app.load_dataframe", return_value=pd.DataFrame()):
        yield


def _df_numeric_only() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})


def _df_categorical_only() -> pd.DataFrame:
    return pd.DataFrame({"label": ["a", "b", "a", "c", "b"]})


def _df_mixed() -> pd.DataFrame:
    return pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "label": ["a", "b", "a", "b", "a"],
    })


def test_build_filter_numeric():
    from dash_app import build_filter_for_column
    df = _df_mixed()
    widget = build_filter_for_column(df, "value")
    assert widget is not None
    # dbc.Col -> first child is dbc.Label, second is dcc.RangeSlider
    inner = widget.children[1]
    assert type(inner).__name__ == "RangeSlider"


def test_build_filter_low_cardinality_categorical():
    from dash_app import build_filter_for_column
    df = _df_mixed()
    widget = build_filter_for_column(df, "label")
    assert widget is not None
    inner = widget.children[1]
    assert type(inner).__name__ == "Dropdown"


def test_build_filter_high_cardinality_returns_none():
    from dash_app import build_filter_for_column
    df = pd.DataFrame({"s": [f"v{i}" for i in range(100)]})
    assert build_filter_for_column(df, "s") is None


def test_pick_chart_numeric_column():
    from dash_app import pick_chart
    fig = pick_chart(_df_numeric_only())
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"


def test_pick_chart_numeric_plus_categorical():
    from dash_app import pick_chart
    fig = pick_chart(_df_mixed())
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"
    # color creates multiple traces
    assert len(fig.data) >= 2


def test_pick_chart_categorical_only():
    from dash_app import pick_chart
    fig = pick_chart(_df_categorical_only())
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


def test_pick_chart_empty_dataframe():
    from dash_app import pick_chart
    fig = pick_chart(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    # the empty-data branch adds an annotation; figure must not raise
    assert any("No data" in (a.text or "") for a in fig.layout.annotations)


def test_load_dataframe_is_cached():
    from dash_app import load_dataframe
    # lru_cache exposes __wrapped__ on the wrapper
    assert hasattr(load_dataframe, "__wrapped__")
```
