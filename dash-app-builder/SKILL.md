---
name: dash-app-builder
description: >
  Generate Dash analytics apps from Python scripts, Jupyter notebooks, or
  notebook URLs. Triggers: turn a notebook into a Dash app, build a dashboard,
  create an analytics UI, make code interactive with Dash, wrap a model in a
  Dash app, build a KPI dashboard, or any link to a .ipynb file when Dash is
  requested or an analytics dashboard is implied.
---

# Dash App Builder

Generate single-file Dash apps that wrap existing Python code. Sources include local `.py` files, `.ipynb` notebooks in project knowledge, or notebooks fetched from URLs.

## Step 1: Analyze Source Code

Scan for Python files and Jupyter notebooks. Identify functions (with parameters and return types), imports, data patterns (DataFrames, file I/O, model loading), and visualizations.

### Fetching Notebooks from URLs

When the user provides a notebook link:

1. **Resolve the raw URL:**
   - GitHub: `github.com/.../blob/.../*.ipynb` → `raw.githubusercontent.com/.../*.ipynb`
   - Colab: `colab.research.google.com/drive/<id>` → export URL
   - GitLab: append `?ref=main&format=json` or use raw endpoint
   - Other: use URL directly if it serves `.ipynb` JSON

2. **Download and extract code cells:**
   ```bash
   curl -L -o notebook.ipynb "<resolved_raw_url>"
   ```
   ```python
   import json

   with open("notebook.ipynb") as f:
       nb = json.load(f)

   code_cells = [
       "".join(cell["source"])
       for cell in nb["cells"]
       if cell["cell_type"] == "code"
   ]
   ```

3. Proceed with analysis on the extracted code.

## Step 2: Classify and Select Pattern

| Pattern | Indicators | UI Approach |
|---------|-----------|-------------|
| Analytics Dashboard | Aggregations, groupby, multiple charts, KPIs | Multi-chart layout with filters using `dbc` |
| ML Model | `sklearn`, `torch`, `keras`, `.predict()`, model files | Input form → prediction display |
| Data Processing | `pandas`, file I/O, DataFrame transforms | File upload → transform → download |
| Visualization | `matplotlib`, `plotly`, `seaborn`, `.plot()` | Interactive chart controls with callbacks |
| General Script | Functions with parameters | Form inputs → output display |

**DBC vs plain Dash:** Use `dash-bootstrap-components` when the app has multiple charts, KPI cards, or more than two input controls. For simple single-function apps, plain `dash.html` and `dash.dcc` components are fine.

## Step 3: Generate App

Follow these principles:
- Controls in a sidebar or top row → outputs in the main area
- Single-page apps with standard `@callback` decorators — no multi-page routing
- Preserve original function logic — wrap, don't rewrite
- Include type annotations on function signatures and key variables
- Add brief inline comments explaining UI and callback choices
- Load configuration from a `.env` file when present using `python-dotenv`
- Use Plotly for all charts (Dash-native)

### Environment Configuration

Generated apps should load settings (API keys, file paths, model paths, thresholds, etc.) from a `.env` file when one exists. Add this near the top of every generated app:

```python
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

# Example: read config with sensible defaults
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
API_KEY = os.getenv("API_KEY")
```

When generating an app, also create a `.env.example` file documenting all configurable values with placeholder defaults:

```
# .env.example
MODEL_PATH=model.pkl
API_KEY=your-api-key-here
```

### File Upload Helper

Dash file uploads arrive as base64 strings. Include this helper in any template that accepts file uploads:

```python
import base64
import io
import pandas as pd

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode a dcc.Upload payload into a DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
```

### Analytics Dashboard Template

Use DBC for multi-chart dashboards with linked filters:

```python
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv("data.csv")

def compute_kpis(filtered_df: pd.DataFrame) -> dict:
    return {
        "total": len(filtered_df),
        "mean_value": filtered_df["value"].mean(),
    }

def kpi_card(title: str, card_id: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="card-subtitle mb-2 text-muted"),
        html.H4(id=card_id),
    ]))

app.layout = dbc.Container([
    html.H1("Analytics Dashboard", className="my-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Category"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": c, "value": c} for c in df["category"].unique()],
                value=None,
                placeholder="All",
            ),
        ], md=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(kpi_card("Total Records", "kpi-total"), md=3),
        dbc.Col(kpi_card("Mean Value", "kpi-mean"), md=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="main-chart"), md=8),
        dbc.Col(dcc.Graph(id="side-chart"), md=4),
    ]),
], fluid=True)

@callback(
    Output("kpi-total", "children"),
    Output("kpi-mean", "children"),
    Output("main-chart", "figure"),
    Output("side-chart", "figure"),
    Input("category-filter", "value"),
)
def update_dashboard(category: str | None):
    filtered = df if category is None else df[df["category"] == category]
    kpis = compute_kpis(filtered)
    main_fig = px.scatter(filtered, x="x_col", y="value", color="category")
    side_fig = px.histogram(filtered, x="value")
    return f"{kpis['total']:,}", f"{kpis['mean_value']:.2f}", main_fig, side_fig

if __name__ == "__main__":
    app.run(debug=True)
```

Dashboard heuristics: KPIs → `dbc.Card`, filters → `dcc.Dropdown`/`dcc.RangeSlider`, multiple charts → `dbc.Row`/`dbc.Col` grid, date ranges → `dcc.DatePickerRange`.

### ML Model Template

```python
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
from functools import lru_cache

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

@lru_cache(maxsize=1)
def load_model():
    return joblib.load("model.pkl")

def predict(feature_1: float, feature_2: str) -> str:
    model = load_model()
    input_data = pd.DataFrame([[feature_1, feature_2]], columns=["f1", "f2"])
    return str(model.predict(input_data)[0])

app.layout = dbc.Container([
    html.H1("Model Prediction", className="my-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Feature 1"),
            dbc.Input(id="feature-1", type="number", value=0.0),
        ], md=4),
        dbc.Col([
            dbc.Label("Feature 2"),
            dcc.Dropdown(
                id="feature-2",
                options=[{"label": v, "value": v} for v in ["A", "B", "C"]],
                value="A",
            ),
        ], md=4),
    ], className="mb-3"),

    dbc.Button("Predict", id="predict-btn", color="primary", className="mb-3"),
    dbc.Alert(id="prediction-output", color="success", is_open=False),
], fluid=True)

@callback(
    Output("prediction-output", "children"),
    Output("prediction-output", "is_open"),
    Input("predict-btn", "n_clicks"),
    State("feature-1", "value"),
    State("feature-2", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks: int, feat1: float, feat2: str):
    return f"Prediction: {predict(feat1, feat2)}", True

if __name__ == "__main__":
    app.run(debug=True)
```

Widget heuristics: int → `dbc.Input(type="number", step=1)`, float → `dbc.Input(type="number", step=0.1)`, low cardinality (<10) → `dcc.Dropdown`, binary → `dbc.Switch`, free text → `dbc.Input(type="text")`, long text → `dbc.Textarea`.

### Data Processing Template

```python
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df  # [Insert transformation logic]
    return result

app.layout = dbc.Container([
    html.H1("Data Processor", className="my-3"),

    dcc.Upload(
        id="file-upload",
        children=dbc.Button("Upload CSV", color="secondary", className="mb-3"),
        accept=".csv",
    ),
    html.Div(id="input-preview"),

    dbc.Button("Process", id="process-btn", color="primary", className="mb-3"),
    html.Div(id="output-area"),
    dcc.Download(id="download-csv"),
], fluid=True)

@callback(
    Output("input-preview", "children"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def preview_upload(contents: str, filename: str):
    df = parse_upload(contents, filename)
    return dash_table.DataTable(
        data=df.head(10).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
    )

@callback(
    Output("output-area", "children"),
    Input("process-btn", "n_clicks"),
    State("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def run_processing(n_clicks: int, contents: str, filename: str):
    result = process_data(parse_upload(contents, filename))
    return html.Div([
        dash_table.DataTable(
            data=result.head(50).to_dict("records"),
            columns=[{"name": c, "id": c} for c in result.columns],
            style_table={"overflowX": "auto"},
        ),
        dbc.Button("Download CSV", id="download-btn", color="success", className="mt-2"),
    ])

@callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def download_result(n_clicks: int, contents: str, filename: str):
    result = process_data(parse_upload(contents, filename))
    return dcc.send_data_frame(result.to_csv, "result.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)
```

Parameter heuristics: column names → `dcc.Dropdown(options=df.columns)`, thresholds → `dcc.Slider` or `dbc.Input(type="number")`, multi-column → `dcc.Dropdown(multi=True)`, dates → `dcc.DatePickerSingle` or `dcc.DatePickerRange`.

### Visualization Template

```python
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

app.layout = dbc.Container([
    html.H1("Interactive Visualization", className="my-3"),

    dcc.Upload(
        id="file-upload",
        children=dbc.Button("Upload CSV", color="secondary"),
        accept=".csv",
    ),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="x-col", placeholder="X axis"), md=4),
        dbc.Col(dcc.Dropdown(id="y-col", placeholder="Y axis"), md=4),
        dbc.Col(dcc.Dropdown(id="color-col", placeholder="Color (optional)"), md=4),
    ], className="my-3"),

    dbc.Button("Plot", id="plot-btn", color="primary", className="mb-3"),
    dcc.Graph(id="chart"),
], fluid=True)

@callback(
    Output("x-col", "options"),
    Output("y-col", "options"),
    Output("color-col", "options"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def populate_columns(contents: str, filename: str):
    df = parse_upload(contents, filename)
    opts = [{"label": c, "value": c} for c in df.columns]
    return opts, opts, [{"label": "(none)", "value": ""}] + opts

@callback(
    Output("chart", "figure"),
    Input("plot-btn", "n_clicks"),
    State("file-upload", "contents"),
    State("file-upload", "filename"),
    State("x-col", "value"),
    State("y-col", "value"),
    State("color-col", "value"),
    prevent_initial_call=True,
)
def update_chart(n_clicks: int, contents: str, filename: str, x_col: str, y_col: str, color_col: str):
    df = parse_upload(contents, filename)
    return px.scatter(df, x=x_col, y=y_col, color=color_col or None)

if __name__ == "__main__":
    app.run(debug=True)
```

Chart heuristics: scatter → x, y, color, size; line → x, y, group; bar → category, value; histogram → column, bins; heatmap → x, y, value.

### General Script Template

For simple single-function wrappers, plain `dash.html` components are sufficient:

```python
import dash
from dash import dcc, html, Input, Output, State, callback

app = dash.Dash(__name__)

def run_script(param1: str, param2: int) -> str:
    result = main_function(param1, param2)
    return str(result)

app.layout = html.Div([
    html.H1("Script Name"),

    html.Label("Parameter 1"),
    dcc.Input(id="param1", type="text", value=""),

    html.Br(),
    html.Label("Parameter 2"),
    dcc.Input(id="param2", type="number", value=10),

    html.Br(),
    html.Button("Run", id="run-btn", n_clicks=0),

    html.Hr(),
    html.Div(id="output"),
])

@callback(
    Output("output", "children"),
    Input("run-btn", "n_clicks"),
    State("param1", "value"),
    State("param2", "value"),
    prevent_initial_call=True,
)
def execute(n_clicks: int, param1: str, param2: int):
    return run_script(param1, param2)

if __name__ == "__main__":
    app.run(debug=True)
```

For functions that return DataFrames, use `dash_table.DataTable`. For dicts/lists, use `html.Pre(json.dumps(data, indent=2))`. For images, encode as base64 and use `html.Img`.

### Component Quick Reference

| Need | Component |
|------|-----------|
| Short text input | `dcc.Input(type="text")` or `dbc.Input(type="text")` |
| Long text input | `dbc.Textarea()` |
| Integer | `dbc.Input(type="number", step=1)` |
| Float | `dbc.Input(type="number", step=0.1)` |
| Range/threshold | `dcc.Slider(min=0, max=100)` or `dcc.RangeSlider` |
| Single choice | `dcc.Dropdown(options=[...])` or `dcc.RadioItems` |
| Multiple choice | `dcc.Dropdown(options=[...], multi=True)` or `dcc.Checklist` |
| Boolean | `dbc.Checkbox` or `dbc.Switch` |
| Date picker | `dcc.DatePickerSingle` or `dcc.DatePickerRange` |
| File upload | `dcc.Upload` |
| Display table | `dash_table.DataTable` |
| Display text | `html.Div`, `html.P`, `dcc.Markdown` |
| Display chart | `dcc.Graph(figure=fig)` |
| Download file | `dcc.Download` + trigger callback |
| Layout grid | `dbc.Row` / `dbc.Col` |
| Feedback | `dbc.Alert(color="success"/"danger"/"warning"/"info")` |
| Spinner | `dbc.Spinner(children=...)` or `dcc.Loading` |
| Button | `dbc.Button(color="primary"/"secondary"/"success")` |
| Client state | `dcc.Store(id="...", storage_type="memory")` |

### Callback Wiring

Wire interactions with `@callback`. Use `Input` for triggers, `State` for passive values:

```python
# Button-triggered — use State for inputs that shouldn't trigger on their own
@callback(
    Output("result", "children"),
    Input("run-btn", "n_clicks"),
    State("input-field", "value"),
    prevent_initial_call=True,
)
def update(n_clicks: int, value: str):
    return process(value)

# Live filter — fires on every value change
@callback(
    Output("chart", "figure"),
    Input("category-filter", "value"),
    Input("date-range", "start_date"),
)
def update_chart(category: str, start: str):
    return px.scatter(filter_data(category, start), x="x", y="y")
```

## Step 4: Code Quality

Run ruff and ty (both from Astral) to lint, format, and type check. Use uv for project management.

```bash
pip install uv --break-system-packages

uv init --name dash-app
uv add dash plotly python-dotenv  # plus dependencies identified in Step 1
uv add dash-bootstrap-components  # omit if pattern uses plain dash.html only
uv add --dev ruff ty pytest

uv run ruff check --fix dash_app.py
uv run ruff format dash_app.py
uv run ty check dash_app.py
```

Fix all lint and type errors before delivering.

## Step 5: Testing

Create `test_dash_app.py` with pytest unit tests for all functions that can be tested independently of the Dash UI (data transforms, validation, utilities, prediction wrappers). Do **not** test Dash callbacks or layout — test the underlying Python functions directly.

```python
import pytest
from dash_app import process_data, compute_kpis, parse_upload

def test_process_data_basic():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = process_data(df)
    assert len(result) == 3

def test_compute_kpis():
    df = pd.DataFrame({"value": [10, 20, 30]})
    kpis = compute_kpis(df)
    assert kpis["total"] == 3
    assert kpis["mean_value"] == 20.0

def test_parse_upload_invalid():
    with pytest.raises(Exception):
        parse_upload(None, "bad.csv")
```

Guidelines: at least one test per function, cover happy path + edge cases + errors, use `pytest.fixture` for shared setup (e.g., sample DataFrames, loaded models), keep tests independent of Dash.

```bash
uv run pytest test_dash_app.py -v
```

Fix all failures before delivering.

## Output Checklist

- [ ] `dash_app.py` — single-file app with type annotations and inline comments
- [ ] `test_dash_app.py` — pytest unit tests for all testable functions
- [ ] `pyproject.toml` — uv-managed project with all dependencies
- [ ] `.env.example` — documents all configurable environment variables with defaults
- [ ] ruff check, ruff format, and ty check pass clean
- [ ] All pytest tests pass
