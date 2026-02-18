---
name: gradio-app-builder
description: >
  Generate Gradio apps from Python scripts, Jupyter notebooks, or notebook
  URLs. Triggers: turn a notebook into a Gradio app, build an interactive
  demo, create a UI for a script, make code interactive with Gradio, wrap a
  model in a demo, or any link to a .ipynb file.
---

# Gradio App Builder

Generate single-file Gradio apps that wrap existing Python code. Sources include local `.py` files, `.ipynb` notebooks in project knowledge, or notebooks fetched from URLs.

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
| ML Model | `sklearn`, `torch`, `keras`, `.predict()`, model files | Prediction interface with `gr.Interface` |
| Data Processing | `pandas`, file I/O, DataFrame transforms | File upload → transform → download with `gr.Blocks` |
| Visualization | `matplotlib`, `plotly`, `seaborn`, `.plot()` | Interactive chart controls with `gr.Blocks` |
| General Script | Functions with parameters | Form inputs → output display with `gr.Interface` or `gr.Blocks` |

**Interface vs Blocks:** Use `gr.Interface` for simple single-function apps (one set of inputs → one set of outputs). Use `gr.Blocks` when you need multiple steps, tabs, conditional UI, or stateful interactions.

## Step 3: Generate App

Follow these principles:
- Use Gradio defaults — no custom CSS unless necessary
- Inputs on the left (or top) → outputs on the right (or bottom), following Gradio conventions
- Preserve original function logic — wrap, don't rewrite
- Include type annotations on function signatures and key variables
- Add brief inline comments explaining UI choices
- Load configuration from a `.env` file when present using `python-dotenv`

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

### ML Model Template

Use `gr.Interface` for straightforward predict-from-inputs patterns:

```python
import gradio as gr
import joblib
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    return joblib.load("model.pkl")

def predict(feature_1: float, feature_2: str) -> str:
    model = load_model()
    input_data = pd.DataFrame([[feature_1, feature_2]], columns=["f1", "f2"])
    prediction = model.predict(input_data)
    return str(prediction[0])

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1", value=0.0),
        gr.Dropdown(choices=["A", "B", "C"], label="Feature 2"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Model Prediction",
    flagging_mode="never",  # disable flagging for demos
)

if __name__ == "__main__":
    demo.launch()
```

Widget heuristics: int → `gr.Number(precision=0)`, float → `gr.Number()`, low cardinality (<10) → `gr.Dropdown()`, binary → `gr.Checkbox()`, free text → `gr.Textbox()`, long text → `gr.Textbox(lines=5)`.

### Data Processing Template

Use `gr.Blocks` for multi-step workflows with file I/O:

```python
import gradio as gr
import pandas as pd
import tempfile
import os

def process(file) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(file.name)
    result = df  # [Insert transformation logic]

    # Write result to a temp file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    result.to_csv(tmp.name, index=False)
    return result, tmp.name

with gr.Blocks(title="Data Processor") as demo:
    gr.Markdown("## Data Processor")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])

    with gr.Row():
        run_btn = gr.Button("Process", variant="primary")

    with gr.Row():
        table_out = gr.Dataframe(label="Result")
        download_out = gr.File(label="Download CSV")

    run_btn.click(
        fn=process,
        inputs=file_input,
        outputs=[table_out, download_out],
    )

if __name__ == "__main__":
    demo.launch()
```

Parameter heuristics: column names → `gr.Dropdown(choices=df.columns.tolist())`, thresholds → `gr.Slider()` or `gr.Number()`, multi-column → `gr.CheckboxGroup()`, dates → `gr.Textbox(label="Date (YYYY-MM-DD)")`.

### Visualization Template

```python
import gradio as gr
import pandas as pd
import plotly.express as px

def plot(file, x_col: str, y_col: str, color_col: str | None):
    df = pd.read_csv(file.name)
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col or None)
    return fig

with gr.Blocks(title="Interactive Visualization") as demo:
    gr.Markdown("## Interactive Visualization")

    file_input = gr.File(label="Upload CSV", file_types=[".csv"])

    with gr.Row():
        x_col = gr.Textbox(label="X axis column")
        y_col = gr.Textbox(label="Y axis column")
        color_col = gr.Textbox(label="Color by (optional)")

    plot_btn = gr.Button("Plot", variant="primary")
    chart_out = gr.Plot(label="Chart")

    plot_btn.click(fn=plot, inputs=[file_input, x_col, y_col, color_col], outputs=chart_out)

if __name__ == "__main__":
    demo.launch()
```

Prefer Plotly for interactivity (return a `plotly.graph_objects.Figure` to `gr.Plot`). For matplotlib, use `gr.Image` with `fig.savefig(buf, format="png")` and pass the buffer, or use `gr.Plot` which accepts matplotlib figures directly.

Chart heuristics: scatter → x, y, color, size; line → x, y, group; bar → category, value; histogram → column, bins; heatmap → x, y, value.

### General Script Template

```python
import gradio as gr

def run(param1: str, param2: int) -> str:
    result = main_function(param1, param2)
    return str(result)

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Parameter 1"),
        gr.Number(label="Parameter 2", value=10, precision=0),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Script Name",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
```

For functions that return DataFrames, use `gr.Dataframe` as the output component. For dicts/lists, serialize to JSON and use `gr.JSON`. For images, use `gr.Image`.

### Component Quick Reference

| Need | Component |
|------|-----------|
| Short text input | `gr.Textbox()` |
| Long text input | `gr.Textbox(lines=5)` |
| Integer | `gr.Number(precision=0)` |
| Float | `gr.Number()` |
| Range/threshold | `gr.Slider(minimum=0, maximum=100)` |
| Single choice | `gr.Dropdown(choices=[...])` or `gr.Radio(choices=[...])` |
| Multiple choice | `gr.CheckboxGroup(choices=[...])` |
| Boolean | `gr.Checkbox(label="...")` |
| File upload | `gr.File(file_types=[".csv"])` |
| Image upload | `gr.Image(type="pil")` |
| Audio upload | `gr.Audio()` |
| Display table | `gr.Dataframe()` |
| Display JSON | `gr.JSON()` |
| Display text | `gr.Textbox(interactive=False)` |
| Display image | `gr.Image()` |
| Display chart | `gr.Plot()` |
| Download file | `gr.File()` (as output) |
| Layout | `gr.Row()`, `gr.Column()`, `gr.Tab()`, `gr.Accordion()` |
| Markdown | `gr.Markdown("## heading")` |
| Button | `gr.Button(label, variant="primary"/"secondary"/"stop")` |
| State | `gr.State(default_value)` |

### Blocks Event Wiring

In `gr.Blocks`, wire interactions explicitly:

```python
# Simple click
btn.click(fn=my_fn, inputs=[inp1, inp2], outputs=[out1])

# Change event (re-runs on every keystroke / value change)
slider.change(fn=update_preview, inputs=slider, outputs=preview)

# Chained events
btn.click(fn=step1, inputs=inp, outputs=intermediate).then(
    fn=step2, inputs=intermediate, outputs=final_out
)
```

## Step 4: Code Quality

Run ruff and ty (both from Astral) to lint, format, and type check. Use uv for project management.

```bash
pip install uv --break-system-packages

uv init --name gradio-app
uv add gradio python-dotenv  # plus dependencies identified in Step 1
uv add --dev ruff ty pytest

uv run ruff check --fix gradio_app.py
uv run ruff format gradio_app.py
uv run ty check gradio_app.py
```

Fix all lint and type errors before delivering.

## Step 5: Testing

Create `test_gradio_app.py` with pytest unit tests for all functions that can be tested independently of the Gradio UI (data transforms, validation, utilities, prediction wrappers). Do **not** test Gradio component wiring — test the underlying Python functions directly.

```python
import pytest
from gradio_app import predict, process

def test_predict_basic():
    result = predict(1.0, "A")
    assert isinstance(result, str)

def test_process_edge_case():
    with pytest.raises(ValueError):
        process(None)
```

Guidelines: at least one test per function, cover happy path + edge cases + errors, use `pytest.fixture` for shared setup (e.g., sample DataFrames, loaded models), keep tests independent of Gradio.

```bash
uv run pytest test_gradio_app.py -v
```

Fix all failures before delivering.

## Output Checklist

- [ ] `gradio_app.py` — single-file app with type annotations and inline comments
- [ ] `test_gradio_app.py` — pytest unit tests for all testable functions
- [ ] `pyproject.toml` — uv-managed project with all dependencies
- [ ] `.env.example` — documents all configurable environment variables with defaults
- [ ] ruff check, ruff format, and ty check pass clean
- [ ] All pytest tests pass
