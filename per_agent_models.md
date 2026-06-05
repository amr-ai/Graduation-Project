# Per-Agent Model Selection — Implementation

Each agent in Smart Analyst can now use a different LLM model.
Users pick per-agent models from the sidebar expander "Per-Agent Settings".

---

## Files Modified (12 files)

### 1. `streamlit_app.py` — Sidebar UI
- Replaced single `st.selectbox("Model", ...)` with **5 per-agent selectors**:
  - **Planner** → `st.session_state.planner_model`
  - **Coder** → `st.session_state.coder_model`
  - **Viz Coder** → `st.session_state.viz_model`
  - **Dashboard Generator** → `st.session_state.dashboard_model`
  - **Insights Agent** → `st.session_state.insights_model`
- All wrapped in `st.expander("Per-Agent Settings", expanded=True)`.

### 2. `core/state.py` — GraphState
- Removed: `model_name: str`
- Added:
  - `planner_model: str` — model for the planner agent
  - `coder_model: str` — model for the code-generation agent

### 3. `agents/visualization/viz_models.py` — VizState
- Added: `model: str` — model for the viz coder agent (was hardcoded before)

### 4. `agents/cleaning/planner.py`
- Changed `state.get("model_name", ...)` → `state.get("planner_model", ...)`

### 5. `agents/cleaning/coder.py`
- Changed `state.get("model_name", ...)` → `state.get("coder_model", ...)`

### 6. `agents/visualization/viz_graph.py`
- Removed hardcoded `model="llama-3.3-70b-versatile"`
- Added `model = state.get("model", "llama-3.3-70b-versatile")` and passes it to the API call

### 7. `agents/visualization/dashboard.py`
- Changed `st.session_state.get("model", ...)` → `st.session_state.get("dashboard_model", ...)`

### 8. `agents/insights/agent.py`
- Changed `st.session_state.get("model", ...)` → `st.session_state.get("insights_model", ...)`

### 9. `pages/1_Data_Cleaning.py`
- `_base_state()`: parameter changed from `mdl` to `planner_mdl, coder_mdl`; writes `planner_model` + `coder_model` instead of `model_name`
- Both call sites updated to pass `planner_model` + `coder_model`
- Chat widget reads `planner_model` instead of `model`

### 10. `pages/2_Visualization.py`
- Viz graph state now includes `"model": st.session_state.get("viz_model", "llama-3.3-70b-versatile")`

### 11. `pages/3_Business_Insights.py`
- Follow-up chat reads `insights_model` instead of `model`

### 12. `main.py` (CLI entry point)
- Both `planner_graph.invoke()` and `cleaning_graph.invoke()` use `planner_model` + `coder_model` instead of `model_name`

---

## Model Key → Agent Mapping

| Session state key       | Agent / Component              |
|-------------------------|--------------------------------|
| `planner_model`         | Cleaning plan generator        |
| `coder_model`           | Cleaning code generator        |
| `viz_model`             | Chat-driven chart generator    |
| `dashboard_model`       | Auto-dashboard generator       |
| `insights_model`        | Business insights report agent |

Each falls back to `"llama-3.3-70b-versatile"` if not set.
