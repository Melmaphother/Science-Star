import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure styles can be imported when run from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from styles import apply_shared_style, hero_html

# streamlit run vis_output.py
st.set_page_config(
    page_title="Results Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_shared_style(st)
st.markdown(
    hero_html(
        "📊 Results Visualization",
        "HLE / GAIA evaluation dashboard — inspect predictions, metrics, and reasoning steps",
    ),
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<p class="filter-label" style="margin-bottom: 0.5rem;">📁 Data source</p>', unsafe_allow_html=True)
    jsonl_file = st.text_input(
        "JSONL file path",
        "/Users/melmaphother/Documents/Code/Science-Star/output/gpt-4o-mini-gaia/20260215_201521/answers.jsonl",
        help="Path to the answers JSONL file",
        label_visibility="collapsed",
    )

if jsonl_file and Path(jsonl_file).exists():
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if not data:
        st.warning("⚠️ File is empty or invalid format.")
    else:
        df = pd.DataFrame(data)
        # Normalize: use `id` as task_id if task_id not present (GAIA format)
        if "task_id" not in df.columns and "id" in df.columns:
            df["task_id"] = df["id"]
        elif "id" not in df.columns and "task_id" in df.columns:
            df["id"] = df["task_id"]
        # Extract is_correct from judgment_result for GAIA format
        if "judgment_result" in df.columns:
            df["is_correct"] = df["judgment_result"].apply(
                lambda x: x.get("is_correct") if isinstance(x, dict) else None
            )
        # Filters
        st.markdown('<p class="section-title">⚙️ Filters</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cat_col = "category" if "category" in df.columns else None
            if cat_col:
                task_filter = st.multiselect(
                    "Category",
                    options=sorted(df[cat_col].dropna().unique().tolist()),
                    default=sorted(df[cat_col].dropna().unique().tolist()),
                )
            else:
                task_filter = []
        with col2:
            parsing_filter = st.selectbox(
                "Parsing error", options=["All", True, False], index=0
            )
        with col3:
            iter_filter = st.selectbox(
                "Iter. limit", options=["All", True, False], index=0
            )
        with col4:
            correct_filter = st.selectbox(
                "Correct", options=["All", True, False], index=0
            )
        # Apply filters
        filtered = df.copy()
        if cat_col and task_filter:
            filtered = filtered[filtered[cat_col].isin(task_filter)]
        if parsing_filter != "All":
            filtered = filtered[filtered["parsing_error"] == parsing_filter]
        if iter_filter != "All":
            filtered = filtered[filtered["iteration_limit_exceeded"] == iter_filter]
        if correct_filter != "All" and "is_correct" in filtered.columns:
            filtered = filtered[filtered["is_correct"] == correct_filter]
        # Main table columns (support both HLE and GAIA)
        id_col_name = "task_id" if "task_id" in filtered.columns else "id"
        base_cols = [
            id_col_name,
            "agent_name",
            "question",
            "prediction",
            "true_answer",
            "is_correct",
            "parsing_error",
            "iteration_limit_exceeded",
            "category",
            "start_time",
            "end_time",
        ]
        show_cols = [c for c in base_cols if c in filtered.columns]
        # Summary stats
        total = len(filtered)
        correct_count = (filtered["is_correct"] == True).sum() if "is_correct" in filtered.columns else None
        m1, m2, m3 = st.columns(3)
        with m1:
            if correct_count is not None:
                st.markdown(
                    f'<div class="metric-card"><div class="label">Correct / Total</div><div class="value">{int(correct_count)} / {total}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="metric-card"><div class="label">Total rows</div><div class="value">{total}</div></div>',
                    unsafe_allow_html=True,
                )
        with m2:
            if correct_count is not None and total:
                pct = correct_count / total * 100
                st.markdown(
                    f'<div class="metric-card"><div class="label">Accuracy</div><div class="value">{pct:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="metric-card"><div class="label">Filtered</div><div class="value">{total}</div></div>',
                    unsafe_allow_html=True,
                )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="label">From total</div><div class="value">{len(df)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<p class="section-title">📋 Data table</p>', unsafe_allow_html=True)
        st.dataframe(
            filtered[show_cols],
            use_container_width=True,
            height=min(400, 35 * len(filtered) + 38),
        )
        # Details
        if not filtered.empty:
            st.markdown('<p class="section-title">🔍 Task details</p>', unsafe_allow_html=True)
            id_col = "task_id" if "task_id" in filtered.columns else "id"
            id_list = filtered[id_col].astype(str).tolist()
            selected = st.selectbox("Select task ID to view details", id_list, key="detail_select")
            detail = filtered[filtered[id_col].astype(str) == selected].iloc[0]
            # Correct/incorrect badge
            is_correct = detail.get("is_correct") if "is_correct" in detail.index else None
            if is_correct is not None:
                badge = "correct" if is_correct else "incorrect"
                label = "✓ Correct" if is_correct else "✗ Incorrect"
                st.markdown(
                    f'<span class="badge badge-{badge}">{label}</span>',
                    unsafe_allow_html=True,
                )
            def _escape(s: str) -> str:
                s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                return s.replace("\n", "<br>")

            q = _escape(detail["question"])
            st.markdown(
                f'<div class="detail-block"><h3>Question</h3><div class="content">{q}</div></div>',
                unsafe_allow_html=True,
            )
            pred = _escape(detail["prediction"])
            st.markdown(
                f'<div class="detail-block"><h3>Predicted Answer</h3><div class="content">{pred}</div></div>',
                unsafe_allow_html=True,
            )
            true_ans = _escape(detail["true_answer"])
            st.markdown(
                f'<div class="detail-block"><h3>True Answer</h3><div class="content">{true_ans}</div></div>',
                unsafe_allow_html=True,
            )
            # Meta info as styled grid
            def _esc(s: str) -> str:
                return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            meta_items = [
                ("Category", detail.get("category", "—")),
                ("Agent", detail.get("agent_name", "—")),
                ("Parsing Error", "Yes" if detail.get("parsing_error") else "No"),
                ("Iter. Limit", "Yes" if detail.get("iteration_limit_exceeded") else "No"),
                ("Start", str(detail.get("start_time", "—"))),
                ("End", str(detail.get("end_time", "—"))),
            ]
            meta_html = '<div class="meta-grid">'
            for k, v in meta_items:
                meta_html += f'<div class="meta-item"><div class="key">{_esc(k)}</div><div class="val">{_esc(str(v))}</div></div>'
            meta_html += "</div>"
            st.markdown(meta_html, unsafe_allow_html=True)
            st.markdown('<p class="section-title">📎 Additional data</p>', unsafe_allow_html=True)
            if detail.get("judgment_result"):
                with st.expander("📌 Judgment Result (GAIA)"):
                    st.json(detail["judgment_result"])
            if detail.get("agent_error"):
                with st.expander("⚠️ Error details (agent_error)"):
                    st.error(detail["agent_error"])
            with st.expander("📝 Augmented question"):
                st.write(detail.get("augmented_question", ""))
            with st.expander("🧠 Reasoning steps (intermediate_steps)"):
                steps = detail.get("intermediate_steps", [])
                if not steps:
                    st.info("No reasoning steps recorded.")
                else:
                    for i, step in enumerate(steps):
                        with st.expander(
                            f"Step {i+1} — {step.get('step_type', 'unknown')}"
                        ):
                            st.json(step)
            if detail.get("search_agent_actions"):
                with st.expander("🔗 search_agent_actions records"):
                    for i, act in enumerate(detail["search_agent_actions"]):
                        with st.expander(f"Action {i+1}"):
                            st.json(act)
        else:
            st.info("No data after filtering.")
else:
    st.info("📁 Please provide a valid JSONL file path.")
