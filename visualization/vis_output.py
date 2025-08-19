import streamlit as st
import pandas as pd
import json
from pathlib import Path

# streamlit run analyze_result.py
st.title("HLE Results Visualization (Enhanced)")

jsonl_file = st.text_input(
    "JSONL file path", "output/test/deepseek-ai/DeepSeek-R1-0528-gaia.jsonl"
)
# jsonl_file = st.text_input('JSONL file path', 'output/validation/gpt-4o-mini-gaia.jsonl')

if jsonl_file and Path(jsonl_file).exists():
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if not data:
        st.warning("File is empty or invalid format.")
    else:
        df = pd.DataFrame(data)
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            task_filter = st.multiselect(
                "Category filter",
                options=sorted(df["category"].dropna().unique().tolist()),
                default=sorted(df["category"].dropna().unique().tolist()),
            )
        with col2:
            parsing_filter = st.selectbox(
                "Parsing error", options=["All", True, False], index=0
            )
        with col3:
            iter_filter = st.selectbox(
                "Iteration limit exceeded", options=["All", True, False], index=0
            )
        # Apply filters, ensure `filtered` always a DataFrame
        filtered = df.copy()
        filtered = filtered[filtered["category"].isin(task_filter)]
        if parsing_filter != "All":
            filtered = filtered[filtered["parsing_error"] == parsing_filter]
        if iter_filter != "All":
            filtered = filtered[filtered["iteration_limit_exceeded"] == iter_filter]
        # Main table
        show_cols = [
            c
            for c in [
                "task_id",
                "question",
                "prediction",
                "true_answer",
                "parsing_error",
                "iteration_limit_exceeded",
                "category",
                "start_time",
                "end_time",
            ]
            if c in filtered.columns
        ]
        st.dataframe(filtered[show_cols], use_container_width=True)
        # Details
        if not filtered.empty:
            task_id_list = filtered["task_id"].tolist()
            selected = st.selectbox("Select task ID to view details", task_id_list)
            detail = filtered[filtered["task_id"] == selected].iloc[0]
            st.markdown(f"### Question\n{detail['question']}")
            st.markdown(f"### Predicted Answer\n{detail['prediction']}")
            st.markdown(f"### True Answer\n{detail['true_answer']}")
            st.markdown(f"### Category: {detail.get('category', '')}")
            st.markdown(f"### Start Time: {detail.get('start_time', '')}")
            st.markdown(f"### End Time: {detail.get('end_time', '')}")
            st.markdown(f"### Parsing Error: {detail.get('parsing_error', '')}")
            st.markdown(
                f"### Iteration Limit Exceeded: {detail.get('iteration_limit_exceeded', '')}"
            )
            if detail.get("agent_error"):
                with st.expander("Error details (agent_error)"):
                    st.error(detail["agent_error"])
            with st.expander("Augmented question (augmented_question)"):
                st.write(detail.get("augmented_question", ""))
            with st.expander("Reasoning steps (intermediate_steps)"):
                steps = detail.get("intermediate_steps", [])
                if not steps:
                    st.info("No reasoning steps recorded.")
                else:
                    for i, step in enumerate(steps):
                        with st.expander(
                            f"Step {i+1} - {step.get('step_type', 'unknown')}"
                        ):
                            st.json(step)
            if detail.get("search_agent_actions"):
                with st.expander("search_agent_actions records"):
                    for i, act in enumerate(detail["search_agent_actions"]):
                        with st.expander(f"Action {i+1}"):
                            st.json(act)
        else:
            st.info("No data after filtering.")
else:
    st.info("Please provide a valid JSONL file path.")
