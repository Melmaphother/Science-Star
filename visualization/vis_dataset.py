# -*- coding: utf-8 -*-
"""
HLE Dataset Visualization — browse dataset items with LaTeX/Markdown rendering.
"""

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure styles can be imported when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
from styles import apply_shared_style, hero_html


def _resolve_path(path: str) -> Path:
    """Resolve path relative to project root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    # Try cwd first, then project root
    if p.exists():
        return p.resolve()
    root_path = _PROJECT_ROOT / path
    return root_path if root_path.exists() else p


# ------------- LaTeX/Markdown Preprocessing -------------


def tex_to_markdown(text: str) -> str:
    """Convert common LaTeX document commands to Markdown."""
    if not isinstance(text, str):
        return str(text)
    text = (
        text.replace("\\\\section", "\\section")
        .replace("\\\\subsection", "\\subsection")
        .replace("\\\\textbf", "\\textbf")
        .replace("\\\\emph", "\\emph")
    )
    text = re.sub(r"\\section\*?\{([^}]*)\}", r"### \1\n", text)
    text = re.sub(r"\\subsection\*?\{([^}]*)\}", r"#### \1\n", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\begin\{itemize\}\s*", "", text)
    text = re.sub(r"\\end\{itemize\}\s*", "", text)
    text = re.sub(r"\\item\s*", "- ", text)
    return text


def convert_latex_delimiters(text: str) -> str:
    """Convert \\( … \\)/\\[ … \\] to $…$/$$…$$ for Streamlit math rendering."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
    text = text.replace("\\\\[", "\\[").replace("\\\\]", "\\]")
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.S)
    return text


def auto_wrap_equation_lines(text: str) -> str:
    """Auto-wrap suspected equation lines with $$...$$."""
    if not isinstance(text, str):
        return str(text)
    lines = text.split("\n")
    out = []
    for s in lines:
        s_strip = s.strip()
        if ("$" not in s_strip) and (r"\(" not in s_strip) and (r"\[" not in s_strip):
            has_cmd = bool(re.search(r"\\[a-zA-Z]+", s_strip))
            has_math = bool(re.search(r"(=|\\frac|\\sum|\\int|\^|_)", s_strip))
            if has_cmd and has_math:
                out.append("$$\n" + s_strip + "\n$$")
                continue
        out.append(s)
    return "\n".join(out)


# ------------- Data Loading -------------


def load_jsonl(filename: str) -> list:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _esc(s: str) -> str:
    """Escape HTML for safe display."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------- Main Application -------------


def main() -> None:
    """Run the dataset visualization app."""
    st.set_page_config(
        page_title="HLE Dataset Visualization",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_shared_style(st)
    st.markdown(
        hero_html(
            "📚 HLE Dataset Visualization",
            "Browse dataset items with LaTeX and Markdown rendering",
        ),
        unsafe_allow_html=True,
    )

    # Data source
    st.markdown(
        '<p class="filter-label" style="margin-bottom: 0.5rem;">📁 Data source</p>',
        unsafe_allow_html=True,
    )
    common_paths = [
        str(_PROJECT_ROOT / "data/HLE/subset/hle_subset_50.jsonl"),
        str(_PROJECT_ROOT / "data/HLE/subset/hle_subset_200.jsonl"),
        str(_PROJECT_ROOT / "data/HLE/subset/hle_subset_500.jsonl"),
        str(_PROJECT_ROOT / "data/GAIA/subset/gaia_subset_20.jsonl"),
        str(_PROJECT_ROOT / "data/GAIA/gaia.jsonl"),
    ]
    existing_paths = [p for p in common_paths if Path(p).exists()]

    if existing_paths:
        selected_path = st.selectbox("Select dataset", existing_paths, index=0)
        file_path = st.text_input("Or enter custom path", value=selected_path)
    else:
        file_path = st.text_input(
            "Dataset path",
            value=str(_PROJECT_ROOT / "data/HLE/subset/hle_subset_50.jsonl"),
        )

    resolved = _resolve_path(file_path)
    if not resolved.exists():
        st.error(f"⚠️ File does not exist: {file_path}")
        st.caption("Tip: Use absolute path or path relative to project root.")
        return

    data = load_jsonl(str(resolved))
    df = pd.DataFrame(data)

    # Summary metrics
    total = len(df)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Total entries</div><div class="value">{total}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        cats = df["category"].nunique() if "category" in df.columns and len(df) else 0
        st.markdown(
            f'<div class="metric-card"><div class="label">Categories</div><div class="value">{cats}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Columns</div><div class="value">{len(df.columns)}</div></div>',
            unsafe_allow_html=True,
        )

    # Raw table
    st.markdown(
        '<p class="section-title">📋 Raw data table</p>', unsafe_allow_html=True
    )
    with st.expander("View raw data", expanded=False):
        st.dataframe(
            df,
            use_container_width=True,
            height=min(300, 35 * min(len(df), 20) + 38),
        )

    # Item-by-item browse
    st.markdown(
        '<p class="section-title">🔍 Item-by-item browse</p>',
        unsafe_allow_html=True,
    )

    if total == 0:
        st.info("Data is empty.")
        return

    if "browse_idx" not in st.session_state:
        st.session_state.browse_idx = 0

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col2:
        jump_idx = st.number_input(
            "Jump to index (0-based)",
            min_value=0,
            max_value=total - 1,
            value=st.session_state.browse_idx,
            step=1,
        )
    if jump_idx != st.session_state.browse_idx:
        st.session_state.browse_idx = int(jump_idx)

    # Position + prev/next
    pos_col1, pos_col2, pos_col3 = st.columns(3)
    with pos_col1:
        b_prev = st.button(
            "⬅️ Previous",
            key="prev_top",
            use_container_width=True,
            disabled=(st.session_state.browse_idx <= 0),
        )
    with pos_col2:
        st.markdown(
            f'<div class="metric-card" style="margin-bottom: 0;"><div class="label">Position</div><div class="value">{st.session_state.browse_idx + 1} / {total}</div></div>',
            unsafe_allow_html=True,
        )
    with pos_col3:
        b_next = st.button(
            "Next ➡️",
            key="next_top",
            use_container_width=True,
            disabled=(st.session_state.browse_idx >= total - 1),
        )

    if b_prev:
        st.session_state.browse_idx = max(0, st.session_state.browse_idx - 1)
        st.rerun()
    if b_next:
        st.session_state.browse_idx = min(total - 1, st.session_state.browse_idx + 1)
        st.rerun()

    row = df.iloc[st.session_state.browse_idx]

    # Meta info grid
    meta_items = []
    id_field = "task_id" if "task_id" in row else ("id" if "id" in row else None)
    if id_field:
        meta_items.append(("ID", str(row[id_field])[-12:]))
    if "category" in row:
        meta_items.append(("Category", str(row["category"])))
    if "answer_type" in row:
        meta_items.append(("Answer Type", str(row["answer_type"])))

    if meta_items:
        meta_html = '<div class="meta-grid">'
        for k, v in meta_items:
            meta_html += f'<div class="meta-item"><div class="key">{_esc(k)}</div><div class="val">{_esc(str(v))}</div></div>'
        meta_html += "</div>"
        st.markdown(meta_html, unsafe_allow_html=True)

    # Content: question + answer | JSON
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<p class="section-title">📝 Question</p>', unsafe_allow_html=True)
        qkey = (
            "Question"
            if "Question" in row
            else ("question" if "question" in row else None)
        )
        if qkey:
            raw_q = str(row[qkey])
            processed = auto_wrap_equation_lines(raw_q)
            processed = convert_latex_delimiters(processed)
            processed = tex_to_markdown(processed)
            st.markdown(processed, unsafe_allow_html=False)
        else:
            st.info("No Question/question field found.")

        # Final answer
        ak = None
        for field in ["Final answer", "final_answer", "answer"]:
            if field in row:
                ak = field
                break
        if ak:
            st.markdown(
                '<p class="section-title">✅ Final answer</p>', unsafe_allow_html=True
            )
            st.code(str(row[ak]))

    with right:
        st.markdown(
            '<p class="section-title">📄 Complete sample (JSON)</p>',
            unsafe_allow_html=True,
        )
        display_dict = row.to_dict()
        if "rationale" in display_dict and len(str(display_dict["rationale"])) > 500:
            rationale = display_dict.pop("rationale")
            st.json(display_dict)
            with st.expander("📎 View rationale"):
                st.text(str(rationale))
        else:
            st.json(display_dict)


if __name__ == "__main__":
    main()
