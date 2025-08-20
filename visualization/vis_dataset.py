# -*- coding: utf-8 -*-

import json
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------- LaTeX/Markdown Preprocessing -------------
# Note: To avoid unicodeescape issues, avoid \x... in regular strings/triple quotes,
# This file doesn't use docstrings with backslashes, explanations are written in comments.

def tex_to_markdown(text: str) -> str:
    # Convert common LaTeX document commands to Markdown: \section/\subsection/\textbf/\emph/itemize
    if not isinstance(text, str):
        return str(text)
    # Restore possible double backslashes
    text = (text
            .replace('\\\\section', '\\section')
            .replace('\\\\subsection', '\\subsection')
            .replace('\\\\textbf', '\\textbf')
            .replace('\\\\emph', '\\emph'))
    # \section*{Title} / \section{Title} -> ### Title
    text = re.sub(r'\\section\*?\{([^}]*)\}', r'### \1\n', text)
    # \subsection*{Title} / \subsection{Title} -> #### Title
    text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'#### \1\n', text)
    # \textbf{X} -> **X**
    text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)
    # \emph{X} -> *X*
    text = re.sub(r'\\emph\{([^}]*)\}', r'*\1*', text)
    # Simplified handling of itemize environment
    text = re.sub(r'\\begin\{itemize\}\s*', '', text)
    text = re.sub(r'\\end\{itemize\}\s*', '', text)
    text = re.sub(r'\\item\s*', '- ', text)
    return text

def convert_latex_delimiters(text: str) -> str:
    # Convert \( … \)/\[ … \] to $…$/$$…$$ for Streamlit math rendering
    if not isinstance(text, str):
        return str(text)
    # Restore double backslashes to single backslashes
    text = text.replace('\\\\(', '\\(').replace('\\\\)', '\\)')
    text = text.replace('\\\\[', '\\[').replace('\\\\]', '\\]')
    # \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.S)
    # \( ... \) -> $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$',   text, flags=re.S)
    return text

def auto_wrap_equation_lines(text: str) -> str:
    # Automatically add $$...$$ for "suspected equation" lines:
    # Condition: line contains no existing $ or \( or \[, and contains both LaTeX commands \xxx and math validation symbols
    if not isinstance(text, str):
        return str(text)
    lines = text.split('\n')
    out = []
    for s in lines:
        s_strip = s.strip()
        if (('$' not in s_strip) and (r'\(' not in s_strip) and (r'\[' not in s_strip)):
            has_cmd = bool(re.search(r'\\[a-zA-Z]+', s_strip))
            has_math = bool(re.search(r'(=|\\frac|\\sum|\\int|\^|_)', s_strip))
            if has_cmd and has_math:
                out.append('$$\n' + s_strip + '\n$$')
                continue
        out.append(s)
    return '\n'.join(out)

# ------------- Data Loading -------------

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ------------- Main Application -------------

def main():
    st.set_page_config(page_title="HLE Dataset Visualization", layout="wide")
    st.title("HLE Dataset Visualization")

    # Dataset path with dropdown for common files
    common_paths = [
        "data/HLE/subset/hle_subset_50.jsonl", 
        "data/HLE/subset/hle_subset_200.jsonl",
        "data/HLE/subset/hle_subset_500.jsonl"
    ]
    
    # Check which files exist
    existing_paths = [path for path in common_paths if os.path.exists(path)]
    
    if existing_paths:
        default_path = existing_paths[0]
        selected_path = st.selectbox("Select Dataset", existing_paths, index=0)
        file_path = st.text_input("Or enter custom path", value=selected_path)
    else:
        file_path = st.text_input("Dataset Path", value="data/HLE/subset/hle_subset_50.jsonl")
    if not os.path.exists(file_path):
        st.error(f"File does not exist: {file_path}")
        return

    data = load_jsonl(file_path)
    st.write(f"Loaded {len(data)} data entries")
    df = pd.DataFrame(data)

    # Raw table
    with st.expander("View Raw Data Table"):
        st.dataframe(df)

    # —— Browse Item by Item (Index Jump + Bottom Pagination) ——
    st.markdown("---")
    st.subheader("Detailed Item-by-Item Browse")

    total = len(df)
    if total == 0:
        st.info("Data is empty.")
        return

    if "browse_idx" not in st.session_state:
        st.session_state.browse_idx = 0

    jump_idx = st.number_input(
        "Jump to Index (0-based)",
        min_value=0,
        max_value=total - 1,
        value=st.session_state.browse_idx,
        step=1,
    )
    if jump_idx != st.session_state.browse_idx:
        st.session_state.browse_idx = int(jump_idx)

    st.caption(f"Position: {st.session_state.browse_idx + 1} / {total}")
    
    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    b_prev = nav_col1.button("⬅️ Previous", key="prev_top", use_container_width=True,
                             disabled=(st.session_state.browse_idx <= 0))
    b_next = nav_col3.button("Next ➡️", key="next_top", use_container_width=True,
                             disabled=(st.session_state.browse_idx >= total - 1))
    if b_prev:
        st.session_state.browse_idx = max(0, st.session_state.browse_idx - 1)
        st.rerun()
    if b_next:
        st.session_state.browse_idx = min(total - 1, st.session_state.browse_idx + 1)
        st.rerun()
    
    row = df.iloc[st.session_state.browse_idx]
    
    # Display basic info about the current item
    info_cols = st.columns(3)
    
    # Show ID field
    id_field = "task_id" if "task_id" in row else ("id" if "id" in row else None)
    if id_field:
        info_cols[0].metric("ID", str(row[id_field])[-8:])
    
    # Show category
    if "category" in row:
        info_cols[1].metric("Category", str(row["category"]))
    
    # Show answer type if available
    if "answer_type" in row:
        info_cols[2].metric("Answer Type", str(row["answer_type"]))

    # Left and right columns: left renders question, right shows JSON
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Question (Markdown + LaTeX Rendering)**")
        # Compatible with field names Question/question
        qkey = "Question" if "Question" in row else ("question" if "question" in row else None)
        if qkey:
            raw_q = str(row[qkey])
            # Preprocessing order: auto-wrap equation lines -> delimiter conversion -> document commands to Markdown
            processed = auto_wrap_equation_lines(raw_q)
            processed = convert_latex_delimiters(processed)
            processed = tex_to_markdown(processed)
            st.markdown(processed, unsafe_allow_html=False)
        else:
            st.info("No Question/question field found.")

        # Final answer - support multiple field names
        ak = None
        for field in ["Final answer", "final_answer", "answer"]:
            if field in row:
                ak = field
                break
        
        if ak:
            st.markdown("**Final Answer**")
            st.code(str(row[ak]))

    with right:
        st.markdown("**Complete Sample (JSON)**")
        # Create a more readable JSON display by excluding very long fields
        display_dict = row.to_dict()
        
        # Show rationale separately if it exists and is long
        if "rationale" in display_dict and len(str(display_dict["rationale"])) > 500:
            rationale = display_dict.pop("rationale")
            st.json(display_dict)
            st.markdown("**Rationale**")
            with st.expander("View rationale", expanded=False):
                st.text(str(rationale))
        else:
            st.json(display_dict)



if __name__ == "__main__":
    main()
