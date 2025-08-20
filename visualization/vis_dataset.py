# -*- coding: utf-8 -*-

import json
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------- LaTeX/Markdown 预处理 -------------
# 注意：为避免 unicodeescape 问题，避免在普通字符串/三引号中出现 \x...，
# 本文件不使用含反斜杠的 docstring，说明写在注释里。

def tex_to_markdown(text: str) -> str:
    # 将常见 LaTeX 文档命令转换为 Markdown：\section/\subsection/\textbf/\emph/itemize
    if not isinstance(text, str):
        return str(text)
    # 还原可能的双反斜杠
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
    # 精简处理 itemize 环境
    text = re.sub(r'\\begin\{itemize\}\s*', '', text)
    text = re.sub(r'\\end\{itemize\}\s*', '', text)
    text = re.sub(r'\\item\s*', '- ', text)
    return text

def convert_latex_delimiters(text: str) -> str:
    # 将 \( … \)/\[ … \] 转为 $…$/$$…$$ 以便 Streamlit 渲染数学
    if not isinstance(text, str):
        return str(text)
    # 还原双反斜杠到单反斜杠
    text = text.replace('\\\\(', '\\(').replace('\\\\)', '\\)')
    text = text.replace('\\\\[', '\\[').replace('\\\\]', '\\]')
    # \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.S)
    # \( ... \) -> $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$',   text, flags=re.S)
    return text

def auto_wrap_equation_lines(text: str) -> str:
    # 为“疑似公式”的行自动添加 $$...$$：
    # 条件：该行不含现有 $ 或 \( 或 \[，且同时包含 LaTeX 命令 \xxx 与数学校验符号
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

# ------------- 数据加载 -------------

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ------------- 主应用 -------------

def main():
    st.set_page_config(page_title="HLE 数据集可视化", layout="wide")
    st.title("HLE 数据集可视化")

    # 数据集路径
    default_path = "data/HLE/hle_subset.jsonl"
    file_path = st.text_input("数据集路径", value=default_path)
    if not os.path.exists(file_path):
        st.error(f"文件不存在: {file_path}")
        return

    data = load_jsonl(file_path)
    st.write(f"共加载 {len(data)} 条数据")
    df = pd.DataFrame(data)

    # 原始表格
    with st.expander("查看原始数据表格"):
        st.dataframe(df)

    # —— 逐条浏览（索引跳转 + 底部翻页）——
    st.markdown("---")
    st.subheader("逐条详细浏览")

    total = len(df)
    if total == 0:
        st.info("数据为空。")
        return

    if "browse_idx" not in st.session_state:
        st.session_state.browse_idx = 0

    jump_idx = st.number_input(
        "跳转到索引（0-based）",
        min_value=0,
        max_value=total - 1,
        value=st.session_state.browse_idx,
        step=1,
    )
    if jump_idx != st.session_state.browse_idx:
        st.session_state.browse_idx = int(jump_idx)

    st.caption(f"位置：{st.session_state.browse_idx + 1} / {total}")
    row = df.iloc[st.session_state.browse_idx]

    # 左右列：左侧渲染题面，右侧 JSON
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**题面（Markdown + LaTeX 渲染）**")
        # 兼容字段名 Question/question
        qkey = "Question" if "Question" in row else ("question" if "question" in row else None)
        if qkey:
            raw_q = str(row[qkey])
            # 预处理顺序：自动包公式行 -> 分隔符转换 -> 文档命令转 Markdown
            processed = auto_wrap_equation_lines(raw_q)
            processed = convert_latex_delimiters(processed)
            processed = tex_to_markdown(processed)
            st.markdown(processed, unsafe_allow_html=False)
        else:
            st.info("没有 Question/question 字段。")

        # Final answer
        ak = "Final answer" if "Final answer" in row else ("final_answer" if "final_answer" in row else None)
        if ak:
            st.markdown("**Final answer**")
            st.code(str(row[ak]))

    with right:
        st.markdown("**完整样本（JSON）**")
        st.json(row.to_dict())

    # 底部翻页按钮
    bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
    b_prev = bcol1.button("⬅️ 上一条", key="prev_bottom", use_container_width=True,
                          disabled=(st.session_state.browse_idx <= 0))
    b_next = bcol3.button("下一条 ➡️", key="next_bottom", use_container_width=True,
                          disabled=(st.session_state.browse_idx >= total - 1))
    if b_prev:
        st.session_state.browse_idx = max(0, st.session_state.browse_idx - 1)
        st.rerun()
    if b_next:
        st.session_state.browse_idx = min(total - 1, st.session_state.browse_idx + 1)
        st.rerun()

if __name__ == "__main__":
    main()
