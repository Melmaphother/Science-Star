# -*- coding: utf-8 -*-
"""
Shared CSS styles for Streamlit visualization apps.
Science-lab aesthetic: dark theme, teal/amber accents, DM Sans + JetBrains Mono.
"""

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-dark: #0f1419;
    --bg-card: #1a2332;
    --accent-teal: #2dd4bf;
    --accent-amber: #fbbf24;
    --text-primary: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #334155;
    --success: #34d399;
    --error: #f87171;
}

.stApp { background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%); }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

h1, h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

.hero {
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.08) 0%, rgba(251, 191, 36, 0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
}

.hero h1 { margin: 0; font-size: 1.75rem !important; }
.hero p { margin: 0.25rem 0 0 0; color: var(--text-muted); font-size: 0.9rem; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 1rem;
}

.metric-card .label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: var(--accent-teal); font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

.section-title {
    font-size: 1rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.detail-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.detail-block h3 { font-size: 0.85rem !important; color: var(--text-muted) !important; margin-bottom: 0.5rem !important; }
.detail-block .content { color: var(--text-primary); font-size: 0.95rem; line-height: 1.6; }

div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
.stSelectbox > div, .stMultiSelect > div { border-radius: 8px; }

.filter-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
}

.filter-panel .filter-label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }

.badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 500; }
.badge-correct { background: rgba(52, 211, 153, 0.2); color: var(--success); }
.badge-incorrect { background: rgba(248, 113, 113, 0.2); color: var(--error); }

.input-hint { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem; }

[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.5rem; }
[data-testid="stExpander"] summary { padding: 0.75rem 1rem; }

/* Alert container: move to DOM order index 7 (after hero, filters, etc.) */
div[data-testid="stAlertContainer"] { order: 7; }

.meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
.meta-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; }
.meta-item .key { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.25rem; }
.meta-item .val { color: var(--text-primary); font-size: 0.9rem; }
</style>
"""


def apply_shared_style(st_module) -> None:
    """
    Inject shared CSS into the Streamlit app.

    Args:
        st_module: The streamlit module (e.g., `import streamlit as st`).
    """
    st_module.markdown(SHARED_CSS, unsafe_allow_html=True)


def hero_html(title: str, subtitle: str = "") -> str:
    """Build hero section HTML."""
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    return f'<div class="hero"><h1>{title}</h1>{sub}</div>'
