#!/usr/bin/env python

"""
Authorized imports for code-execution tools (e.g. PythonInterpreterTool).

Smolagents uses code-as-communication; the agent writes Python that runs in a sandbox.
This list controls which top-level modules are allowed when the agent's code is executed.
"""

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yfinance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
]
