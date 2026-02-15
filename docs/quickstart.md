# 🚀 Quick Start Guide

Welcome to Science-Star! Get up and running in minutes with this simple guide. ⚡

## 📋 Prerequisites

Make sure you have completed the [Installation](installation.md) first! 🔧

## 🎯 Part 1: Running Evaluations

### 🔥 One-Command Launch

Science-Star supports both **Humanity's Last Exam (HLE)** and **GAIA** benchmarks. Choose your dataset:

**HLE:**

```bash
sh scripts/run_hle.sh
```

**GAIA:**

```bash
sh scripts/run_gaia.sh
```

These scripts will:
- 🤖 Launch the evaluation with **gpt-4o-mini** (multi-agent: Manager Agent + Sub-Agent for Search)
- 📊 Process the dataset using LLM agents with search, crawl, and inspector tools
- 💾 Save results to `output/<run_name>/<timestamp>/answers.jsonl`
- ⏱️ Use configuration from `configs/hle.yaml` or `configs/gaia.yaml`

### 🛠️ Custom Configuration

Want to customize your run? Use dot notation to override config parameters:

```bash
# Multi-agent (CodeAgent + search agent) — recommended
PYTHONPATH=.:science_star python3 science_star/run_multi_agent.py \
  config=configs/hle.yaml \
  models.name=gpt-4o-mini \
  dataset.subset=small \
  agents.max_steps=12 \
  runtime.run_name=my-custom-run

# Single-agent (ToolCallingAgent only)
PYTHONPATH=.:science_star python3 science_star/run_single_agent.py \
  config=configs/gaia.yaml \
  models.name=gpt-4o-mini \
  runtime.run_name=my-single-agent-run
```

**Key Parameters (dot notation):**
- `config`: Config file path (`configs/hle.yaml` or `configs/gaia.yaml`)
- `models.name`: LLM model (gpt-4o-mini, gpt-4o, etc.)
- `dataset.subset`: Dataset size (`small` | `medium` | `large` | `null` for full)
- `dataset.category`: (HLE) Focus on domains: bio, chem, cs, math, physics, etc.
- `dataset.level`: (GAIA) Difficulty: level1, level2, level3
- `dataset.selected_tasks`: (GAIA) 1-based task IDs for quick tests, e.g. `[1,2,3]`
- `agents.max_steps`: Maximum reasoning steps per problem
- `runtime.run_name`: Custom name for your experiment
- `runtime.concurrency`: Parallel task execution (default: 1)

### 📈 Monitor Progress

Watch your agents work in real-time! The console will show:
- ✅ Completed tasks
- 🔄 Current reasoning steps
- 📊 Success rates
- ⏱️ Execution times

## 🎨 Part 2: Interactive Data Visualization

### 🔍 Explore Your Dataset

Launch the interactive dataset explorer (supports both HLE and GAIA):

```bash
streamlit run visualization/vis_dataset.py
```

Run from the project root so paths resolve correctly. This opens a web interface where you can:
- 📖 **Browse Questions**: Navigate through dataset entries with LaTeX/Markdown rendering
- 🏷️ **Filter by Category**: Explore Biology, Chemistry, Math, Physics, and more (HLE)
- 🔎 **Search & Jump**: Quick navigation to specific entries
- 📊 **View Metadata**: See question types, difficulty levels, and annotations
- 📁 **Multi-Dataset**: Switch between HLE and GAIA subsets

### 🎛️ Visualization Features

- **Smart Rendering**: Mathematical equations and scientific notation display perfectly
- **Multi-Format Support**: Works with both HLE and GAIA dataset schemas
- **Real-time Navigation**: Instant switching between questions
- **Responsive Design**: Works on desktop and mobile

### 📊 Analyze Results

```bash
streamlit run visualization/vis_output.py
```

Perfect for analyzing experiment results! Supports:
- HLE and GAIA answer formats
- Filtering by category, parsing error, iteration limit, correctness
- Task details with reasoning steps, agent errors, and judgment results

## 🆘 Need Help?

- 📖 Check [Project Structure](project_structure.md) for detailed architecture
- 🐛 Found a bug? Open an issue on GitHub

---

**Happy experimenting!** 🚀✨ Your scientific AI agents are ready to tackle complex problems!
