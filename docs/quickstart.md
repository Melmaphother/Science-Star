# 🚀 Quick Start Guide

Welcome to Open-Agent! Get up and running in minutes with this simple guide. ⚡

## 📋 Prerequisites

Make sure you have completed the [Environment Setup](installation.md) first! 🔧

## 🎯 Part 1: Running HLE Evaluation

### 🔥 One-Command Launch
The fastest way to see Open-Agent in action:

```bash
sh scripts/run_hle.sh
```

This will:
- 🤖 Launch the HLE evaluation with OpenAI-o4-mini
- 📊 Process the HLE dataset using ReAct agents
- 💾 Save results to the `output/` directory
- ⏱️ Run with default configuration from `configs/hle.yaml`

### 🛠️ Custom Configuration
Want to customize your run? Modify the parameters:

```bash
python3 open_agent/run_hle.py \
  config=configs/hle.yaml \
  model_id=OpenAI-o4-mini \
  subset=small \
  max_steps=10 \
  run_name=my-custom-run
```

**Key Parameters:**
- `model_id`: Choose your model (OpenAI-o4-mini, claude-4-sonnet, etc.)
- `subset`: Dataset size (small/medium/large or null for full dataset)
- `category`: Focus on specific domains (bio, chem, cs, math, physics, etc.)
- `max_steps`: Maximum reasoning steps per problem
- `run_name`: Custom name for your experiment

### 📈 Monitor Progress
Watch your agents work in real-time! The console will show:
- ✅ Completed tasks
- 🔄 Current reasoning steps  
- 📊 Success rates
- ⏱️ Execution times

## 🎨 Part 2: Interactive Data Visualization

### 🔍 Explore Your Dataset
Launch the interactive dataset explorer:

```bash
streamlit run visualization/vis_dataset.py
```

This opens a beautiful web interface where you can:
- 📖 **Browse Questions**: Navigate through dataset entries with LaTeX/Markdown rendering
- 🏷️ **Filter by Category**: Explore Biology, Chemistry, Math, Physics, and more
- 🔎 **Search & Jump**: Quick navigation to specific entries
- 📊 **View Metadata**: See question types, difficulty levels, and annotations

### 🎛️ Visualization Features
- **Smart Rendering**: Mathematical equations and scientific notation display perfectly
- **Multi-Format Support**: Works with both old and new dataset schemas
- **Real-time Navigation**: Instant switching between questions
- **Responsive Design**: Works on desktop and mobile

### 📊 Analyze Results
```bash
streamlit run visualization/vis_output.py
```

Perfect for analyzing your experiment results and comparing different model performances!

## 🎉 What's Next?

### 🔬 **For Researchers**
- Modify `configs/hle.yaml` to test different strategies
- Add custom tools in `open_agent/tools/`
- Experiment with reflection and planning parameters

### 🛠️ **For Developers** 
- Explore the RAG pipeline in `open_agent/rag/`
- Build custom retrievers and embeddings
- Integrate new model backends

### 📊 **For Data Scientists**
- Use visualization tools to understand dataset patterns
- Analyze agent reasoning traces
- Compare performance across different categories

## 🆘 Need Help?

- 📖 Check [Project Structure](project_structure.md) for detailed architecture
- 🐛 Found a bug? Open an issue on GitHub
- 💬 Join our [WeChat community](../assets/wechat.jpeg) for discussions

---

**Happy experimenting!** 🚀✨ Your scientific AI agents are ready to tackle complex problems!