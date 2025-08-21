# 🏗️ Science-Star Project Architecture Guide

Welcome to the **Science-Star** universe! 🚀 This isn't just another project structure - it's a carefully crafted ecosystem designed to empower scientific AI agents. Let's take a journey through each component and discover what makes this platform tick! ⚡

## 🧠 Core Intelligence Hub - `science_star/`

The beating heart of our scientific agent ecosystem! This is where all the magic happens. ✨

### 🎯 **run_hle.py** - The Orchestrator Supreme
Your mission control center! 🎮 This powerful entry point coordinates the entire HLE evaluation workflow, managing agent lifecycles, experiment configurations, and result collection. Think of it as the conductor of a scientific symphony orchestra! 🎼

### 🛠️ **tools/** - The Agent's Swiss Army Knife
A treasure trove of specialized tools that give your agents superpowers! 💪
- 🔍 **searcher.py** & **reformulator.py** - Information hunting and query refinement
- 🕷️ **web_crawler.py** family - Web exploration capabilities  
- 🎯 **scorer.py** & **gaia_scorer.py** - Performance evaluation engines
- 👁️ **visual_inspector_tool.py** & **audio_inspector_tool.py** - Multimodal analysis
- 🤖 **automodel.py** - Dynamic model management
- And many more specialized instruments! 🧰

### 📊 **data_utils/** - Data Wizardry Department
Lightweight but mighty data preprocessing and loading utilities! 🎭
- **hle_loader.py** - Your gateway to the HLE dataset universe

### 🧠 **agent_kb/** - Knowledge Base Command Center
The brain trust of your agents! 🧙‍♂️
- **agent_kb_retrieval.py** - Smart knowledge retrieval
- **agent_kb_service.py** - Knowledge service orchestration  
- **prompts.yaml** - The secret sauce of agent communication

### 🔄 **reflectors/** - The Wisdom Amplifiers
Where agents learn to think about their thinking! 🤔💭
- **search_reflector.py** - Self-improving search strategies
- **prompts/** - Reflection prompt templates for deeper reasoning

### 🎯 **hle_eval/** - Evaluation Excellence Center
The quality assurance department! 📏✅
- **run_model_predictions.py** - Prediction pipeline orchestrator
- **run_judge_results.py** - Result validation and scoring

### 🤖 **rag/** - Retrieval-Augmented Generation Powerhouse
A comprehensive RAG ecosystem that would make any AI researcher jealous! 🏆

#### 🔌 **embeddings/** - The Vector Virtuosos
- Support for **OpenAI**, **Jina**, **Mistral**, **SentenceTransformers**, and more!
- **vlm_embedding.py** - Vision-language model embeddings for multimodal magic ✨

#### 📚 **loaders/** - Data Ingestion Masters
- **firecrawl_reader.py** & **jina_url_reader.py** - Web content extraction
- **chunkr_reader.py** & **unstructured_io.py** - Document processing specialists
- **apify_reader.py** - Advanced web scraping capabilities

#### 🗄️ **storages/** - The Data Fortress
Multi-tier storage architecture:
- **vectordb_storages/** - Milvus, Qdrant vector databases 🎯
- **graph_storages/** - Neo4j, NebulaGraph for complex relationships 🕸️
- **key_value_storages/** - Redis, JSON, in-memory solutions ⚡
- **object_storages/** - AWS S3, Azure Blob, Google Cloud integration ☁️

#### 🔍 **retrievers/** - Information Hunters
- **auto_retriever.py** & **graph_auto_retriever.py** - Smart retrieval
- **bm25_retriever.py** - Classical text retrieval
- **cohere_rerank_retriever.py** - Advanced reranking capabilities

#### 💬 **messages/** - Communication Protocol Center
- **conversion/** - Format transformation wizards
- **sharegpt/** - ShareGPT format support with Hermes integration

## 🏭 Foundation Framework - `src/`

### 🔧 **smolagents/** - The Engine Room
Built on the powerful smolagents framework! 🚂
- **agents.py** & **agent_types.py** - Core agent abstractions
- **tools.py** & **default_tools.py** - Tool ecosystem
- **models.py** & **monitoring.py** - Model management & observability
- **prompts/** - Rich prompt template library
- **workflow.py** - Workflow orchestration capabilities

## ⚙️ Configuration Command Center - `configs/`

### 📋 **hle.yaml** - The Master Control File
Your experiment's DNA! 🧬 Configure everything from model selection to search strategies:
- 🎛️ Runtime parameters (concurrency, debugging)
- 🎯 Dataset selection (subset, category filtering)
- 🤖 Model configurations (main, search, retrieval models)
- 🔄 Advanced features (reflection, planning, memory)
- 🎮 Search strategies (Beam Search, Tree Search, BON)

## 📊 Data Universe - `data/`

### 🏆 **HLE/** - The High-Level Evaluation Dataset
A comprehensive scientific evaluation ecosystem! 🔬
- **hle.jsonl** - Main dataset with rich metadata
- **category/** - Domain-specific subsets (Biology, Chemistry, CS, Physics, etc.) 🧪⚗️🖥️⚛️
- **subset/** - Curated evaluation sets (50, 200, 500 samples)
- **split_*.py** - Dataset management utilities

## 📈 Visualization Studio - `visualization/`

Where data comes alive! 🎨📊

### 🎭 **vis_dataset.py** - Dataset Explorer Extraordinaire
Interactive dataset visualization with:
- 📖 LaTeX/Markdown rendering for scientific content
- 🔄 Multi-format support (old & new data schemas)
- 🎯 Smart field detection and compatibility
- 📊 Real-time data exploration

### 📊 **vis_output.py** - Results Visualization Engine
Transform your experimental results into insightful visualizations! 📈

## 🚀 Launch Pad - `scripts/`

### 🎬 **run_hle.sh** - One-Click Experiment Launcher
Your shortcut to scientific discovery! Just one command to rule them all! 👑

## 📚 Documentation Hub - `docs/`

Your knowledge companion! 📖
- **installation.md** - Setup your scientific laboratory 🧪
- **quickstart.md** - From zero to hero in minutes ⚡
- **project_structure.md** - This very guide you're reading! 😊

## 🎁 Extras & Assets

### 📦 **assets/** - Visual Resources
Screenshots, diagrams, and visual aids to help you navigate! 🖼️

### 📋 **requirements.txt** - Dependency Manifest
All the Python packages your agents need to thrive! 🐍

### ⚖️ **LICENSE** - Legal Framework
Open-source goodness with proper attribution! 📜

---

## 🎯 Quick Navigation Tips

🔥 **Hot Paths for Development:**
- Start with `science_star/run_hle.py` for main workflows
- Explore `science_star/tools/` for extending agent capabilities  
- Dive into `science_star/rag/` for advanced retrieval features
- Check `visualization/` for data exploration and result analysis
- Modify `configs/hle.yaml` for experiment customization

🚀 **Ready to Build Something Amazing?**
This architecture is designed for extensibility and experimentation. Each component is carefully crafted to work harmoniously while maintaining clear boundaries. Whether you're building new tools, integrating novel models, or creating custom evaluation pipelines, Science-Star provides the foundation you need!

Happy coding, and may your agents be ever intelligent! 🤖✨