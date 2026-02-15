# 🏗️ Science-Star Project Architecture Guide

Welcome to the **Science-Star** universe! 🚀 This isn't just another project structure - it's a carefully crafted ecosystem designed to empower scientific AI agents. Let's take a journey through each component and discover what makes this platform tick! ⚡

## 🧠 Core Intelligence Hub - `science_star/`

The beating heart of our scientific agent ecosystem! This is where all the magic happens. ✨

### 🎯 **run_hle.py** - The Orchestrator Supreme
Your mission control center! 🎮 This powerful entry point coordinates the entire HLE evaluation workflow, managing agent lifecycles, experiment configurations, and result collection. Think of it as the conductor of a scientific symphony orchestra! 🎼

### 🛠️ **tools/** - The Agent's Swiss Army Knife
A treasure trove of specialized tools, organized by category! 💪

- **search/** - 🔍 search_backends (serpapi, tavily, duckduckgo, wiki), search_tool (SearchTool, AggregatedSearchTool, WaybackSearchTool)
- **crawl/** - 🕷️ crawl_backends (jina, crawl4ai), crawl_tools (CrawlUrlTool, SearchAndCrawlTool)
- **pdf/** - 📄 PDF processing (pdf_extractor, pdf_utils)
- **inspector/** - 👁️ document_inspector (DocumentInspectorTool), audio, visual + convert_backends (doc→markdown)
- **retriever/** - 🔎 Semantic retrieval over text (RetrieverTool, wraps rag_processor)
- **browser/** - 🌐 agent_browser (BrowserUseTool), lynx_browser (SimpleTextBrowser), cookies
- **code/** - 📝 authorized_imports (AUTHORIZED_IMPORTS for code-execution sandbox)

Import directly from submodules, e.g. `from tools.search.search_tool import SearchTool`, `from validator import get_scorer`.

### 📏 **validator/** - Answer Evaluation
Unified entry point for dataset-specific scoring via `get_scorer(dataset, config)`:
- **base.py** - BaseScorer, EvaluationResult
- **llm_judge_scorer.py** - LLM-as-a-Judge (parent, default: gpt-4o-2024-11-20, temp=0, max_tokens=512)
- **gaia_scorer.py** - GAIA: rule-based (numbers, lists, strings, close-call)
- **hle_scorer.py** - HLE: inherits LLM-as-a-Judge

### 📥📤 **io_processor/** - I/O Processing
File context preparation and response reformulation (not tools):

- **file_context.py** - Generate descriptions of attached files for agent context
- **reformulator.py** - Extract final answer from agent conversation

### 📊 **data/** - Data & Loaders
Dataset loaders and data files. 🎭
- **hle_loader.py** - HLE dataset loader
- **gaia_loader.py** - GAIA dataset loader
- **HLE/**, **GAIA/** - Dataset directories

### 🎯 **hle_eval/** - Evaluation Excellence Center
The quality assurance department! 📏✅
- **run_model_predictions.py** - Prediction pipeline orchestrator
- **run_judge_results.py** - Result validation and scoring

### 🤖 **rag_processor/** - Retrieval-Augmented Generation Powerhouse
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

## 🏭 Foundation Framework - `smolagents/` (git submodule)

### 🔧 **smolagents/** - The Engine Room
Built on the powerful smolagents framework! 🚂
- **agents.py** & **agent_types.py** - Core agent abstractions
- **tools.py** & **default_tools.py** - Tool ecosystem
- **models.py** & **monitoring.py** - Model management & observability
- **prompts/** - Rich prompt template library
- **workflow.py** - Workflow orchestration capabilities

## ⚙️ Configuration Command Center - `configs/`

### 📋 **gaia.yaml** / **hle.yaml** - Nested Config Structure
Your experiment's DNA! 🧬 All configs use nested sections. CLI override with dot notation: `runtime.concurrency=2`, `dataset.subset=medium`, `models.main=gpt-4o`.

- **runtime/** - concurrency, debug, run_name, date_time_load_from
- **dataset/** - name, subset, level (gaia) | category (hle), selected_tasks
- **models/** - name, temperature, max_tokens (gaia)
- **agents/** - max_steps, planning_interval, verbose (gaia)
- **validator/** - model, temperature, max_tokens (for LLM-as-a-Judge)

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
- Dive into `science_star/rag_processor/` for advanced retrieval features
- Check `visualization/` for data exploration and result analysis
- Modify `configs/hle.yaml` for experiment customization

🚀 **Ready to Build Something Amazing?**
This architecture is designed for extensibility and experimentation. Each component is carefully crafted to work harmoniously while maintaining clear boundaries. Whether you're building new tools, integrating novel models, or creating custom evaluation pipelines, Science-Star provides the foundation you need!

Happy coding, and may your agents be ever intelligent! 🤖✨