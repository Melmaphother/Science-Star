# 🛠️ Installation Guide

Ready to set up your scientific AI laboratory? Let's get Science-Star running on your system! 🚀

## 📦 Quick Setup

### 🔄 **Step 1: Clone the Repository**
Get the latest version of Science-Star:

```bash
git clone git@github.com:Melmaphother/Science-Star.git
cd Science-Star/
```

### 🐍 **Step 2: Create Your Python Environment**
Set up a clean conda environment for optimal performance:

> ⚠️ **Note:** The `browser-use` package requires Python ≥ 3.11. We use Python 3.11 to avoid compatibility issues.

```bash
# Create a fresh conda environment with Python 3.11
conda create -n science_star python=3.11 -y

# Activate the environment
conda activate science_star
```

### 📚 **Step 3: Install Dependencies**
Install all the required packages for your AI agents:

```bash
# Install smolagents dependencies
# Clone submodules (smolagents)
git submodule update --init --recursive

# Install the smolagents framework (development mode)
cd smolagents
pip install -e ".[dev]"

# Install core dependencies
cd ..
pip install -r requirements.txt
```

> **Optional – crawl4ai backend:** If you need the crawl4ai crawler (alternative to Jina, no API key):
> ```bash
> pip install crawl4ai
> playwright install chromium
> ```
> crawl4ai uses Playwright; `playwright install chromium` installs the Chromium browser.

## 🎉 Configure Environment Variables

```bash
# Return to project root (from smolagents/)
cd ..

# Copy template and edit with your API keys
cp .env_template .env
# Edit .env and fill in: HF_TOKEN, SERP_API_KEY or TAVILY_API_KEY, JINA_API_KEY (or use crawl4ai), OPENAI_BASE_URL, OPENAI_API_KEY
```

## 💻 Run Tests

Ensure the conda environment is activated, then run all API tests:

```bash
conda activate science_star
./test/run_all_tests.sh
```

Tests cover: Hugging Face token, Search API (SerpAPI/Tavily), Crawler (Jina/crawl4ai), LLM API (OpenAI-compatible). At least one search backend and one crawler backend must be configured for tests to pass.

## 🆘 Need Help?

- 🐛 Report installation issues on our [GitHub Issues](https://github.com/Melmaphother/Science-Star/issues)
- 📖 Check the [Project Structure](project_structure.md) for more details

---

**🎉 Congratulations!** Your Science-Star installation is complete! Ready to dive into the [Quick Start Guide](quickstart.md)? 🚀✨