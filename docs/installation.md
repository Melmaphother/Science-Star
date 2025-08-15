### Environment Setup

**Clone the repository**
```bash
git clone git@github.com:Melmaphother/Open-Agent.git
cd Open-Agent/
```

**Install dependencies**
```bash
# Create the conda environment
conda create -n open-agent python==3.10
conda activate open-agent
# Install the required dependencies from the requirements.txt
pip install -r requirements.txt
pip install crawl4ai==0.6.3
pip install langchain==0.3.23
# Install smolagents
cd src
pip install -e ./.[dev]
```