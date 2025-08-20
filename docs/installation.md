# 🛠️ Installation Guide

Ready to set up your scientific AI laboratory? Let's get Open-Agent running on your system! 🚀

## 📦 Quick Setup

### 🔄 **Step 1: Clone the Repository**
Get the latest version of Open-Agent:

```bash
git clone git@github.com:Melmaphother/Open-Agent.git
cd Open-Agent/
```

### 🐍 **Step 2: Create Your Python Environment**
Set up a clean conda environment for optimal performance:

```bash
# Create a fresh conda environment with Python 3.10
conda create -n open-agent python==3.10
conda activate open-agent
```

### 📚 **Step 3: Install Dependencies**
Install all the required packages for your AI agents:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional specialized packages
pip install crawl4ai==0.6.3
pip install langchain==0.3.23

# Install the smolagents framework (development mode)
cd src
pip install -e ./.[dev]
```

## 🎉 Verification

Once installation completes, verify everything is working:

```bash
# Return to project root
cd ..

# Quick test - this should show the help message
python3 open_agent/run_hle.py --help
```

## 🚨 Common Installation Issues (Don't Panic!)

During installation, you might see some scary-looking error messages. **Don't worry!** These are typically harmless warnings. Here's what to expect:

### ⚠️ **Error Type 1: crawl4ai Installation Warnings**

![pip error 1](../assets/pip_error1.png)

**What's happening?** 🤔 
- Dependency version conflicts between packages
- Build warnings from native extensions
- Missing optional dependencies

**Solution:** ✅ As long as you see `Successfully installed crawl4ai-0.6.3` at the end, you're good to go!

### ⚠️ **Error Type 2: smolagents Development Installation**

![pip error 2](../assets/pip_error2.png)

**What's happening?** 🤔
- Development mode installation warnings
- Optional dependency conflicts
- Build system notifications

**Solution:** ✅ Look for `Successfully installed` message - that's your green light! 🟢

## 🆘 Need Help?

- 💬 Join our [WeChat community](../assets/wechat.jpeg) for real-time support
- 🐛 Report installation issues on our [GitHub Issues](https://github.com/Melmaphother/Open-Agent/issues)
- 📖 Check the [Project Structure](project_structure.md) for more details

---

**🎉 Congratulations!** Your Open-Agent installation is complete! Ready to dive into the [Quick Start Guide](quickstart.md)? 🚀✨