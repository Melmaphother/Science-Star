<h1 align="center"> Open-Agent: An Open Platform for Building, Extending, and Experimenting with Scientific Agents. </h1>

<p align="center">
  <a href="https://github.com/Melmaphother/Open-Agent/stargazers"><img src="https://img.shields.io/github/stars/Melmaphother/Open-Agent" alt="GitHub Repo stars"></a>
  <a href="https://github.com/Melmaphother/Open-Agent/network/members"><img src="https://img.shields.io/github/forks/Melmaphother/Open-Agent" alt="GitHub forks"></a>
  <a href="https://github.com/Melmaphother/Open-Agent/blob/main/assets/wechat.jpeg"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</p>

## 📢 News

<details open>
<summary><b>Recent Updates</b></summary>

- [2025.08.15] **Open-Agent Init**: We release Open-Agent. It is an user-friendly open platform for building, extending, and experimenting with scientific agents.

</details>

## 🧠 Overview

**Open-Agent** is an open-source framework designed to accelerate research and development at the critical intersection of **RL** and **Agent**. Our framework employs **End-to-End** reinforcement learning to train agents in specific environments. Developers need only define domain-specific tools and reward functions to extend Agent-R1 to their unique use cases, eliminating the need for complex workflow engineering. We hope our modest contribution can benefit the open-source community, making it easier for researchers and developers to create and explore agents in their own domains, collectively advancing the development of autonomous agents. For more details on the algorithm, see [algorithm doc](https://github.com/0russwest0/Agent-R1/blob/main/docs/algorithm/algorithm.md).

> **Also check out [Awesome-Agent](https://github.com/0russwest0/Awesome-Agent-RL)**: Our curated collection of papers and resources on unlocking the potential of Agents through Reinforcement Learning.

## 🔥 Key Features

- **Integrated Visualization**: An end-to-end, extensible visualization suite powered by [streamlit](https://github.com/streamlit/streamlit). It streamlines the entire workflow from data inspection and real-time experiment monitoring to results logging and analysis.

- **Plug-and-Play Modularity**: Core components (`DataLoader`, `Memory`, `Planner`, `Tool`, `Evaluator`) are designed with well-defined interfaces. This modularity enables effortless substitution and customization.

- **Scientific Extensibility**: A dedicated adaptation layer for seamless integration of scientific tools (e.g., Chemistry, Biology). It has built-in support for advanced retrieval and literature-based Retrieval-Augmented Generation (RAG).

## 🚀 Easy Experiments

- [Brief Project Structure](https://github.com/Melmaphother/Open-Agent/blob/main/docs/project_structure.md)
- [Environment Setup](https://github.com/Melmaphother/Open-Agent/blob/main/docs/installation.md)
- [Quick Start: Try o4-mini + ReAct on HLE-Small](https://github.com/Melmaphother/Open-Agent/blob/main/docs/quickstart.md)

## 🛠️ Extending Open-Agent with Your Own Tools and Workflows

Additional resources are available in the codebase:

- Example tools: `open_agent/tools/`
- Data preprocessing: `open_agent/data_utils`
- Visualization: `visualization`

## ℹ️ Feedback

We welcome all forms of feedback! Please raise an issue for bugs, questions, or suggestions. This helps our team address common problems efficiently and builds a more productive community.

**Join our community**: Connect with other users and our development team in our [WeChat group](https://github.com/Melmaphother/Open-Agent/assets/wechat.jpeg).

## 🤝 Contributors

**Student Contributors**: [**Daoyu Wang**](https://github.com/Melmaphother), [**Qingchuan Li**](https://github.com/wufeiwuwoshihua), [**Tian Gao**](https://github.com/SkyeGT), [**Shuo Yu**](https://github.com/fishsure), [**Xiaoyu Tao**](https://github.com/Xiaoyu-Tao)

**Supervisors**: [**Qi Liu**](http://staff.ustc.edu.cn/~qiliuql/), [**Mingyue Cheng**](https://mingyue-cheng.github.io/)

**Affiliation**: **State Key Laboratory of Cognitive Intelligence, USTC**

## 🥰 Acknowledgements

We extend our gratitude to [OAgent](https://github.com/OPPO-PersonalAI/OAgents) for providing the OAgent and hard work in engineering. We are also thankful to the [smolagent](https://github.com/huggingface/smolagents) team for their fundamental support. Lastly, we deeply appreciate the insightful discussions and contributions from Daoyu Wang, Qingchuan Li, Tian Gao, Shuo Yu, Xiaoyu Tao.

## ✍️ Citation

**Open-Agent**

```md
@misc{Open-Agent,
author = {Daoyu Wang, Qingchuan Li, Tian Gao, Shuo Yu, Xiaoyu Tao},
title = {Open-Agent: An Open Platform for Building, Extending, and Experimenting with Scientific Agents.},
year = {2025},
organization = {GitHub},
url = {https://github.com/Melmaphother/Open-Agent},
}
```
