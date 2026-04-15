# 🔬 TheoryAgent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

**TheoryAgent** 是一款面向科研场景的自动化研究工作流引擎。它将复杂的研究过程解构为 9 个连续、可观测且可恢复的阶段，支持从“研究主题”到“高质量论文草稿”的全流程自动化产出。

---

## 🚀 核心价值主张 (Core Value Proposition)

不同于单纯的文本生成工具，TheoryAgent 强调**研究过程的完整性**与**结论的可靠性**：

*   **🔍 全链路可追踪 (Traceable)**：每个研究阶段均产生结构化的中间产物，记录完整的决策链路。
*   **🔄 断点续跑 (Resumable)**：基于 `manifest.json` 的状态持久化机制，支持在任何中断点无损恢复运行。
*   **⚖️ 结论可核对 (Verifiable)**：生成的论文结论严格绑定到实验输出或真实的文献证据。
*   **🏗️ 模块化架构**：支持原始研究 (Original Research) 与 文献综述 (Survey) 两种核心模式。

---

## 🛠️ 标准研究工作流 (Workflow)

TheoryAgent 遵循严格的九阶段线性推进逻辑：

```mermaid
graph LR
    IDEATION[文献检索] --> PLANNING[实验蓝图]
    PLANNING --> SETUP[环境准备]
    SETUP --> CODING[代码生成]
    CODING --> EXECUTION[实验执行]
    EXECUTION --> ANALYSIS[证据提取]
    ANALYSIS --> FIGURE_GEN[图表生成]
    FIGURE_GEN --> WRITING[论文撰写]
    WRITING --> REVIEW[自动审阅]
```

1.  **IDEATION**: 检索前沿文献，识别研究空白，提炼核心假设。
2.  **PLANNING**: 生成详细的实验方案或综述大纲。
3.  **SETUP**: 自动准备依赖环境、代码参考及计算资源。
4.  **CODING**: 基于蓝图生成可运行的实验代码。
5.  **EXECUTION**: 在本地或高性能集群 (SLURM) 上执行实验。
6.  **ANALYSIS**: 从原始日志中提取结构化证据，进行统计分析。
7.  **FIGURE_GEN**: 自动绘制高质量的论文插图与实验图表。
8.  **WRITING**: 组装 LaTeX 源码，产出符合顶级会议标准的论文草稿。
9.  **REVIEW**: 进行多维度的审阅修订与编译检查。

---

## 📦 安装与配置 (Setup)

### 1. 快速安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd theory-agent

# 创建并激活环境
python -m venv .venv
source .venv/bin/activate

# 以编辑模式安装
pip install -e .
```

### 2. 初始化运行目录

TheoryAgent 默认在项目根目录下的 `./.theoryagent/` 存储所有运行时数据。

```bash
mkdir -p ./.theoryagent
# 创建配置文件
touch ./.theoryagent/config.json
```

### 3. 配置示例 (`config.json`)

```json
{
  "research": {
    "base_url": "https://api.example.com/v1",
    "api_key": "your-api-key",
    "template_format": "neurips",
    "execution_profile": "local_quick"
  }
}
```

---

## 📖 快速上手 (Quick Start)

### 模式一：原始研究 (Original Research)
```bash
theoryagent run --topic "original: Adaptive Sparse Attention in Transformers" --format neurips
```

### 模式二：文献综述 (Literature Survey)
```bash
# 支持 short, standard, long 三种深度
theoryagent run --topic "survey:standard: Efficient Multimodal Learning"
```

### 常用管理命令
*   **查看进度**: `theoryagent status --workspace <session_dir>`
*   **恢复运行**: `theoryagent resume --workspace <session_dir>`
*   **导出产物**: `theoryagent export --workspace <session_dir> --output ./my_paper`
*   **健康检查**: `theoryagent health`

---

## 📂 项目结构 (Repository Structure)

```text
theory-agent/
├── theoryagent/         # 核心逻辑：Agent 行为、Pipeline 编排、数据 Schema
├── mcp_server/          # MCP 工具服务：搜索、PDF 解析、图表生成、LaTeX 编译
├── skills/              # 任务技能定义文档
├── theoryagent_.../     # 示例输出产物（论文、代码、中间数据）
└── pyproject.toml       # 项目依赖与入口定义
```

详细架构说明请参考 [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)。

---

## 📄 许可证 (License)

本项目基于 [MIT License](LICENSE) 开源。

---
<p align="center">Built with ❤️ for the AI Research Community</p>
