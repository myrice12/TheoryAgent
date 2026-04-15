# TheoryAgent 项目说明（供 Claude Code 使用）

TheoryAgent 是一个分阶段运行的研究自动化仓库。Claude Code 在本项目中应使用仓库内置的工作流，而不是额外创建一条只给 Claude 使用的平行路径。

## 目标

给定一个研究主题，TheoryAgent 应留下一个可恢复的研究工作区，至少包含以下内容：

- 文献相关产物
- 规划阶段产物
- 在需要时生成可运行的实验代码
- 执行结果或文献分析证据
- 图表
- LaTeX 论文草稿
- 审阅结果与最终导出产物

## 工作流

TheoryAgent 采用固定的多阶段流程：

```text
IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS -> FIGURE_GEN -> WRITING -> REVIEW
```

各阶段含义如下：

- `ideation`：检索文献、寻找研究空白、提炼假设或主题
- `planning`：生成实验蓝图或综述蓝图
- `setup`：准备环境与资源
- `coding`：生成可运行实验
- `execution`：本地或基于 SLURM 执行实验
- `analysis`：从输出中提取结构化证据
- `figure_gen`：生成论文图表资源
- `writing`：撰写 LaTeX 论文
- `review`：审阅、核验与修订

## 工作区约定

工作区默认位于仓库根目录下的 `./.theoryagent/workspace/research/`。

典型目录结构如下：

```text
{session_dir}/
├── manifest.json
├── papers/
├── plans/
├── experiment/
├── drafts/
├── figures/
├── output/
└── logs/
```

当用户要求继续、查看状态、恢复运行或修订旧结果时，应优先复用已有工作区。

## 论文模式

主题前缀约定如下：

- `original: Topic` -> `original_research`
- `survey:short: Topic` -> `survey_short`
- `survey:standard: Topic` -> `survey_standard`
- `survey:long: Topic` -> `survey_long`

行为约定如下：

- 原始研究模式走完整阶段流程
- 综述模式跳过重实验阶段，重点使用文献驱动的 planning、writing 与 review
- 主题前缀由现有 CLI 与 manifest 逻辑解析，Claude Code 应直接复用

## 可用命令

| 命令 | 说明 |
| --- | --- |
| `/project:research` | 运行完整阶段流程 |
| `/project:ideation` | 运行第 1 阶段的文献检索与想法生成 |
| `/project:planning` | 运行第 2 阶段 planning |
| `/project:experiment` | 运行原始研究的 setup、coding 与 execution |
| `/project:analysis` | 运行实验分析 |
| `/project:writing` | 运行图表生成与论文撰写 |
| `/project:review` | 运行审阅与修订 |
| `/project:status` | 查看工作区状态 |
| `/project:resume` | 从最近检查点恢复 |

## 约束规则

- 不得伪造实验结果
- 不得伪造引用
- 优先使用已有 CLI / orchestrator 行为，而不是临时脚本
- 论文中的结论应尽量绑定到真实输出或可核对文献
- 通过 `manifest.json` 保持检查点和恢复语义

## Claude Code 的角色

在本仓库中，Claude Code 是研究工作流的操作者，主要借助原生工具完成任务：

- **WebSearch**：检索学术文献
- **Bash**：执行代码、提交 SLURM、编译 LaTeX
- **File read/write**：处理工作区产物、代码和论文草稿

## Claude Code 规则

1. 使用本项目既有的工作区和 manifest 约定。
2. 综述模式沿用现有主题前缀语法。
3. 优先使用 TheoryAgent 既有工作流和输出，而不是一次性脚本。
4. 不得伪造结果或引用。
5. 保持工作区与 Python CLI 兼容。
