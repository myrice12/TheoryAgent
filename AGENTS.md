# TheoryAgent 项目说明（供 Codex 使用）

TheoryAgent 是一个分阶段运行的研究自动化仓库。Codex 在本项目中应复用现有工作流，而不是另外创造一套只给 Codex 使用的执行路径。

## 目标

给定一个研究主题，TheoryAgent 应产出一个可恢复的研究工作区，至少包含以下内容：

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

## 工作区规则

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
- 主题前缀由现有 CLI 与 manifest 逻辑解析，Codex 应直接复用

## 意图映射

| 如果用户要做什么 | Codex 应如何理解 | 优先使用的入口或行为 |
| --- | --- | --- |
| 完整研究运行 | 从主题到论文的全流程 | `theoryagent run --topic "..."` 或等价的工作区流程 |
| ideation 或文献综述 | 第 1 阶段 ideation | 在工作区中产出 `papers/ideation_output.json` |
| planning | 第 2 阶段 planning | 产出 `plans/experiment_blueprint.json` 或 `plans/survey_blueprint.json` |
| 实验执行 | setup + coding + execution | 原始研究模式走现有实验路径 |
| analysis | 第 6 阶段证据提取 | 产出 `plans/analysis_output.json` |
| writing | 图表生成 + 论文撰写 | 在工作区中产出或更新论文相关文件 |
| review | 审阅与修订 | 产出或更新 `drafts/review_output.json` 与修订稿 |
| status | 查看当前工作区状态 | 读取并规范化 `manifest.json` |
| resume | 恢复中断运行 | 从第一个未完成阶段继续 |

## 约束规则

- 不得伪造实验结果
- 不得伪造引用
- 优先使用已有 CLI / orchestrator 行为，而不是临时脚本
- 论文中的结论应尽量绑定到真实输出或可核对文献
- 通过 `manifest.json` 保持检查点和恢复语义

## 操作规则

1. 驱动系统时优先使用已有的 TheoryAgent CLI 或 Python 入口。
2. 不要引入只面向 Codex 的独立执行模式。
3. 产物要保持与现有工作区和 manifest 兼容。
4. 不得伪造结果或引用。
5. 当用户请求流水线相关工作时，应沿用现有工作区产物和阶段边界推进。
