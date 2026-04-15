# TheoryAgent 总体架构与目录说明

本文档用于解释 TheoryAgent 的整体架构、主数据流，以及仓库中各个文件夹的功能和作用。阅读完本文档后，你应该能回答三类问题：

- 这个项目整体是怎么工作的
- 核心能力分别落在哪些目录
- 哪些目录属于源码，哪些目录属于样例、缓存或运行时产物

## 1. 项目总体定位

TheoryAgent 是一个分阶段运行的研究自动化系统。它不是只负责“生成论文文本”，而是把研究流程拆成多个连续阶段，并把每个阶段的产物写入工作区，便于恢复、审计和导出。

默认主流程如下：

```text
IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS -> FIGURE_GEN -> WRITING -> REVIEW
```

这 9 个阶段对应的职责可以概括为：

- `IDEATION`：检索文献、整理问题、提炼研究假设
- `PLANNING`：生成实验蓝图或综述蓝图
- `SETUP`：准备代码参考、运行环境与资源
- `CODING`：生成实验代码或项目骨架
- `EXECUTION`：在本地或集群执行实验
- `ANALYSIS`：从日志和结果中提取结构化证据
- `FIGURE_GEN`：生成论文图表或示意图
- `WRITING`：写作论文草稿并组装 LaTeX
- `REVIEW`：自动审阅、修订与编译检查

## 2. 整体架构分层

从工程角度看，仓库可以分成 7 层：

```text
CLI / 外部入口
    ↓
Pipeline 编排层
    ↓
Agent 执行层
    ↓
Tools / MCP 工具层
    ↓
Prompt 与模板层
    ↓
Schema 结构层
    ↓
Workspace 与导出层
```

每一层大致对应如下目录：

- 外部入口：`theoryagent/cli.py`、`theoryagent/cli_commands.py`、`theoryagent/feishu_bot.py`
- 编排层：`theoryagent/pipeline/`
- 执行层：`theoryagent/agents/`
- 工具层：`mcp_server/`
- 提示词与模板层：`theoryagent/prompts/`、`theoryagent/templates/`
- 结构层：`theoryagent/schemas/`
- 工作区层：`theoryagent/pipeline/workspace.py` 以及仓库根目录下的 `./.theoryagent/workspace/`

## 3. 主执行链路

以命令行为例，系统的主链路通常是这样的：

```text
用户命令
-> CLI 解析参数与配置
-> 创建或加载 workspace
-> UnifiedPipelineOrchestrator 决定当前该跑哪一阶段
-> 对应 Agent 执行任务
-> Agent 调用工具、提示词、模板和 schema
-> 阶段输出写回 workspace
-> 全部阶段结束后导出论文与相关产物
```

其中最关键的几个入口文件是：

- `theoryagent/cli.py`：主命令入口
- `theoryagent/pipeline/unified_orchestrator.py`：统一编排入口
- `theoryagent/pipeline/deep_orchestrator.py`：完整阶段流程主干
- `theoryagent/pipeline/workspace.py`：工作区创建、加载与 manifest 管理

## 4. 仓库顶层目录说明

下面先看仓库根目录下最重要的文件夹。

### `theoryagent/`

这是项目的核心 Python 包，几乎所有主要逻辑都在这里。它负责：

- CLI 和配置加载
- 流水线编排
- 各阶段 agent
- 数据结构定义
- 写作模板和提示词
- LaTeX 处理

如果你只看一个目录，优先看这个。

### `mcp_server/`

这是项目的 MCP 工具服务层，负责把“搜索论文、搜索 GitHub、读取 PDF、生成图表、编译 PDF”等能力暴露成标准工具接口，供 agent 或外部兼容 MCP 的系统调用。

### `skills/`

这里存放的是技能说明文档。每个子目录对应一类高层任务说明，例如 ideation、planning、experiment、writing。它们不是核心执行代码，而是帮助代理理解阶段目标与输入输出约定的说明层。

### `theoryagent_Adaptive_Sparse_Attention_Mechanisms_e5464161/`

这是一个示例输出目录，可以把它理解为“导出的研究结果样本”。它展示了一次研究运行结束后，论文、参考文献、结构化中间结果等会是什么样子。

### 运行时与缓存目录

仓库中可能还会出现一些不是源码的目录：

- `.venv/`：本地虚拟环境
- `.pytest_cache/`：pytest 缓存
- `__pycache__/`：Python 编译缓存
- `.DS_Store`：macOS 生成的系统文件

这些目录不属于业务架构主体，只是本地运行或系统自动生成的附属内容。

## 5. `theoryagent/` 包的详细说明

### `theoryagent/agents/`

这是项目最核心的执行层。每个阶段都对应一个或一组 agent 实现。

它的职责包括：

- 组织阶段内部逻辑
- 调用工具和模型
- 读取上游阶段产物
- 产出当前阶段的结构化结果

这个目录下既有各阶段入口，也有大量辅助模块。

#### `theoryagent/agents/analysis/`

负责分析实验结果、统计指标、对比方法、生成结构化分析结论。

常见职责：

- 训练动态分析
- 消融分析
- 结果对比矩阵
- 统计摘要

#### `theoryagent/agents/execution/`

负责运行实验以及处理执行期问题。

常见职责：

- 本地执行
- quick eval
- dry run
- 结果回收
- 执行期修复
- 运行失败恢复

这是“把代码真正跑起来”的核心目录之一。

#### `theoryagent/agents/experiment/`

负责实验生成和迭代控制，处于“规划”和“执行”之间。

常见职责：

- 生成代码
- 代码运行辅助
- 快速评测
- ReAct 模式实验控制
- 迭代式改进实验

#### `theoryagent/agents/figure_gen/`

负责图表和论文可视化资源的生成。

常见职责：

- 规划图表
- 生成图代码
- 生成 AI 示意图
- 保存和裁剪图像

#### `theoryagent/agents/review/`

负责论文审阅和修订。

常见职责：

- 单审稿人或多审稿人模式
- 一致性检查
- 自动应用修订
- LaTeX 编译修复
- 审稿意见抽取

#### `theoryagent/agents/runtime_env/`

负责实验运行环境管理。

常见职责：

- 发现本地 Python/conda/venv
- 安装依赖
- 检查 GPU
- 管理运行时环境清单
- 校验环境可用性

#### `theoryagent/agents/writing/`

负责论文内容组装和写作。

常见职责：

- 章节上下文构建
- 引文管理
- grounding 信息整理
- 图表与表格嵌入
- LaTeX 组装
- 章节写作

#### `theoryagent/agents/` 根目录下的其他重要文件

除了子目录，这一层还放了很多关键模块：

- `base.py`：所有研究 agent 的抽象基类
- `ideation.py`：文献检索与假设生成入口
- `planning.py`：实验蓝图生成入口
- `setup.py`：资源与环境准备入口
- `coding.py`：代码生成入口
- `paper_editor.py`、`code_editor.py`：交互式编辑能力
- `cluster_executor.py`：集群执行器
- `project_runner_*`：运行器脚本生成、校验与执行
- `tools.py`、`experiment_tools.py`：实验相关工具定义
- `repair_journal.py`：修复日志记录

### `theoryagent/pipeline/`

这是编排层，负责决定“当前阶段是什么、下一步怎么走、状态如何持久化”。

核心职责：

- 创建和驱动 orchestrator
- 维护状态机
- 追踪进度
- 统计成本
- 校验蓝图
- 读写工作区

其中几个关键文件：

- `unified_orchestrator.py`：统一入口
- `deep_orchestrator.py`：完整阶段编排主干
- `base_orchestrator.py`：重试、跳过、保存等通用逻辑
- `state.py`：阶段状态机
- `workspace.py`：工作区和 manifest 管理
- `progress.py`：进度表示
- `cost_tracker.py`：成本统计

### `theoryagent/schemas/`

这是数据结构层，负责定义项目中大量结构化产物的格式。

常见对象包括：

- `ideation.py`：文献检索与假设输出结构
- `experiment.py`：实验蓝图结构
- `evidence.py`：证据结构
- `paper.py`：论文草稿结构
- `review.py`：审稿结果结构
- `manifest.py`：工作区 manifest 结构
- `iteration.py`：实验迭代状态结构
- `figure.py`、`writing.py`：图表与写作相关结构

这一层的作用是把“自然语言阶段产物”尽可能约束为可校验的数据对象。

### `theoryagent/prompts/`

这是提示词资源目录，按功能拆分。

#### `theoryagent/prompts/ideation/`

负责 ideation 阶段使用的提示词，例如：

- 查询生成
- 研究空白分析
- 必引文献约束
- 证据提取

#### `theoryagent/prompts/experiment/`

负责实验阶段提示词，例如：

- 生成项目计划
- 生成文件内容
- ReAct 实验系统提示词

#### `theoryagent/prompts/figure_gen/`

负责图表与示意图生成相关提示词。

其中又分成两类：

- `chart_types/`：不同图表类型的生成规则
- `ai_templates/`：不同 AI 示意图模板，如结构图、流程图、对比图等

#### `theoryagent/prompts/writing/`

负责论文写作阶段提示词，按章节拆分，例如摘要、引言、方法、实验、结论等。

#### `theoryagent/prompts/review/`

负责审稿和修订阶段提示词，也按章节或任务拆分。

### `theoryagent/templates/`

这是 LaTeX 模板资源目录。

#### `theoryagent/templates/base/`

放基础章节模板，例如：

- 摘要
- 引言
- 方法
- 实验
- 结果
- 结论
- 参考文献
- 整体论文骨架

#### `theoryagent/templates/neurips/`

NeurIPS 风格模板与样式文件。

#### `theoryagent/templates/icml/`

ICML 风格模板。

#### `theoryagent/templates/arxiv/`

通用 arXiv 输出模板。

#### `theoryagent/templates/survey/`

综述型论文模板。

### `theoryagent/latex/`

负责 LaTeX 修复和辅助处理。

主要用于：

- 处理 LaTeX 细节问题
- 修复编译相关问题
- 给写作与 review 阶段提供底层支持

### `theoryagent/` 根目录下的其他核心文件

这些文件虽然不是文件夹，但对理解架构很关键：

- `cli.py`：主命令入口
- `cli_commands.py`：扩展命令，如导出、检查、飞书、环境选择
- `cli_code_edit.py`：代码编辑命令入口
- `cli_paper_edit.py`：论文编辑命令入口
- `config.py`：全局配置与阶段模型配置
- `constants.py`：常量定义
- `exceptions.py`：异常定义
- `feishu_bot.py`、`feishu_bot_core.py`、`feishu_bot_handlers.py`：飞书机器人入口与逻辑
- `skill_prompts.py`、`skills.py`：技能匹配与技能提示词注入
- `smoke_execution.py`、`_smoke_helpers.py`：端到端 smoke 测试辅助

## 6. `mcp_server/` 目录说明

`mcp_server/` 是工具服务层，目的是把仓库内部能力标准化暴露出去。

### `mcp_server/server.py`

这是 MCP stdio 服务入口，负责：

- 注册工具
- 接收请求
- 调度到具体工具实现
- 返回结构化结果

### `mcp_server/tools/`

这里是一组具体工具实现。

主要包括：

- `arxiv_search.py`：搜索 arXiv
- `semantic_scholar.py`：搜索 Semantic Scholar
- `openalex.py`：搜索 OpenAlex
- `paperswithcode.py`：查询 Papers With Code
- `github_search.py`：搜索 GitHub 仓库
- `web_search.py`：通用网页搜索
- `pdf_reader.py`：下载并解析 PDF
- `latex_gen.py`：生成 LaTeX
- `pdf_compile.py`：编译 PDF
- `figure_gen.py`：生成图像或图表

### `mcp_server/utils.py`

放工具层的公共辅助逻辑。

## 7. `skills/` 目录说明

`skills/` 保存的是技能说明文件，而不是核心执行代码。

当前包含：

- `skills/theoryagent-ideation/`：文献检索与研究假设生成技能说明
- `skills/theoryagent-planning/`：实验蓝图生成技能说明
- `skills/theoryagent-experiment/`：实验代码骨架生成技能说明
- `skills/theoryagent-writing/`：论文草稿写作技能说明

每个子目录目前主要包含一个 `SKILL.md`，用于描述：

- 技能目标
- 所需工具
- 输入
- 处理流程
- 输出

它们更像“任务规范”，而不是 Python 模块。

## 8. 示例输出目录说明

### `theoryagent_Adaptive_Sparse_Attention_Mechanisms_e5464161/`

这个目录是一次示例运行的导出结果。它的价值在于帮助理解“项目最终会产出什么”。

其中包含：

- `paper.pdf`：论文 PDF
- `paper.tex`：LaTeX 源文件
- `references.bib`：参考文献
- `manifest.json`：会话执行记录
- `README.md`：该导出目录的说明
- `data/`：结构化中间结果
- `code/`：导出的代码目录

#### `theoryagent_Adaptive_Sparse_Attention_Mechanisms_e5464161/data/`

保存结构化阶段产物，例如：

- `ideation_output.json`
- `experiment_blueprint.json`
- `paper_skeleton.json`

#### `theoryagent_Adaptive_Sparse_Attention_Mechanisms_e5464161/code/`

用于保存导出的实验代码或项目文件。当前样例中该目录存在，但未必包含大量内容，是否充实取决于当次运行是否完成完整代码导出。

## 9. 仓库顶层文件说明

虽然你问的是“文件夹”，但顶层几个文件也很重要：

- `README.md`：主说明文档
- `PROJECT_ANALYSIS.md`：对项目代码结构的分析文档
- `AGENTS.md`：面向 Codex 的项目约定
- `CLAUDE.md`：面向 Claude Code 的项目约定
- `pyproject.toml`：Python 包配置、依赖和 CLI 入口定义
- `requirements.txt`：依赖列表
- `LICENSE`：许可证

## 10. 运行时目录说明

除了仓库本身，项目还有一个非常重要的运行时目录：

### `./.theoryagent/`

这是默认运行目录，位于仓库根目录下，对系统运行非常关键。

通常会包含：

- `config.json`：运行配置
- `workspace/research/`：研究工作区
- `chat_memory/`：聊天记忆
- `cache/models/`：模型缓存
- `cache/data/`：数据缓存

### `./.theoryagent/workspace/research/<session_id>/`

这是每次研究运行的核心工作区。典型结构如下：

```text
{session_id}/
├── manifest.json
├── papers/
├── plans/
├── experiment/
├── drafts/
├── figures/
├── output/
└── logs/
```

其中：

- `papers/`：文献与 ideation 输出
- `plans/`：蓝图、执行摘要、分析结果
- `experiment/`：代码与运行上下文
- `drafts/`：论文草稿与 review 结果
- `figures/`：图像资源
- `logs/`：日志、错误与成本记录
- `output/`：最终导出内容

## 11. 哪些目录最值得先读

如果是第一次接手项目，建议按这个顺序理解目录：

1. `README.md`
2. `theoryagent/cli.py`
3. `theoryagent/pipeline/`
4. `theoryagent/agents/`
5. `theoryagent/schemas/`
6. `theoryagent/prompts/`
7. `theoryagent/templates/`
8. `mcp_server/`
9. `skills/`
10. 示例输出目录和 `./.theoryagent/workspace/`

## 12. 一句话总结每个核心目录

如果只用一句话概括：

- `theoryagent/`：项目主程序
- `theoryagent/agents/`：各阶段执行逻辑
- `theoryagent/pipeline/`：流程编排与状态管理
- `theoryagent/schemas/`：结构化数据定义
- `theoryagent/prompts/`：提示词资源
- `theoryagent/templates/`：LaTeX 模板
- `theoryagent/latex/`：LaTeX 修复与辅助
- `mcp_server/`：外部工具服务层
- `skills/`：技能说明层
- `theoryagent_.../`：导出样例目录
- `./.theoryagent/`：真实运行时数据目录

如果你后续还想进一步深入，我建议下一步可以再补一份“按文件级别展开”的文档，也就是把 `theoryagent/agents/` 和 `theoryagent/pipeline/` 里的关键 Python 文件逐个解释清楚。
