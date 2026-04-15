---
name: theoryagent-writing
description: 根据前序阶段输出撰写 LaTeX 论文草稿
version: 0.1.0
---

# 写作技能

## 目标
读取前序阶段输出，包括 ideation、planning 和实验结果，生成包含图表、表格和参考文献的完整 LaTeX 论文草稿。

## 所需工具
- `generate_latex`：为各章节生成并组装 LaTeX 源文件
- `compile_pdf`：将 LaTeX 源文件编译为 PDF
- `generate_figure`：根据实验结果生成适合论文使用的图表

## 输入
- `ideation_output`：来自 ideation 技能的 `papers/ideation_output.json`
- `experiment_blueprint`：来自 planning 技能的 `papers/experiment_blueprint.json`
- `experiment_results`：来自 experiment 技能、包含代码与结果的 `experiments/` 目录

## 处理流程
1. 解析所有上游输出，收集假设、文献、实验设计与实验结果
2. 按标准论文结构生成提纲，如摘要、引言、相关工作、方法、实验、结论
3. 撰写摘要，概括问题、方法与核心发现
4. 撰写引言，说明研究动机与贡献
5. 撰写相关工作，总结 ideation 阶段得到的文献
6. 撰写方法章节，详细描述提出的方法
7. 撰写实验章节，覆盖数据集描述、baseline 对比与消融结果
8. 使用 `generate_figure` 生成图表，如性能曲线、消融图和结构图
9. 生成量化结果表格
10. 撰写结论，总结发现并给出后续方向
11. 根据所有引用论文整理参考文献
12. 使用 `generate_latex` 组装完整 LaTeX 文档
13. 使用 `compile_pdf` 编译 PDF，并检查输出是否可用

## 输出
产出 `papers/draft/` 目录，内容包括：
- `main.tex`：完整论文的 LaTeX 源文件
- `references.bib`：包含所有引用的参考文献文件
- `figures/`：生成的 PDF 或 PNG 图表
- `tables/`：LaTeX 表格源文件
- `main.pdf`：编译得到的论文 PDF
