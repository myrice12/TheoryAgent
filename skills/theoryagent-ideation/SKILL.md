---
name: theoryagent-ideation
description: 检索学术文献并生成研究假设
version: 0.1.0
---

# 构思技能

## 目标
围绕研究主题检索 arXiv 和 Semantic Scholar 文献，完成研究空白分析，并提出新的研究假设。

## 所需工具
- `search_arxiv`：检索 arXiv 论文
- `search_semantic_scholar`：检索 Semantic Scholar 论文与引用信息

## 输入
- `topic`：需要研究的主题或问题

## 处理流程
1. 根据主题生成 5 到 8 个多样化检索词
2. 使用每个检索词检索 arXiv 和 Semantic Scholar
3. 对论文结果去重，并按相关性排序
4. 分析收集到的论文，识别研究空白
5. 生成 2 到 4 个用于回应这些空白的新假设
6. 选择最有前景的假设，并给出理由

## 输出
产出 `papers/ideation_output.json`，内容包括：
- 带元数据的检索论文列表
- 文献综述摘要
- 研究空白分析
- 生成的假设
- 最终选中的假设及其理由
