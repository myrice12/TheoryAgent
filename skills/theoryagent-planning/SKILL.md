---
name: theoryagent-planning
description: 根据研究假设产出实验蓝图
version: 0.1.0
---

# 规划技能

## 目标
读取 ideation 阶段选中的假设，生成详细实验蓝图，明确数据集、baseline、评测指标与消融分组。

## 所需工具
无。该技能主要依赖 LLM 对 ideation 输出进行推理。

## 输入
- `ideation_output`：由 ideation 技能生成的 `papers/ideation_output.json` 路径

## 处理流程
1. 从 ideation 输出中解析选中的假设及其支撑文献
2. 识别适合验证该假设的公开候选数据集
3. 从综述文献中选择 2 到 4 个 baseline 用于比较
4. 定义与假设对齐的主指标和次指标
5. 设计能隔离各创新组件影响的消融分组
6. 估算各实验所需算力与时间
7. 整理为结构化实验蓝图

## 输出
产出 `papers/experiment_blueprint.json`，内容包括：
- 继承下来的目标假设
- 数据集规格说明，如名称、来源、划分与预处理步骤
- 带参考文献的 baseline 方法
- 评测指标与成功标准
- 消融实验设计，如分组、变量与预期结果
- 资源估算与实验排期
