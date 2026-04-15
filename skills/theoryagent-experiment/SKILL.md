---
name: theoryagent-experiment
description: 根据实验蓝图生成 Python 代码骨架
version: 0.1.0
---

# 实验技能

## 目标
读取实验蓝图，生成可运行的 Python 代码骨架，覆盖方法实现、baseline、训练循环、评测框架和消融配置。

## 所需工具
无。该技能主要依赖 LLM 基于实验蓝图生成代码。

## 输入
- `experiment_blueprint`：由 planning 技能生成的 `papers/experiment_blueprint.json` 路径

## 处理流程
1. 解析实验蓝图中的数据集、baseline、指标与消融分组
2. 生成项目目录结构，如数据加载、模型、训练、评测与配置
3. 为每个指定数据集生成数据加载与预处理代码
4. 为目标方法和各个 baseline 实现模型骨架
5. 生成包含日志、检查点与早停的训练循环
6. 实现覆盖全部指标的评测框架
7. 为每个消融分组生成对应配置文件
8. 添加主入口，能够读取配置并运行完整训练与评测流程

## 输出
产出 `experiments/` 目录，内容包括：
- `data/`：数据加载与预处理模块
- `models/`：模型结构实现，包括目标方法与 baseline
- `training/`：训练循环与优化工具
- `evaluation/`：指标计算与结果汇总
- `configs/`：各实验和消融变体的 YAML 配置
- `run.py`：实验启动主入口
- `requirements.txt`：Python 依赖列表
