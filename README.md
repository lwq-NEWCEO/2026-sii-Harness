# 基于检索式 Few-shot 与输出约束的 Harness 设计

## 项目简介

本项目实现了一个智能文本分类系统，通过检索式 Few-shot 学习技术，结合输出约束机制，在有限的训练样本下实现了较高的分类准确率。

## 核心功能

- **检索式样本选择**：基于 TF-IDF 和字符 n-gram 相似度，动态选择最相关的训练样本构建提示
- **标签排序**：通过标签与查询文本的语义相似度进行预排序，提高分类准确性
- **输出约束与解析**：设计了严格的输出解析逻辑，确保模型返回的标签在允许范围内
- **防 Prompt 注入**：系统提示词设计了安全机制，防止恶意文本干扰分类结果

## 技术实现

### 主要模块

- **Harness 基类** (`harness_base.py`)：提供了核心接口，包括 `update()`（更新训练样本）和 `predict()`（预测标签）
- **MyHarness 类** (`solution.py`)：核心实现，包含：
  - 索引构建与相似度计算
  - 样本检索与排序
  - 提示词构建与预算控制
  - 输出解析与标签匹配

### 关键技术点

1. **混合相似度计算**：结合词级别 TF-IDF 相似度（74%）、字符 n-gram 相似度（18%）和标签提示相似度（8%）
2. **标签感知检索**：在检索样本时考虑标签与查询文本的匹配度
3. **动态提示裁剪**：根据 Token 预算自动调整提示长度，确保不超过限制
4. **鲁棒的标签解析**：多级匹配策略处理模型输出的各种变体

## 性能表现

在训练集上达到 **82.6%** 的准确率。

## 项目文件结构

```
student_package/
├── harness_base.py    # Harness 基类定义
├── solution.py        # MyHarness 核心实现
├── run.py             # 运行脚本
├── requirements.txt   # 依赖声明
├── data/              # 数据集目录
│   ├── train_dev.jsonl
│   └── test_dev.jsonl
└── tokenizer/         # 分词器配置
```

## 使用方法

1. 初始化 Harness：

```python
from solution import MyHarness
harness = MyHarness(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
```

2. 更新训练样本：

```python
harness.update(text, label)
```

3. 预测标签：

```python
predicted_label = harness.predict(text)
```

## 项目报告

详细的设计思路、技术细节和实验分析请参考：[基于检索式 Few-shot 与输出约束的 Harness 设计探索报告.pdf](./基于检索式 Few-shot 与输出约束的 Harness 设计探索报告.pdf)
