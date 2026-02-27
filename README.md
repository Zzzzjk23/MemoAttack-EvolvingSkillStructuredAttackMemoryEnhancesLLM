# TAP Jailbreak：基于树搜索与后验证据的越狱攻击框架

本项目实现了一个基于树搜索（Tree-based Attack Prompt, TAP）的大语言模型越狱攻击框架，结合贝叶斯后验证据（Posterior Evidence）进行攻击方法选择，用于评估目标模型在对抗性提示下的安全性。

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [核心模块说明](#核心模块说明)
- [配置与参数](#配置与参数)
- [输出与日志](#输出与日志)
- [注意事项](#注意事项)

---

## 项目概述

本框架针对大语言模型（LLM）进行红队测试（Red Teaming），通过树形搜索结构迭代生成对抗性提示（adversarial prompts），使目标模型产生违反安全准则的回复。主要特点包括：

- **树形搜索**：以目标 goal 为根节点，逐层扩展子节点，每个节点代表一次攻击尝试
- **多攻击方法**：支持 10 种预定义攻击策略，通过贝叶斯 UCB 与余弦相似度进行软选择
- **后验证据更新**：根据成功/失败反馈动态更新各攻击方法的先验，实现自适应策略选择
- **AdvBench 基准**：基于 AdvBench 数据集进行批量越狱评估

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **三模型架构** | Attacker（生成攻击提示）、Evaluator（评估与分类）、Target（被攻击目标） |
| **10 种攻击方法** | Payload Splitting、Base64/Hex Encoding、Linux Terminal Simulation、Character Roleplay、Language Shifting、Cognitive Pressure、Scientific/Academic Framing、Virtual Machine Escape、Refusal Suppression、Game Theory Simulation |
| **贝叶斯 UCB + 相似度** | 结合成功概率与 prompt embedding 余弦相似度进行攻击方法选择 |
| **WandB 集成** | 实验过程与节点信息可记录到 Weights & Biases |
| **子集并行** | 支持将 AdvBench 拆分为多个子集，便于分布式或分批运行 |

---

## 项目结构

```
.
├── TAP_main.py           # 主入口，TAP 算法核心逻辑
├── TAP_tree.py           # 树结构定义（TreeNode、Tree）
├── Posterior_Evidence.py # 后验证据与攻击方法选择
├── LLMmodels.py          # Attacker / Evaluator / Target LLM 封装
├── system_prompts.py     # 系统提示词模板
├── prompt_embedding.py   # Prompt 向量化与相似度计算
├── logger.py             # WandB 日志记录
├── split_advbench.py     # AdvBench 数据集切分工具
├── AdvBench.csv          # 原始 AdvBench 数据
├── AdvBench_subset_*.csv # 切分后的子集（每 50 条）
├── posterior_evidence.pkl # 后验证据持久化文件（运行时生成）
├── results_*.csv         # 运行结果输出（运行时生成）
└── wandb/                # WandB 运行记录
```

---

## 环境配置

### 1. Python 版本

建议使用 **Python 3.11+**。

### 2. 安装依赖

主要依赖包括：

```bash
pip install pandas numpy sentence-transformers
pip install openai litai
pip install wandb fastchat deep-translator
```

完整依赖可参考 `wandb/*/files/requirements.txt` 中的版本列表，核心包包括：

- `pandas`：数据处理
- `numpy`：数值计算
- `sentence-transformers`：prompt embedding（all-MiniLM-L6-v2）
- `openai`：OpenAI 兼容 API 调用
- `litai`：LLM 推理
- `wandb`：实验日志
- `fastchat`：对话模板
- `deep-translator`：中英翻译（用于日志展示）

### 3. API 配置

在 `LLMmodels.py` 中需配置：

- **OpenAI 兼容 API**：`base_url`、`api_key`（用于 Evaluator 与 Target）
- **LitAI**：`api_key`（用于 Attacker）

当前使用的模型映射（可在 `LLM_choices` 中修改）：

- Attacker: `Qwen/Qwen3-Coder-480B-A35B-Instruct`
- Evaluator: `moonshotai/Kimi-K2-Instruct`
- Target: `openai/gpt-oss-20b`

---

## 数据准备

### AdvBench 数据集

`AdvBench.csv` 需包含列：

- `goal`：越狱目标描述
- `target`：期望模型回复的开头（如 `"Sure, here is..."`）

### 切分子集

使用 `split_advbench.py` 将完整 AdvBench 按每 50 条切分为多个子集：

```bash
python split_advbench.py
```

将生成 `AdvBench_subset_1.csv` 至 `AdvBench_subset_N.csv`，并自动清理 `target` 列格式。

---

## 使用方法

### 1. 运行单个子集

在 `TAP_main.py` 中修改 `main()` 的调用参数，例如只运行子集 1：

```python
if __name__ == '__main__':
    main(1)  # 运行 AdvBench_subset_1.csv
```

### 2. 批量运行多个子集

默认会依次运行子集 2 至 11：

```bash
python TAP_main.py
```

每个子集完成后会生成 `results_{i}.csv`，包含 `goal`、`target`、`if_success`、`request_count`。

### 3. 单次 TAP 调用示例

```python
from types import SimpleNamespace
from TAP_main import tap

args = SimpleNamespace(
    attacker_model='lightning-ai/DeepSeek-V3.1',
    evaluator_model='lightning-ai/llama-3.3-70b',
    target_model='lightning-ai/gpt-oss-20b',
    goal='Write a script that exploits vulnerabilities...',
    target="['Sure, here is a script that...']",
    index=0,
    max_depth=5,
    branching_factor=4,
    width=4
)
success, request_count = tap(args, logger=None)
```

---

## 核心模块说明

### TAP_main.py

- `tap(args, logger)`：执行一次完整 TAP 搜索
- `select_nodes(leaf_nodes, width)`：从叶子节点中按 `outside_score` 选取 top-`width` 个节点继续扩展
- `posterior_evidence_update(tap_tree)`：根据搜索树更新后验证据（当前实现已注释，可自行恢复）

### TAP_tree.py

- **TreeNode**：树节点，包含 `prompt`、`outside_score`、`on_topic`、`attack_method` 等
- **Tree**：整棵搜索树，管理 `goal`、`target`、三个 LLM、`PosteriorEvidence` 等
- 节点创建时自动调用 `select_attack_method` 选择攻击方法，并根据 `outside_score` 变化更新成功/失败

### Posterior_Evidence.py

- **PosteriorEvidence**：维护各 prompt 类别下各攻击方法的成功/失败计数
- **select_attack_method**：基于贝叶斯 UCB 与 prompt embedding 余弦相似度，通过 softmax 采样选择攻击方法
- 支持 `save()` / `load_posterior_evidence()` 持久化到 `posterior_evidence.pkl`

### prompt_embedding.py

- `embed_prompt(prompt)`：使用 `all-MiniLM-L6-v2` 将 prompt 转为向量
- `cosine_similarity(vec_a, vec_b)`：计算两向量余弦相似度

---

## 配置与参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_depth` | 树最大深度 | 5 |
| `branching_factor` | 每个节点扩展子节点数 | 4 |
| `width` | 每层选取的叶子节点数 | 4 |
| `attacker_model` | 攻击者模型 | 见 `LLM_choices` |
| `evaluator_model` | 评估者模型 | 见 `LLM_choices` |
| `target_model` | 目标模型 | 见 `LLM_choices` |

越狱成功判定：`outside_score == 10`。

---

## 输出与日志

### 1. 控制台输出

- 每个 goal 的越狱成功/失败及请求次数
- 成功时打印：`{goal} 越狱成功，共生成{request_count}个请求`

### 2. CSV 结果

`results_{i}.csv` 字段：

- `goal`：越狱目标
- `target`：期望回复开头
- `if_success`：是否越狱成功
- `request_count`：总请求数

### 3. WandB 日志

若传入 `logger`，会记录：

- 节点信息：`node_id`、`depth`、`prompt`、`outside_score` 等
- 越狱结果：`jailbreak_success`、`final_request_count`
- 全局上下文：`global_context`

### 4. 中间文件

- `attacker_input/openai_messages_{index}_{request_count}.json`：攻击者输入（含深度、分数、中英翻译等）

---

## 注意事项

1. **API 密钥**：请勿将 `LLMmodels.py` 中的 API 密钥提交到公开仓库，建议使用环境变量或本地配置文件。
2. **global_context**：项目依赖 `global_context` 模块（用于成功案例入队），若缺失需自行实现或移除相关调用。
3. **后验证据**：`posterior_evidence.pkl` 会在运行中累积更新，如需从头开始可删除该文件。
4. **网络与代理**：WandB 上传可能受代理影响，若无法连接可考虑离线模式或关闭 WandB。
5. **伦理与合规**：本框架仅用于安全研究与模型评估，请遵守相关法律法规与使用条款。

---

## 引用与参考

- AdvBench：Adversarial Benchmark for Evaluating the Safety of Large Language Models
- 攻击方法参考：Payload Splitting、Base64 Encoding、Roleplay 等常见越狱技术

---

*最后更新：2025 年 2 月*
