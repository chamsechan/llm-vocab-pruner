# LLM Reordered Vocabulary Pruner

<p align="center"><strong>保留完整多语种输入词表，将允许输出的 Token 重排为连续前缀，只计算紧凑 LM Head。</strong></p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.0+" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/docs/transformers/"><img alt="Transformers 4.55+" src="https://img.shields.io/badge/Transformers-4.55+-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-22C55E"></a>
</p>

面向“**多语种输入、有限目标语言输出**”的词表裁剪工具。典型场景是翻译模型需要理解很多源语言，但只生成中文、数字、单位、专名和少量符号。

项目不会删除任何输入 Embedding 行，而是在导出时同步重排 tokenizer Token ID 与完整 Embedding：允许输出的 Token 位于连续前缀 `[0, V_output)`，其他仅输入 Token 位于尾部。LM Head 只保留前缀，因此推理时直接产生新 Token ID，不需要映射表、scatter 或完整词表 logits。

<p align="center"><a href="assets/asymmetric-vocab-flow.svg"><img src="assets/asymmetric-vocab-flow.svg" width="100%" alt="Tokenizer 与 Embedding 同步重排动画"></a></p>
<p align="center"><sub>动画展示导出时的一次性行重排，以及导出模型运行时的无映射紧凑输出路径。</sub></p>

## 核心特性

- **完整输入词表**：所有原始 Embedding 向量都保留，因此多语种输入能力不因输出裁剪而丢失。
- **同步重排**：tokenizer ID 和 Embedding 行使用同一个排列，Token 与向量不会错位。
- **连续输出前缀**：允许输出 Token 的新 ID 固定为 `0..V_output-1`。
- **无运行时映射**：LM Head 后没有 `output_token_ids`、scatter 或 `-∞` 填充。
- **紧凑 logits**：模型直接返回 `[..., V_output]`，不再临时创建 `[..., V_input]`。
- **解除权重共享**：导出配置强制 `tie_word_embeddings=false`，LM Head 是独立 Parameter。
- **独立加载**：导出目录包含运行时模型定义，可通过 Transformers `AutoModelForCausalLM` 重载。
- **已验证架构**：Qwen2/Qwen2.5 与纯文本 Gemma 3；实现方式可扩展到其他 decoder-only 模型。

## 支持范围

| 架构 | 状态 | 验证内容 |
|---|---|---|
| Qwen2 / Qwen2.5 | 支持 | tokenizer/Embedding 重排、紧凑生成、保存与独立重载 |
| Gemma 3 text | 支持 | tokenizer/Embedding 重排、紧凑生成、保存与独立重载 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 真实权重验证 | 向量与 logits 一致、无映射生成、保存重载文本一致 |
| `google/gemma-3-270m-it` | 真实权重验证 | 向量与 logits 一致、图像占位 ID 处理、保存重载文本一致 |
| Gemma 3 multimodal | 暂不支持 | `Gemma3ForConditionalGeneration` 不在当前适配范围 |
| 其他 decoder-only 模型 | 可扩展 | 需要增加并测试对应模型适配器 |

## 原理

### 1. 构造一次性排列

设完整输入词表大小为 `V_input`，允许输出的旧 Token ID 集合为：

```text
K_old = [2, 8, 11, ...]
```

导出器构造完整排列 `P`：

```text
P = [所有保留 ID, 所有其余 ID]
    └── V_output ──┘└── 仅输入 Token ──┘
```

其中 `P[new_id] = old_id`。这份排列只在导出时使用，不会作为运行时映射表保存到模型中。

### 2. tokenizer 与 Embedding 同步重排

对于每个新 ID `j`：

```text
new_tokenizer_id(token_at_old_id=P[j]) = j
new_embedding[j]                       = old_embedding[P[j]]
```

因此任意 Token `t` 的向量保持不变：

```text
new_embedding[new_tokenizer(t)] = old_embedding[old_tokenizer(t)]
```

Embedding 的形状仍为 `[V_input, H]`，只是行顺序改变。所有被禁止输出的外语 Token 仍然位于 Embedding 尾部，可以正常出现在输入中。

### 3. LM Head 直接对应输出前缀

LM Head 只复制允许输出的旧权重行：

```text
new_lm_head[j] = old_lm_head[K_old[j]],  0 <= j < V_output
```

转换后：

```text
Embedding  [V_input,  H]   # 完整、已重排
LM Head    [V_output, H]   # 独立、紧凑
logits     [..., V_output] # 直接使用新 tokenizer ID
```

模型生成的新 ID 天然落在 `0..V_output-1`，可以直接交给新 tokenizer 解码，也可以直接反馈给完整 Embedding。LM Head 后没有映射步骤。

### 4. 为什么结果等价

对于保留 Token 的新 ID `j`：

```text
logit_new(j) = logit_old(P[j])
```

如果原模型的每一步贪心结果都在保留集合中，则：

- Transformer 输入向量逐元素一致；
- hidden states 一致；
- 保留 Token logits 一致；
- 生成 Token 字符串和最终文本一致；
- 内部整数 ID 不同，因为词表已经重新编号。

如果原模型原本要生成一个被删除 Token，新模型会选择输出前缀中的下一个候选，这正是输出裁剪的预期行为。

## 生成兼容性

输入序列可以包含 `>= V_output` 的仅输入 ID，但部分 Transformers logits processor 默认假设输入 ID 小于 logits 宽度。本项目为紧凑输出实现了兼容处理：

- repetition penalty 忽略没有输出 logit 的仅输入 ID；
- no-repeat n-gram 忽略无法输出的 banned next ID；
- bad words / sequence bias 允许前缀中出现仅输入 ID，并忽略不可输出的最终 ID；
- BOS、EOS、PAD 和 forced generation Token 必须保留，否则转换直接报错。

外部代码传入的自定义 logits processor 也必须遵守 `scores.shape[-1] == V_output`。旧 ID 编写的 bad-words、forced-token 等外部配置，应当通过新 tokenizer 重新取得 ID。

## Tokenizer 安全与兼容边界

1. 当前重排要求 Hugging Face **fast tokenizer**。
2. 分词模型、merge 规则、normalizer、pre-tokenizer 和 decoder 不变，只改变 Token ID。
3. Qwen 的 added/special tokens 会被同步纳入新 ID 排列并保留其特殊匹配行为。
4. Gemma 3 的 `<image_soft_token>` ID 为 `262144`，而纯文本 Embedding 只有 `0..262143`。该 tokenizer-only ID 保持为 `262144`，不会进入 LM Head 或越界切片。
5. Gemma 导出物只保存正确的重排 `tokenizer.json`，不会附带仍使用旧 ID 的 `tokenizer.model`；请使用默认 fast tokenizer，不要指定 `use_fast=False`。Transformers 4.57.x 若对本地非 Mistral tokenizer 产生 regex 误报告警，可显式传入 `fix_mistral_regex=False`；独立 Demo 已这样处理。
6. 旧的预分词 `input_ids`、Token ID 缓存或按旧 ID 编写的业务配置不再兼容，必须用导出的 tokenizer 重新编码。

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### Qwen2.5

先生成并人工审核待禁止输出 Token 清单：

```bash
python3 export_delete_tokens.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output qwen-delete-tokens.txt
```

执行重排、LM Head 裁剪、保存重载与生成对比：

```bash
python3 prune_model_by_txt.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --delete_txt qwen-delete-tokens.txt \
  --output ./qwen2.5-reordered-pruned
```

### Gemma 3 270M

需要先在 Hugging Face 接受 Google 模型许可并完成登录：

```bash
python3 export_delete_tokens.py \
  --model google/gemma-3-270m-it \
  --output gemma3-delete-tokens.txt

python3 prune_model_by_txt.py \
  --model google/gemma-3-270m-it \
  --delete_txt gemma3-delete-tokens.txt \
  --output ./gemma3-270m-reordered-pruned
```

`delete_tokens.txt` 中记录的是**原模型旧 ID**。脚本会计算保留集合，然后统一生成新 tokenizer 和重排后的 Embedding。启发式清单必须人工审核，确保目标语言、数字、单位、专名和业务符号完整。

## 独立推理 Demo

```bash
python3 inference_demo.py \
  --model ./gemma3-270m-reordered-pruned \
  --prompt "Translate into Chinese: Hello world."
```

或直接加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./gemma3-270m-reordered-pruned"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, fix_mistral_regex=False
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)

assert model.config.vocab_reordered is True
assert model.config.tie_word_embeddings is False
assert not hasattr(model.config, "output_token_ids")
assert model.get_output_embeddings().weight.shape[0] == model.config.output_vocab_size

inputs = tokenizer("将法语翻译成中文：Bonjour le monde.", return_tensors="pt")
inputs = inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`inference_demo.py` 会在生成前验证 `vocab_reordered=true`、`tie_word_embeddings=false`，并确认模型不包含旧的运行时 ID 映射。

## 导出目录

```text
gemma3-270m-reordered-pruned/
├── asymmetric_models.py      # 紧凑 LM Head 运行时模型定义
├── config.json               # vocab_size、output_vocab_size、tie=false
├── generation_config.json    # 已同步更新的特殊 Token ID
├── model.safetensors         # 重排完整 Embedding + 紧凑 LM Head
├── tokenizer.json            # 重排后的 fast tokenizer
├── tokenizer_config.json
└── ...
```

关键配置：

```json
{
  "vocab_size": 262144,
  "output_vocab_size": 266,
  "vocab_reordered": true,
  "tie_word_embeddings": false
}
```

导出配置中没有 `output_token_ids`。`asymmetric_models.py` 仍然必要，因为标准模型类通常使用同一个 `vocab_size` 创建 Embedding 和 LM Head，而这里两者行数不同。`save_pretrained()` 会复制该运行时代码并写入 `auto_map`；因此必须移动完整导出目录，并只对可信目录启用 `trust_remote_code=True`。

## 参数与内存

| 模块 | 参数量 |
|---|---:|
| 完整输入 Embedding | `V_input × H` |
| 紧凑 LM Head | `V_output × H` |
| LM Head 减少量 | `(V_input - V_output) × H` |
| 运行时 logits | `batch × sequence × V_output` |

注意：

- 原本共享 Embedding/LM Head 的模型在解除共享后会新增一份紧凑 LM Head，因此模型总参数量是“原共享模型 + `V_output × H`”，但不会再常驻完整输出矩阵。
- 完整 Embedding 会在导出时被重写一次；导出后的重排权重可继续用于 SSD mmap/offload。
- 本项目不实现 SSD mmap/offload，只保证导出的 Embedding 布局适合直接按新 tokenizer ID 索引。
- 若用于训练，label 必须位于输出前缀；仅输入 Token 的 label 应设为 `-100`，否则模型会明确报错。

## 验证结果

真实权重验证使用最小保留集合（基础 Token、必要特殊 Token 与基准生成 Token），目的是严格检查重排等价性，而不是推荐生产词表大小。

| 模型 | Embedding | 测试 LM Head | 验证结果 |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | `(151936, 896)` | `(275, 896)` | 输入向量一致、保留 logits 一致、重载生成文本一致 |
| Gemma 3 270M IT | `(262144, 640)` | `(266, 640)` | 输入向量一致、保留 logits 一致、重载生成文本一致 |

运行单元测试：

```bash
python3 -m unittest -v \
  tests/test_asymmetric_models.py \
  tests/test_prune_model_by_txt.py
```

测试覆盖 Qwen2 与 Gemma 3 的同步重排、紧凑 logits、特殊 ID、生成处理器、权重非共享、训练 label 边界、保存重载、Gemma 3 270M 生产形状和 tokenizer 越界 Token。

## 项目结构

```text
.
├── export_delete_tokens.py       # 生成并导出旧 ID 禁止清单
├── prune_model_by_txt.py         # 重排、裁剪、导出、重载与生成验证
├── asymmetric_models.py          # tokenizer 重排与运行时模型适配器
├── inference_demo.py             # 独立推理 Demo
├── assets/
│   └── asymmetric-vocab-flow.svg # 一次性重排与无映射推理动画
└── tests/
```

## License

[MIT](LICENSE)
