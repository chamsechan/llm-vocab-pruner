# LLM Asymmetric Output Vocabulary Pruner

<p align="center"><strong>保留完整多语种输入能力，只压缩模型允许生成的输出词表。</strong></p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.0+" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/docs/transformers/"><img alt="Transformers 4.55+" src="https://img.shields.io/badge/Transformers-4.55+-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-22C55E"></a>
</p>

面向“**多语种输入、受限目标语言输出**”场景的非对称词表裁剪工具。项目完整保留 tokenizer、原始 Token ID 和输入 Embedding，只将输出 LM Head 解除共享并裁剪为允许生成的 Token 集合。

> 典型场景：翻译模型需要理解多种源语言，但输出始终为中文、数字和有限符号。

<p align="center"><a href="assets/asymmetric-vocab-flow.svg"><img src="assets/asymmetric-vocab-flow.svg" width="100%" alt="非对称输出词表裁剪动画"></a></p>
<p align="center"><sub>若当前 Markdown 查看器不播放 SVG 动画，可点击图片查看；下文包含完整静态说明。</sub></p>

## 核心特性

- **输入能力不变**：不修改 tokenizer、分词规则、Token ID 或输入 Embedding。
- **输出权重紧凑**：LM Head 从 `[V_input, H]` 缩小为 `[V_output, H]`。
- **确定解除共享**：导出配置强制 `tie_word_embeddings=false`，两层使用不同 Parameter。
- **原生 ID 回环**：紧凑 logits 散射回原 Token ID，可继续使用 Transformers `generate()`。
- **独立模型目录**：导出物包含运行时模型定义，可由 `AutoModelForCausalLM` 独立加载。
- **结构可扩展**：当前验证 Qwen2/Qwen2.5 与纯文本 Gemma 3。

## 支持范围

| 架构 | 状态 | 验证内容 |
|---|---|---|
| Qwen2 / Qwen2.5 | 支持 | 转换、受限生成、保存、独立重载 |
| Gemma 3 text | 支持 | 转换、受限生成、保存、独立重载 |
| `google/gemma-3-270m-it` | 真实权重验证 | Embedding 不变、解除共享、LM Head 缩小、保留 Token 生成一致 |
| Gemma 3 multimodal | 暂不支持 | `Gemma3ForConditionalGeneration` 不在当前适配范围 |
| 其他 decoder-only 模型 | 可扩展 | 需要实现并测试对应适配器 |

## 为什么裁剪后仍能正常工作

### 1. 输入 Embedding 原样保留

原模型可能让 Embedding 和 LM Head 指向同一个 Parameter：

```text
Embedding.weight ───────────────┐
                                ├── shared Parameter
LM Head.weight ─────────────────┘
```

转换时不替换、不重排、不切片 Embedding。只从原输出权重中选择允许输出的行，并复制为独立 `nn.Linear`：

```python
compact_weight = source_weight[output_token_ids].clone()
new_lm_head.weight = nn.Parameter(compact_weight)
model.config.tie_word_embeddings = False
```

导出后：

```text
Embedding  [V_input,  H]   # 原对象、原尺寸、原权重
LM Head    [V_output, H]   # 独立 Parameter，V_output < V_input
tie_word_embeddings = false
```

### 2. Token ID 不压缩、不重排

`delete_tokens.txt` 只定义“禁止输出”的 Token，不修改 tokenizer。设保留的原 Token ID 为：

```text
K = [0, 1, 2, 100, 205, ...]
```

紧凑 LM Head 第 `j` 行对应原 Token `K[j]`：

```text
compact_logits[..., j]  →  original_token_id = K[j]
```

| 映射存储位置 | 用途 |
|---|---|
| `config.output_token_ids` | 独立加载时重建映射 |
| 模型 buffer `output_token_ids` | 随权重保存并跟随模型设备 |
| `config.output_vocab_size` | 记录紧凑 LM Head 的真实行数 |

### 3. 紧凑 logits 散射回原 ID

紧凑 LM Head 只计算 `V_output` 个 logits。为保持原 tokenizer 与 `generate()` 的 ID 语义，模型创建原词表大小的 logits，将保留结果写回对应 ID，其余位置设为负无穷：

```python
full_logits = compact_logits.new_full(
    (*compact_logits.shape[:-1], V_input),
    -torch.inf,
)
full_logits[..., output_token_ids] = compact_logits
```

```text
原 Token ID → 完整 Embedding → Transformer → 紧凑 LM Head
                                             ↓
                                      compact logits
                                             ↓ scatter
                          原词表 logits（其他位置为 -∞）
                                             ↓
                          generate() 产生原 Token ID
                                             ↓
                              直接反馈给完整 Embedding
```

因此无需压缩 ID 或修改 tokenizer，也不会出现“新 ID 被输入层解释成另一个旧 Token”的错位。

### 4. 正确性边界

对于任意保留 Token `k ∈ K`：

```text
logit_pruned(k) = logit_original(k)
```

被删除 Token 的 logit 为 `-∞`，因此不能生成。若原始贪心生成每一步的最佳 Token 都属于 `K`，裁剪前后会逐 Token 一致；若原模型想生成已删除 Token，结果会变为保留集合中的次优 Token。

## Token 安全处理

输出集合在切片前执行：

1. 仅接受 `0 <= token_id < Embedding.num_embeddings` 的 ID。
2. 有效 BOS、EOS、PAD 等 Special Token 必须保留；误删直接报错。
3. tokenizer 和 BPE/SentencePiece 保持原样，多语种输入不会丢失。
4. 前 256 个基础 Byte/控制 Token 受导出规则保护，便于输出数字、符号及字节回退内容。

Gemma 3 270M 的 tokenizer 包含 `<image_soft_token>`，ID 为 `262144`，但纯文本 Embedding 行号只有 `0..262143`。该 Token 没有文本权重行，脚本会报告并跳过，不会越界切片。

## `asymmetric_models.py` 是什么

[`asymmetric_models.py`](asymmetric_models.py) 是**导出模型运行时所需的模型定义**，不是一次性辅助脚本。

| 组件 | 职责 |
|---|---|
| `_AsymmetricOutputVocabMixin` | 将紧凑 logits 散射回原 Token ID |
| `AsymmetricQwen2ForCausalLM` | Qwen2/Qwen2.5 运行时适配器 |
| `AsymmetricGemma3ForCausalLM` | 纯文本 Gemma 3 运行时适配器 |
| `_SUPPORTED_MODEL_ADAPTERS` | 源模型类到非对称模型类的注册表 |
| `convert_to_asymmetric_output_vocab()` | 解除共享、复制 LM Head、保存映射和配置 |

`save_pretrained()` 会将该文件复制到导出目录，并在 `config.json.auto_map` 中记录运行时类。之后 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 会从导出目录加载它。

因此必须移动**整个导出目录**，不能只复制 `model.safetensors`。请只对可信的本地导出目录启用 `trust_remote_code=True`。

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### Qwen2.5

```bash
python3 export_delete_tokens.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output qwen-delete-tokens.txt

python3 prune_model_by_txt.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --delete_txt qwen-delete-tokens.txt \
  --output ./qwen2.5-output-pruned
```

### Gemma 3 270M

需要先在 Hugging Face 接受 Google 使用许可并完成登录。

```bash
python3 export_delete_tokens.py \
  --model google/gemma-3-270m-it \
  --output gemma3-270m-delete-tokens.txt

python3 prune_model_by_txt.py \
  --model google/gemma-3-270m-it \
  --delete_txt gemma3-270m-delete-tokens.txt \
  --output ./gemma3-270m-output-pruned
```

`delete_tokens.txt` 是启发式生成的输出禁用清单。生产使用前应人工审核，确保目标语言、数字、单位、专名和业务符号均被保留。

## 独立推理

```bash
python3 inference_demo.py \
  --model ./gemma3-270m-output-pruned \
  --prompt "Translate into Chinese: Hello world."
```

或直接加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./gemma3-270m-output-pruned"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)

assert model.config.tie_word_embeddings is False
assert (
    model.get_input_embeddings().weight.data_ptr()
    != model.get_output_embeddings().weight.data_ptr()
)

inputs = tokenizer("将法语翻译成中文：Bonjour le monde.", return_tensors="pt")
inputs = inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

裁剪脚本保存后会重载模型，并强制验证：

```text
tie_word_embeddings=false
Embedding 与 LM Head 不共享 Parameter
LM Head.shape[0] == config.output_vocab_size
```

任一条件不满足都会直接报错。

## 导出目录

```text
gemma3-270m-output-pruned/
├── asymmetric_models.py      # 运行时模型定义
├── config.json               # tie=false、auto_map、output_token_ids
├── generation_config.json
├── model.safetensors         # 完整 Embedding + 紧凑 LM Head
├── tokenizer.json            # 原 tokenizer，不裁剪
├── tokenizer_config.json
└── ...
```

## 参数与内存

| 模块 | 参数量 |
|---|---:|
| 完整输入 Embedding | `V_input × H` |
| 紧凑 LM Head | `V_output × H` |
| LM Head 减少量 | `(V_input - V_output) × H` |

重要限制：

- 原本已解除共享的模型，LM Head 参数会直接减少。
- 原本共享权重的模型，转换后新增一份紧凑 LM Head，磁盘总参数量可能增加 `V_output × H`。
- 常驻内存收益需要配合 Embedding mmap/offload；本项目当前不实现 SSD mmap。
- 紧凑 LM Head 减少输出矩阵乘法规模，但为兼容原生 `generate()`，当前仍创建 `[..., V_input]` 的完整 logits 张量。

## 验证

```bash
python3 -m unittest -v \
  tests/test_asymmetric_models.py \
  tests/test_prune_model_by_txt.py
```

测试覆盖转换、受限生成、`tie=false` 导出配置、权重非共享、独立重载、Token 映射、Gemma 3 270M 结构、越界图像 Token 和 Special Token 保护。

## 项目结构

```text
.
├── export_delete_tokens.py       # 生成输出禁用清单
├── prune_model_by_txt.py         # 裁剪、导出、重载及验证
├── asymmetric_models.py          # 导出模型运行时适配器
├── inference_demo.py             # 独立推理 Demo
├── assets/
│   └── asymmetric-vocab-flow.svg # 原理动画
└── tests/
```

## License

[MIT](LICENSE)
