# 词表重排与输出裁剪原理

[返回 README](../README.md)

## 目标

模型需要理解完整多语种输入，但只允许生成有限的目标语言 Token。设：

- `V_input`：完整输入词表大小；
- `V_output`：允许输出的 Token 数量；
- `H`：hidden size；
- `K_old`：允许输出 Token 在原模型中的 ID 集合。

导出结果保持完整输入 Embedding `[V_input, H]`，同时把 LM Head 缩小为 `[V_output, H]`。

## 一次性排列

假设保留的旧 ID 为：

```text
K_old = [2, 8, 11, ...]
```

导出器构造完整排列：

```text
P = [所有保留 ID, 所有其余 ID]
    └── V_output ──┘└── 仅输入 Token ──┘
```

其中：

```text
P[new_id] = old_id
```

`P` 只用于导出时重写 tokenizer 和权重，不保存在运行时模型中。

## tokenizer 与 Embedding 同步重排

对每个新 ID `j`：

```text
new_tokenizer_id(token_at_old_id=P[j]) = j
new_embedding[j]                       = old_embedding[P[j]]
```

因此任意 Token `t` 都满足：

```text
new_embedding[new_tokenizer(t)] = old_embedding[old_tokenizer(t)]
```

Token 的整数 ID 改变了，但实际查到的向量保持不变。禁止输出的外语 Token 被移动到尾部，仍可作为输入正常使用。

## 紧凑 LM Head

LM Head 解除与输入 Embedding 的共享，只复制保留行：

```text
new_lm_head[j] = old_lm_head[K_old[j]]
0 <= j < V_output
```

导出后：

```text
Embedding  [V_input,  H]   # 完整、已重排
LM Head    [V_output, H]   # 独立、紧凑
logits     [..., V_output] # 直接对应新 tokenizer ID
```

生成 ID 天然位于 `0..V_output-1`，可以直接解码或反馈给输入 Embedding。因此 LM Head 后不存在映射表、scatter、完整 logits 或被删除位置的 `-∞` 填充。

## 等价性

对保留 Token 的新 ID `j`：

```text
logit_new(j) = logit_old(P[j])
```

原因是：

1. tokenizer 和 Embedding 使用同一个排列；
2. Transformer 接收到的输入向量逐元素一致；
3. Transformer 主体权重不变；
4. 新 LM Head 第 `j` 行等于原 LM Head 第 `P[j]` 行。

如果原模型贪心生成的每一步最佳 Token 都在保留集合中，那么：

- hidden states 一致；
- 保留 Token logits 一致；
- 生成 Token 字符串和最终文本一致；
- 只有内部整数 ID 不同。

若原模型原本要生成被删除 Token，新模型会从保留前缀中选择下一个候选，这是输出裁剪的预期行为。

## 参数与内存

| 模块 | 参数量或张量形状 |
|---|---:|
| 完整输入 Embedding | `V_input × H` |
| 紧凑 LM Head | `V_output × H` |
| LM Head 减少量 | `(V_input - V_output) × H` |
| 运行时 logits | `batch × sequence × V_output` |

原模型如果共享 Embedding 与 LM Head，解除共享后会新增一份 `V_output × H` 的独立输出权重；但不再需要完整输出矩阵或完整词表 logits。

完整 Embedding 会在导出时重写一次。之后可直接从新权重文件加载或用于 SSD mmap/offload；本项目本身不实现 mmap/offload。

## 为什么仍需要自定义模型类

标准 Transformers causal-LM 通常使用同一个 `config.vocab_size` 同时创建输入 Embedding 和 LM Head。本项目需要：

```text
Embedding rows = V_input
LM Head rows   = V_output
```

因此导出目录包含 `asymmetric_models.py`，并通过 `config.json` 的 `auto_map` 注册运行时类：

```json
{
  "vocab_size": 262144,
  "output_vocab_size": 266,
  "vocab_reordered": true,
  "tie_word_embeddings": false
}
```

移动模型时必须复制完整导出目录。只应对可信目录启用 `trust_remote_code=True`。
