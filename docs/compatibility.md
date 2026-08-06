# 兼容性与使用边界

[返回 README](../README.md)

## 支持架构

| 架构 | 状态 |
|---|---|
| Qwen2 / Qwen2.5 | 支持 |
| Gemma 3 text / `Gemma3ForCausalLM` | 支持 |
| Gemma 3 multimodal / `Gemma3ForConditionalGeneration` | 暂不支持 |
| 其他 decoder-only 模型 | 需要增加适配器与测试 |

该实现是可扩展的 decoder-only 方案，不依赖特定目标语言；当前只在 Qwen2/Qwen2.5 和纯文本 Gemma 3 上完成验证。

## Fast tokenizer 要求

重排需要修改 tokenizer backend 中的词表 ID、added tokens 和 post-processor 特殊 ID，因此要求 Hugging Face fast tokenizer。

导出后：

- 必须使用导出目录中的 `tokenizer.json`；
- 不要指定 `use_fast=False`；
- 分词模型、merge、normalizer、pre-tokenizer 和 decoder 规则不变；
- 只改变 Token ID。

旧的预分词 `input_ids`、Token ID 缓存和按旧 ID 编写的外部配置不再兼容，必须使用导出的 tokenizer 重新生成。

## Qwen added tokens

Qwen 的部分 special/added tokens 不在基础 BPE vocab 中。重排时会把这些 Token 同步纳入新 ID 布局，同时保留 added-token 匹配和特殊 Token 语义。

## Gemma tokenizer-only Token

`google/gemma-3-270m-it` 的 `<image_soft_token>` ID 为 `262144`，纯文本 Embedding 行范围是 `0..262143`。该 Token 没有文本 Embedding 行，因此：

- 保持 tokenizer-only ID `262144`；
- 不进入 LM Head；
- 不参与 Embedding 切片或排列；
- 不会导致越界访问。

Gemma 导出目录只保存正确重排后的 `tokenizer.json`，不会附带仍使用旧 ID 的 `tokenizer.model`。

Transformers 4.57.x 如果对本地非 Mistral tokenizer 产生 regex 误报告警，可显式传入：

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=False,
)
```

## 必须保留的生成 Token

以下配置中的有效 Token ID 必须位于输出前缀，否则转换直接报错：

- BOS；
- EOS；
- PAD；
- decoder start；
- forced BOS/EOS。

模型配置、generation config、tokenizer special ID 和 post-processor special ID 会使用同一个排列同步更新。

## Transformers 生成处理器

输入序列允许包含 `ID >= V_output` 的仅输入 Token，但 Transformers 的部分 logits processor 默认假设所有输入 ID 都能索引输出 logits。本项目为紧凑输出实现了兼容处理：

- repetition penalty：忽略不存在输出 logit 的仅输入 ID；
- no-repeat n-gram：忽略无法输出的 banned next ID；
- bad words / sequence bias：允许前缀包含仅输入 ID，并忽略不可输出的最终 ID。

外部传入的自定义 logits processor 必须遵守：

```text
scores.shape[-1] == V_output
```

旧 ID 编写的 bad words、forced tokens、suppression 和 sequence bias 配置应通过新 tokenizer 重新取得 ID。

## 训练边界

模型可以计算 loss，但 label 必须位于输出前缀：

```text
0 <= label < V_output
```

仅输入 Token 不具备输出 logit。如它们出现在训练目标中，应将 label 设为 `-100`；否则模型会明确报错。

## 导出目录

```text
reordered-pruned-model/
├── asymmetric_models.py
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

必须复制整个目录。`asymmetric_models.py` 是运行时模型定义，不是一次性辅助文件。
