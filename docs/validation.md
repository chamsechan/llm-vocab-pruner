# 验证报告

[返回 README](../README.md)

## 自动化测试

运行：

```bash
python3 -m unittest -v \
  tests/test_asymmetric_models.py \
  tests/test_prune_model_by_txt.py
```

当前 22 项测试覆盖：

- Qwen2、Qwen3、Qwen3.5 与 Gemma 3 转换；
- tokenizer 与 Embedding 同步重排；
- LM Head 行选择与权重解除共享；
- 紧凑 logits 形状；
- 保留 logits 等价性；
- repetition penalty、no-repeat n-gram 和 bad words；
- 必要生成 Token 保护；
- 训练 label 范围；
- `save_pretrained()` 与对应 AutoModel 类独立重载；
- Qwen3 0.6B、Qwen3.5 0.8B 与 Gemma 3 270M 生产结构形状；
- tokenizer-only 越界 Token 处理。

## 真实权重验证

真实权重验证使用最小保留集合：基础 Token、必要特殊 Token 和基准生成 Token。该大小只用于严格验证重排等价性，不是生产推荐词表大小。

| 模型 | 完整 Embedding | 测试 LM Head | 结果 |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | `(151936, 896)` | `(275, 896)` | 通过 |
| Qwen3-0.6B | `(151936, 1024)` | `(275, 1024)` | 通过 |
| Qwen3.5-0.8B（文本） | `(248320, 1024)` | `(270, 1024)` | 通过 |
| Gemma 3 270M IT | `(262144, 640)` | `(266, 640)` | 通过 |

每个模型均执行以下链路：

```text
真实源权重
→ 构造保留集合
→ tokenizer/Embedding 同步重排
→ LM Head 裁剪
→ save_pretrained
→ 新 Python 进程运行 inference_demo.py
→ 比较原模型与导出模型结果
```

## Qwen2.5-0.5B-Instruct

验证内容：

- 输入词表：`151936`；
- 测试输出词表：`275`；
- 移除输出行：`151661`；
- 同一提示词对应 Embedding 向量逐元素一致；
- 保留 Token logits 逐元素一致；
- `tie_word_embeddings=false`；
- LM Head 与输入 Embedding 不共享 Parameter；
- 模型和配置中不存在 `output_token_ids`；
- 独立 Demo 退出码为 0。

测试提示词：

```text
Translate into Chinese: Artificial intelligence is changing the world.
```

原模型与导出模型均生成：

```text
人工智能正在改变世界。
```

## Qwen3-0.6B

验证内容：

- 官方模型：`Qwen/Qwen3-0.6B`；
- 输入词表：`151936`；
- 测试输出词表：`275`；
- 移除输出行：`151661`；
- 导出类型：`AsymmetricQwen3ForCausalLM`；
- 同一提示词对应 Embedding 向量逐元素一致；
- 保留 Token logits 逐元素一致；
- `tie_word_embeddings=false`，LM Head 与输入 Embedding 不共享 Parameter；
- 模型和配置中不存在 `output_token_ids`；
- 保存后由新 Python 进程执行 `inference_demo.py --disable-thinking`，退出码为 0；
- 原模型、内存中转换模型与独立 Demo 生成文本逐字一致。

测试提示词：

```text
Translate into Chinese: Artificial intelligence is changing the world.
```

三个执行路径均生成：

```text
人工智能正在改变世界。
```

## Qwen3.5-0.8B（文本输入路径）

验证内容：

- 官方模型：`Qwen/Qwen3.5-0.8B`；
- 官方类型：`Qwen3_5ForConditionalGeneration`；
- 输入词表：`248320`；
- 测试输出词表：`270`；
- 移除输出行：`248050`；
- 导出类型：`AsymmetricQwen3_5ForConditionalGeneration`；
- tokenizer ID、Embedding 向量与保留 Token logits 等价；
- image、video、vision start/end 特殊 ID 使用同一排列同步更新；
- 顶层与 `text_config` 均为 `tie_word_embeddings=false`；
- LM Head 与输入 Embedding 不共享 Parameter，且不存在运行时 ID 映射；
- 保存后由新 Python 进程通过 `AutoModelForImageTextToText` 加载，独立 Demo 退出码为 0；
- 原模型、内存中转换模型与独立 Demo 生成文本逐字一致。

测试提示词：

```text
Translate into Chinese: Artificial intelligence is changing the world.
```

三个执行路径均生成：

```text
人工智能正在改变世界。
```

本次仅验证纯文本输入与文本生成。没有保存或验证 `AutoProcessor`，也没有执行图像/视频输入，因此不属于完整多模态支持。

## Gemma 3 270M IT

验证内容：

- 输入词表：`262144`；
- 测试输出词表：`266`；
- 移除输出行：`261878`；
- 同一提示词对应 Embedding 向量逐元素一致；
- 保留 Token logits 逐元素一致；
- `tie_word_embeddings=false`；
- LM Head 与输入 Embedding 不共享 Parameter；
- `<image_soft_token>` tokenizer-only ID 保持为 `262144`；
- 独立 Demo 退出码为 0；
- 原模型与导出模型生成文本逐字一致。

Gemma 3 270M IT 的翻译质量属于模型能力，不属于词表重排机制验证范围。验证目标是确认转换前后在保留 Token 上保持一致。

## 验收标准

一次导出被视为结构正确，必须同时满足：

```text
config.vocab_reordered == true
config.tie_word_embeddings == false
Embedding rows == config.vocab_size
LM Head rows == config.output_vocab_size
Embedding.weight and LM Head.weight are different Parameters
logits.shape[-1] == config.output_vocab_size
```

此外，导出 tokenizer 必须能够独立重载，并对测试文本保持相同 Token 字符串与解码结果。
