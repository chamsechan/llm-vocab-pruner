# ✂️ LLM Vocabulary & Embedding Pruner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow.svg)](https://huggingface.co/docs/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个面向“多语种输入、受限目标语言输出”的非对称词表裁剪工具。

完整保留原 tokenizer、多语种输入 `Embedding` 和原 Token ID，仅将输出 `LM Head` 裁剪为目标语言词表。紧凑 LM Head 的 logits 会被放回原 Token ID 位置，未保留 Token 的 logits 为负无穷，因此输入能力不变而输出被严格限制。

---

## 🌟 核心亮点

- ⚡ **输出端瘦身**：LM Head 的常驻权重只包含允许生成的 Token，适合将完整输入 Embedding 另行 mmap/offload 的部署结构。
- 🎯 **完整多语种输入**：Transformer 主干、tokenizer 与输入 Embedding 全部原样保留。
- 🔍 **2-Step 显式人机协同**：导出可读 `delete_tokens.txt` 列表 $\rightarrow$ 支持人工审查 $\rightarrow$ 一键切片导出并同屏对比验证。
- 🚀 **独立加载**：导出目录自带模型实现，可由 `AutoModelForCausalLM` 在新进程中直接加载。

---

## 📊 参数关系

若保留输出 Token 数为 `V_out`，隐藏维度为 `H`，独立 LM Head 的参数量为 `V_out × H`；输入 Embedding 仍为 `V_input × H`。

> Qwen2.5 默认共享 Embedding/LM Head。解除共享并保留完整 Embedding 后，导出的磁盘总参数量可能比原模型更大，因为新增了一份紧凑 LM Head。只有在部署时将完整输入 Embedding mmap/offload、不作为常驻内存计算，才能获得目标场景中的常驻内存收益。本项目当前不实现该 offload 层。

---

## 🚀 快速开始 (Quick Start)

所有脚本均内置默认参数，开箱即用：

```bash
pip install -r requirements.txt
```

### 1. 导出待删除 Token 清单
分析模型词表并将禁止在输出端生成的 Token 写入 `delete_tokens.txt`：
```bash
python3 export_delete_tokens.py
```

### 2. 人工审核（可选）
用文本编辑器打开 `delete_tokens.txt`，可自由删减或微调保留项。

### 3. 一键裁剪导出并同屏对比
读取 TXT 清单，仅切片 LM Head，导出新模型，并同屏对比打印**裁剪前/后的生成文本与速度**：
```bash
python3 prune_model_by_txt.py
```

### 4. 原生加载新模型
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./qwen2.5-pruned-by-txt"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

inputs = tokenizer("你好，请介绍一下你自己。", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

也可以直接运行独立推理 Demo（不依赖裁剪进程）：

```bash
python3 inference_demo.py --model ./qwen2.5-pruned-by-txt --prompt "Translate into Chinese: Hello world."
```

导出模型包含自定义 Qwen2 模型代码，加载时必须保留 `trust_remote_code=True`。当前非对称词表导出支持 Qwen2/Qwen2.5 架构。

---

## ⚠️ 避坑与安全机制

1. **Byte / 控制节点保护 (ID < 256)**：强制保留前 256 个基础字节与控制节点，防止 BPE 分词器崩溃。
2. **Special Tokens 100% 保护**：自动识别并保护 `<|im_start|>`, `<|im_end|>` 等系统标记。
3. **完整输入词表**：不修改 tokenizer 和 BPE merges，多语种输入不会在分词阶段丢失。

---

## 📄 开源协议 (License)

[MIT License](LICENSE)
