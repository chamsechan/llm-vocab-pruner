#!/usr/bin/env python3
"""
步骤 2：读取 TXT 中的待删除 Token 列表，执行 Embedding 矩阵切片与模型裁剪导出
(Script 2: Read TXT Delete List and Prune Model & Tokenizer)

使用方法：
python3 prune_model_by_txt.py --model Qwen/Qwen2.5-0.5B-Instruct --delete_txt delete_tokens.txt --output ./qwen2.5-pruned-from-txt
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_delete_ids_from_txt(txt_path):
    """
    从 TXT 文件中读取待删除的 Token ID 集合
    """
    delete_ids = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            try:
                tid = int(parts[0].strip())
                delete_ids.add(tid)
            except ValueError:
                continue
    return delete_ids

def update_tokenizer_json(output_dir, old_to_new):
    """
    修改并保存新的 tokenizer.json，完成索引重排与 merges BPE 规则清洗
    """
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        print("提示: 未找到 tokenizer.json，跳过 JSON 重排清洗。")
        return

    print("正在更新 tokenizer.json 并清洗 BPE merges 规则...")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 更新 model.vocab
    if "model" in data and "vocab" in data["model"]:
        old_vocab = data["model"]["vocab"]
        new_vocab = {}
        for token_str, old_id in old_vocab.items():
            if old_id in old_to_new:
                new_vocab[token_str] = old_to_new[old_id]
        data["model"]["vocab"] = new_vocab

        # 2. 清洗 model.merges
        if "merges" in data["model"]:
            old_merges = data["model"]["merges"]
            new_merges = []
            for merge_item in old_merges:
                if isinstance(merge_item, str):
                    parts = merge_item.split(" ")
                elif isinstance(merge_item, list):
                    parts = merge_item
                else:
                    parts = []

                if len(parts) == 2:
                    merged_token = parts[0] + parts[1]
                    if merged_token in new_vocab or (isinstance(merge_item, str) and merge_item in new_vocab):
                        new_merges.append(merge_item)
            data["model"]["merges"] = new_merges

    # 3. 更新 added_tokens 列表
    if "added_tokens" in data:
        new_added_tokens = []
        for item in data["added_tokens"]:
            old_id = item["id"]
            if old_id in old_to_new:
                item["id"] = old_to_new[old_id]
                new_added_tokens.append(item)
        data["added_tokens"] = new_added_tokens

    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✓ tokenizer.json 词表与 merges 清洗完成！")

def prune_model_by_txt(model_name_or_path, delete_txt_path, output_dir):
    print(f"\n=======================================================")
    print(f" 步骤 2：读取 TXT 文件并执行模型与 Embedding 裁剪")
    print(f" 源模型: {model_name_or_path}")
    print(f" 删除清单文件: {delete_txt_path}")
    print(f" 导出路径: {output_dir}")
    print(f"=======================================================\n")

    # 1. 加载 Tokenizer 与 Model
    print("1. 正在加载 Tokenizer 与 Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    orig_vocab_size = model.config.vocab_size
    orig_params = sum(p.numel() for p in model.parameters())

    # 2. 读取 TXT 删除 ID 列表并计算保留 ID
    print(f"2. 正在解析 TXT 文件 '{delete_txt_path}'...")
    delete_ids = load_delete_ids_from_txt(delete_txt_path)
    all_vocab_ids = set(tokenizer.get_vocab().values())
    
    keep_old_ids = sorted(list(all_vocab_ids - delete_ids))
    new_vocab_size = len(keep_old_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_old_ids)}

    print(f" -> 原始词表大小: {orig_vocab_size}")
    print(f" -> TXT 删除指定 Token 数: {len(delete_ids)}")
    print(f" -> 实际保留 Token 数: {new_vocab_size}")

    # 3. 切片 Embedding 权重与 LM Head
    print("\n3. 重新构建 Embedding 矩阵与 LM Head...")
    input_embeds = model.get_input_embeddings()
    old_embed_weights = input_embeds.weight.data
    device = old_embed_weights.device

    keep_tensor = torch.tensor(keep_old_ids, dtype=torch.long, device=device)
    new_embed_weights = old_embed_weights[keep_tensor]
    hidden_dim = old_embed_weights.shape[1]

    # 设置新的 input embeddings
    model.set_input_embeddings(nn.Embedding(new_vocab_size, hidden_dim, _weight=new_embed_weights))

    # 设置新的 output embeddings (lm_head)
    output_embeds = model.get_output_embeddings()
    if output_embeds is not None:
        if getattr(model.config, "tie_word_embeddings", True):
            model.set_output_embeddings(model.get_input_embeddings())
        else:
            old_output_weights = output_embeds.weight.data
            new_output_weights = old_output_weights[keep_tensor]
            new_output_layer = nn.Linear(hidden_dim, new_vocab_size, bias=False)
            new_output_layer.weight = nn.Parameter(new_output_weights)
            model.set_output_embeddings(new_output_layer)

    model.config.vocab_size = new_vocab_size

    new_params = sum(p.numel() for p in model.parameters())
    print(f" -> 原始总参数量: {orig_params / 1e6:.2f} M")
    print(f" -> 裁剪后总参数量: {new_params / 1e6:.2f} M")
    print(f" -> 节省参数量: {(orig_params - new_params) / 1e6:.2f} M")

    # 4. 保存新模型
    print(f"\n4. 导出裁剪后的模型与 Tokenizer 至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 5. 更新并清洗 tokenizer.json
    update_tokenizer_json(output_dir, old_to_new)

    # 6. 原生验证推理
    print("\n5. 执行原生无映射推理验证...")
    try:
        test_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        test_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        test_prompt = "你好！请简要介绍一下中国长城。"
        messages = [
            {"role": "system", "content": "你是 AI 助手。"},
            {"role": "user", "content": test_prompt}
        ]
        text = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = test_tokenizer(text, return_tensors="pt").to(test_model.device)

        with torch.no_grad():
            outputs = test_model.generate(**inputs, max_new_tokens=48, do_sample=False)
            
        response = test_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n[原生验证模型回答结果]:")
        print(response)
        print("\n✓ 验证成功！基于 TXT 导出的模型可原生无障碍运行！")
    except Exception as e:
        print(f"\n[验证提示]: {e}")

def main():
    parser = argparse.ArgumentParser(description="步骤 2：读取 TXT 文件并切片模型 Embedding 执行导出")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称或本地路径")
    parser.add_argument("--delete_txt", type=str, default="delete_tokens.txt", help="要读取的删除列表 TXT 文件")
    parser.add_argument("--output", type=str, default="./pruned_model_by_txt", help="导出的新模型目录")
    args = parser.parse_args()

    prune_model_by_txt(args.model, args.delete_txt, args.output)

if __name__ == "__main__":
    main()
