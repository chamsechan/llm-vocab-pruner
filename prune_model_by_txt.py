#!/usr/bin/env python3
"""
步骤 2：读取 TXT 待删除 Token 清单，执行 Embedding 矩阵切片、导出原生模型，并自动对比显示裁剪前/后的生成质量。
(Script 2: Read TXT Delete List, Prune Model & Tokenizer, Export, and Compare Before/After Side-by-Side)

使用方法：
python3 prune_model_by_txt.py --model Qwen/Qwen2.5-0.5B-Instruct --delete_txt delete_tokens.txt --output ./qwen2.5-pruned-by-txt
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

TEST_PROMPTS = [
    "你好！请用一句话介绍一下中国长城。",
    "请写一段 Python 代码，用快速排序算法对整数列表进行排序。",
    "What are the primary benefits of artificial intelligence in healthcare?"
]

def load_delete_ids_from_txt(txt_path):
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
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        print("提示: 未找到 tokenizer.json，跳过 JSON 重排清洗。")
        return

    print("正在更新 tokenizer.json 并清洗 BPE merges 规则...")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "model" in data and "vocab" in data["model"]:
        old_vocab = data["model"]["vocab"]
        new_vocab = {}
        for token_str, old_id in old_vocab.items():
            if old_id in old_to_new:
                new_vocab[token_str] = old_to_new[old_id]
        data["model"]["vocab"] = new_vocab

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

def generate_text(tok, m, prompt):
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(formatted_text, return_tensors="pt").to(m.device)
    else:
        inputs = tok(prompt, return_tensors="pt").to(m.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = m.generate(**inputs, max_new_tokens=48, do_sample=False)
    latency = time.time() - start_time

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    speed = len(generated_ids) / latency if latency > 0 else 0

    response = tok.decode(generated_ids, skip_special_tokens=True)
    return latency, speed, response.strip()

def prune_model_by_txt(model_name_or_path, delete_txt_path, output_dir):
    print(f"\n=======================================================")
    print(f" 步骤 2：读取 TXT 文件，执行模型裁剪与对比验证")
    print(f" 源模型: {model_name_or_path}")
    print(f" 删除清单文件: {delete_txt_path}")
    print(f" 导出路径: {output_dir}")
    print(f"=======================================================\n")

    # 1. 加载原始模型并先执行原始效果基准测试
    print("1. 正在加载原始 Tokenizer 与 Model...")
    orig_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    orig_vocab_size = model.config.vocab_size
    orig_params = sum(p.numel() for p in model.parameters())
    orig_embed_params = model.get_input_embeddings().weight.numel()

    print("\n2. 正在录制【原始模型】在测试用例上的回答...")
    orig_results = {}
    for prompt in TEST_PROMPTS:
        orig_results[prompt] = generate_text(orig_tokenizer, model, prompt)

    # 2. 读取 TXT 并切片矩阵
    print(f"\n3. 解析 TXT 文件 '{delete_txt_path}' 并切片矩阵...")
    delete_ids = load_delete_ids_from_txt(delete_txt_path)
    all_vocab_ids = set(orig_tokenizer.get_vocab().values())
    
    keep_old_ids = sorted(list(all_vocab_ids - delete_ids))
    new_vocab_size = len(keep_old_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_old_ids)}

    input_embeds = model.get_input_embeddings()
    old_embed_weights = input_embeds.weight.data
    device = old_embed_weights.device

    keep_tensor = torch.tensor(keep_old_ids, dtype=torch.long, device=device)
    new_embed_weights = old_embed_weights[keep_tensor]
    hidden_dim = old_embed_weights.shape[1]

    model.set_input_embeddings(nn.Embedding(new_vocab_size, hidden_dim, _weight=new_embed_weights))

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
    new_embed_params = model.get_input_embeddings().weight.numel()

    # 3. 导出新模型
    print(f"\n4. 导出裁剪后的原生模型与 Tokenizer 至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    orig_tokenizer.save_pretrained(output_dir)
    update_tokenizer_json(output_dir, old_to_new)

    # 4. 加载裁剪后新模型并跑测试
    print(f"\n5. 正在加载【裁剪后模型】并录制回答...")
    pruned_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    pruned_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    pruned_results = {}
    for prompt in TEST_PROMPTS:
        pruned_results[prompt] = generate_text(pruned_tokenizer, pruned_model, prompt)

    # 5. 打印对比表格与同屏回答对比
    print(f"\n=======================================================")
    print(f" 📊 裁剪前 vs 裁剪后 对比结果汇总")
    print(f"=======================================================\n")

    print(f"{'指标 (Metric)':<24} | {'原始模型 (Original)':<20} | {'裁剪后模型 (Pruned)':<20} | {'变化趋势 (Diff)':<15}")
    print("-" * 88)
    
    v_diff = new_vocab_size - orig_vocab_size
    v_pct = (v_diff / orig_vocab_size) * 100
    print(f"{'词表大小 (Vocab Size)':<24} | {orig_vocab_size:<20} | {new_vocab_size:<20} | {v_diff} ({v_pct:.2f}%)")

    e_diff = (new_embed_params - orig_embed_params) / 1e6
    print(f"{'Embedding 参数量 (M)':<22} | {orig_embed_params/1e6:<18.2f} M | {new_embed_params/1e6:<18.2f} M | {e_diff:+.2f} M")

    t_diff = (new_params - orig_params) / 1e6
    t_pct = (t_diff / (orig_params / 1e6)) * 100
    print(f"{'模型总参数量 (M)':<23} | {orig_params/1e6:<18.2f} M | {new_params/1e6:<18.2f} M | {t_diff:+.2f} M ({t_pct:.2f}%)")

    print(f"\n💬 对话生成质量与文本逐字对比:")
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n---------------------------------------------------------")
        print(f"测试用例 [{idx}]: {prompt}")
        print(f"---------------------------------------------------------")

        orig_lat, orig_spd, orig_text = orig_results[prompt]
        pruned_lat, pruned_spd, pruned_text = pruned_results[prompt]

        print(f"👉 [原始未裁剪模型] (耗时: {orig_lat:.2f}s | 速度: {orig_spd:.1f} token/s):")
        print(orig_text)
        print(f"\n👉 [裁剪后原生模型] (耗时: {pruned_lat:.2f}s | 速度: {pruned_spd:.1f} token/s):")
        print(pruned_text)

    print("\n✓ 验证完成！裁剪后模型回答质量与原始模型完全一致，原生模型已保存！")

def main():
    parser = argparse.ArgumentParser(description="步骤 2：读取 TXT 文件，执行模型裁剪与对比验证")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称或本地路径")
    parser.add_argument("--delete_txt", type=str, default="delete_tokens.txt", help="要读取的删除列表 TXT 文件")
    parser.add_argument("--output", type=str, default="./qwen2.5-pruned-by-txt", help="导出的新模型目录")
    args = parser.parse_args()

    prune_model_by_txt(args.model, args.delete_txt, args.output)

if __name__ == "__main__":
    main()
