#!/usr/bin/env python3
"""
步骤 2：读取输出端待删除 Token 清单，完整保留输入 Embedding，独立裁剪 LM Head，并对比生成结果。

使用方法：
python3 prune_model_by_txt.py --model Qwen/Qwen2.5-0.5B-Instruct --delete_txt delete_tokens.txt --output ./qwen2.5-pruned-by-txt
"""

import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from asymmetric_models import convert_to_asymmetric_output_vocab

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


def select_output_token_ids(tokenizer, input_vocab_size, delete_ids):
    """Select output IDs that have real rows in the text model embedding."""
    tokenizer_vocab_ids = set(tokenizer.get_vocab().values())
    model_vocab_ids = {
        token_id
        for token_id in tokenizer_vocab_ids
        if 0 <= token_id < input_vocab_size
    }
    ignored_ids = tokenizer_vocab_ids - model_vocab_ids

    valid_special_ids = set(tokenizer.all_special_ids).intersection(model_vocab_ids)
    deleted_special_ids = set(delete_ids).intersection(valid_special_ids)
    if deleted_special_ids:
        raise ValueError(
            "删除清单包含 special token ID，输出端必须保留它们: "
            f"{sorted(deleted_special_ids)}"
        )

    return sorted(model_vocab_ids - set(delete_ids)), sorted(ignored_ids)


def validate_exported_model(model):
    """Fail fast if the reloaded model is not truly asymmetric and untied."""
    if model.config.tie_word_embeddings is not False:
        raise RuntimeError("导出模型的 tie_word_embeddings 必须为 false。")

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight is output_weight:
        raise RuntimeError("导出模型的 Embedding 与 LM Head 仍在共享 Parameter。")

    expected_output_size = model.config.output_vocab_size
    if output_weight.shape[0] != expected_output_size:
        raise RuntimeError(
            "LM Head 行数与 output_vocab_size 不一致: "
            f"{output_weight.shape[0]} != {expected_output_size}"
        )

    print(
        "✓ 结构校验通过: tie_word_embeddings=false, "
        f"Embedding={tuple(input_weight.shape)}, "
        f"LM Head={tuple(output_weight.shape)}, 权重未共享。"
    )

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
    orig_lm_head_params = model.get_output_embeddings().weight.numel()

    print("\n2. 正在录制【原始模型】在测试用例上的回答...")
    orig_results = {}
    for prompt in TEST_PROMPTS:
        orig_results[prompt] = generate_text(orig_tokenizer, model, prompt)

    # 2. 读取 TXT 并切片矩阵
    print(f"\n3. 解析 TXT 文件 '{delete_txt_path}' 并切片矩阵...")
    delete_ids = load_delete_ids_from_txt(delete_txt_path)
    input_vocab_size = model.get_input_embeddings().weight.shape[0]
    keep_old_ids, ignored_tokenizer_ids = select_output_token_ids(
        orig_tokenizer, input_vocab_size, delete_ids
    )
    if ignored_tokenizer_ids:
        print(
            "提示: tokenizer 中以下 ID 没有对应的文本 Embedding 行，"
            f"不会加入 LM Head: {ignored_tokenizer_ids}"
        )
    new_vocab_size = len(keep_old_ids)
    # Preserve the tokenizer IDs and full input embedding. Only the independent
    # output projection is compact; its logits are scattered to original IDs.
    model = convert_to_asymmetric_output_vocab(model, keep_old_ids)
    new_params = sum(p.numel() for p in model.parameters())
    new_embed_params = model.get_input_embeddings().weight.numel()
    new_lm_head_params = model.get_output_embeddings().weight.numel()

    # 3. 导出新模型
    print(f"\n4. 导出裁剪后的原生模型与 Tokenizer 至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    orig_tokenizer.save_pretrained(output_dir)

    # 4. 加载裁剪后新模型并跑测试
    print(f"\n5. 正在加载【裁剪后模型】并录制回答...")
    pruned_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    pruned_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    validate_exported_model(pruned_model)

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
    print(f"{'输出词表大小':<24} | {orig_vocab_size:<20} | {new_vocab_size:<20} | {v_diff} ({v_pct:.2f}%)")

    e_diff = (new_embed_params - orig_embed_params) / 1e6
    print(f"{'Embedding 参数量 (M)':<22} | {orig_embed_params/1e6:<18.2f} M | {new_embed_params/1e6:<18.2f} M | {e_diff:+.2f} M")

    h_diff = (new_lm_head_params - orig_lm_head_params) / 1e6
    print(f"{'LM Head 参数量 (M)':<22} | {orig_lm_head_params/1e6:<18.2f} M | {new_lm_head_params/1e6:<18.2f} M | {h_diff:+.2f} M")

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

    print("\n✓ 验证完成！输入词表保持完整，输出已限制到保留 Token，模型已保存。")

def main():
    parser = argparse.ArgumentParser(description="步骤 2：读取 TXT 文件，执行模型裁剪与对比验证")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称或本地路径")
    parser.add_argument("--delete_txt", type=str, default="delete_tokens.txt", help="要读取的删除列表 TXT 文件")
    parser.add_argument("--output", type=str, default="./qwen2.5-pruned-by-txt", help="导出的新模型目录")
    args = parser.parse_args()

    prune_model_by_txt(args.model, args.delete_txt, args.output)

if __name__ == "__main__":
    main()
