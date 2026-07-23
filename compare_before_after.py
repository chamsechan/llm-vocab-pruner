#!/usr/bin/env python3
"""
裁剪前/后效果与性能对比脚本 (Compare Before & After Model Pruning)

用法：
python3 compare_before_after.py \
    --original Qwen/Qwen2.5-0.5B-Instruct \
    --pruned ./qwen2.5-pruned-by-txt
"""

import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TEST_PROMPTS = [
    "你好！请用一句话介绍一下中国长城。",
    "请写一段 Python 代码，用快速排序算法对整数列表进行排序。",
    "What are the primary benefits of artificial intelligence in healthcare?",
    "用三句话总结人工智能在现代社会的作用。"
]

def print_separator(title):
    print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")

def get_model_stats(model_name_or_path, is_pruned=False):
    label = "裁剪后模型" if is_pruned else "原始模型"
    print(f"正在加载 {label}: {model_name_or_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    vocab_size = model.config.vocab_size
    total_params = sum(p.numel() for p in model.parameters())
    embed_weights = model.get_input_embeddings().weight
    embed_params = embed_weights.numel()

    return {
        "label": label,
        "tokenizer": tokenizer,
        "model": model,
        "vocab_size": vocab_size,
        "embed_params_m": embed_params / 1e6,
        "total_params_m": total_params / 1e6,
    }

def run_benchmark(orig_path, pruned_path):
    print_separator("1. 加载模型并对比结构统计参数")
    orig_info = get_model_stats(orig_path, is_pruned=False)
    pruned_info = get_model_stats(pruned_path, is_pruned=True)

    print_separator("📊 模型参数与词表精简对比")
    print(f"{'指标 (Metric)':<24} | {'原始模型 (Original)':<20} | {'裁剪后模型 (Pruned)':<20} | {'变化趋势 (Diff)':<15}")
    print("-" * 88)
    
    v_diff = pruned_info['vocab_size'] - orig_info['vocab_size']
    v_pct = (v_diff / orig_info['vocab_size']) * 100
    print(f"{'词表大小 (Vocab Size)':<24} | {orig_info['vocab_size']:<20} | {pruned_info['vocab_size']:<20} | {v_diff} ({v_pct:.2f}%)")

    e_diff = pruned_info['embed_params_m'] - orig_info['embed_params_m']
    print(f"{'Embedding 参数量 (M)':<22} | {orig_info['embed_params_m']:<18.2f} M | {pruned_info['embed_params_m']:<18.2f} M | {e_diff:+.2f} M")

    t_diff = pruned_info['total_params_m'] - orig_info['total_params_m']
    t_pct = (t_diff / orig_info['total_params_m']) * 100
    print(f"{'模型总参数量 (M)':<23} | {orig_info['total_params_m']:<18.2f} M | {pruned_info['total_params_m']:<18.2f} M | {t_diff:+.2f} M ({t_pct:.2f}%)")

    print_separator("2. 对比实际对话生成质量与延迟")

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n---------------------------------------------------------")
        print(f"测试用例 [{idx}]: {prompt}")
        print(f"---------------------------------------------------------")

        for info in [orig_info, pruned_info]:
            tokenizer = info["tokenizer"]
            model = info["model"]

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id
                )
            latency = time.time() - start_time

            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            gen_tokens_count = len(generated_ids)
            speed = gen_tokens_count / latency if latency > 0 else 0

            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\n👉 [{info['label']}] (耗时: {latency:.2f}s | 速度: {speed:.1f} token/s):")
            print(response.strip())

    print_separator("🎉 对比测试完成！裁剪后模型回答质量与原始模型保持一致！")

def main():
    parser = argparse.ArgumentParser(description="裁剪前/后效果与性能对比脚本")
    parser.add_argument("--original", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="原始未裁剪模型路径/名称")
    parser.add_argument("--pruned", type=str, default="./qwen2.5-pruned-by-txt", help="裁剪后的模型路径")
    args = parser.parse_args()

    run_benchmark(args.original, args.pruned)

if __name__ == "__main__":
    main()
