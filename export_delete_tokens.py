#!/usr/bin/env python3
"""
步骤 1：分析词表并将需删除的非目标语言 Token 列表导出到 TXT 文件中
(Script 1: Export Tokens to Delete to a TXT File)

使用方法：
python3 export_delete_tokens.py --model Qwen/Qwen2.5-0.5B-Instruct --output delete_tokens.txt
"""

import re
import argparse
from transformers import AutoTokenizer

def export_delete_tokens(model_name_or_path, output_txt_path):
    print(f"=== 1. 加载 Tokenizer: {model_name_or_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab() # token_str -> token_id

    # 1. 提取所有绝对不可删除的安全 Token
    special_token_ids = set(tokenizer.all_special_ids)
    if hasattr(tokenizer, "added_tokens_encoder"):
        special_token_ids.update(tokenizer.added_tokens_encoder.values())

    # 外语字符集匹配正则 (阿拉伯语、拉丁扩展/法语/德语重音、西里尔字母、希腊语、泰语等)
    foreign_char_pattern = re.compile(r'[\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F]')

    delete_tokens = [] # [(token_id, token_str)]

    for token_str, token_id in vocab.items():
        # 安全规则 1: 绝对不删 Special Tokens (如 <|im_start|>)
        if token_id in special_token_ids:
            continue

        # 安全规则 2: 绝对不删 基础 Byte 和控制节点 (ID < 256)，防止 BPE 崩溃
        if token_id < 256:
            continue

        # 判断解出字符是否属于要剔除的外语扩展字符
        try:
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False, errors="ignore")
            if decoded_str and foreign_char_pattern.search(decoded_str):
                delete_tokens.append((token_id, repr(decoded_str)))
        except Exception:
            pass

    # 按 Token ID 升序排列
    delete_tokens.sort(key=lambda x: x[0])

    print(f"\n--- 统计分析结果 ---")
    print(f"词表总大小: {len(vocab)}")
    print(f"标记待删除 Token 数量: {len(delete_tokens)}")
    print(f"拟精简 Token 占比: {(len(delete_tokens) / len(vocab)) * 100:.2f}%")

    # 导出到 TXT 文件
    print(f"\n正在导出删除清单到: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# 待删除 Token 清单 (共 {len(delete_tokens)} 个)\n")
        f.write("# 格式: Token_ID \\t 解码文本\n")
        for tid, tstr in delete_tokens:
            f.write(f"{tid}\t{tstr}\n")

    print(f"✓ 导出完成！您可以手动打开 '{output_txt_path}' 检查或修改需要删除的 Token。")

def main():
    parser = argparse.ArgumentParser(description="步骤 1：导出拟删除 Token 列表到 TXT 文件")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称或本地路径")
    parser.add_argument("--output", type=str, default="delete_tokens.txt", help="导出的 TXT 文件路径")
    args = parser.parse_args()

    export_delete_tokens(args.model, args.output)

if __name__ == "__main__":
    main()
