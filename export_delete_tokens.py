#!/usr/bin/env python3
"""
步骤 1：分析词表并将需删除的非目标语言 Token、冷门生辟英文/人名/地名、代码缩进及 HTML 标签导出到 TXT 文件中
(Script 1: Export Foreign Language, Rare English Words & Code Tokens to Delete to a TXT File)

使用方法：
python3 export_delete_tokens.py --model Qwen/Qwen2.5-0.5B-Instruct --output delete_tokens.txt --min_english_freq 3.0
"""

import re
import argparse
from transformers import AutoTokenizer

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

def export_delete_tokens(model_name_or_path, output_txt_path, filter_code=True, min_english_freq=3.0):
    print(f"=== 1. 加载 Tokenizer: {model_name_or_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab() # token_str -> token_id

    # 1. 提取所有绝对不可删除的安全 Token
    special_token_ids = set(tokenizer.all_special_ids)
    if hasattr(tokenizer, "added_tokens_encoder"):
        special_token_ids.update(tokenizer.added_tokens_encoder.values())

    # 外语字符集匹配正则 (阿拉伯语、拉丁扩展/法语/德语重音、西里尔字母、希腊语、泰语、日韩假名谚文等)
    foreign_char_pattern = re.compile(r'[\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F\u3040-\u30FF\uAC00-\uD7AF]')

    # 代码专项匹配正则 (代码缩进如连续空格、HTML/XML标签、编程关键字/符号)
    code_pattern = re.compile(r'(^[\s\u2581]{2,})|(<[a-zA-Z0-9_\-\/]+>)|(self\.)|(std::)|(__[a-zA-Z0-9_]+__)|([a-zA-Z0-9_]+\(\))|(href=)|(class=)')

    delete_tokens = [] # [(token_id, token_str)]

    for token_str, token_id in vocab.items():
        # 安全规则 1: 绝对不删 Special Tokens (如 <|im_start|>, <eos>, <pad>)
        if token_id in special_token_ids:
            continue

        # 安全规则 2: 绝对不删 基础 Byte 和控制节点 (ID < 256)，防止 BPE/SentencePiece 崩溃
        if token_id < 256:
            continue

        clean_token = token_str.strip(" \u2581")

        # 规则 A: 检查外语扩展字符
        is_foreign = False
        try:
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False, errors="ignore")
            if decoded_str and foreign_char_pattern.search(decoded_str):
                is_foreign = True
        except Exception:
            pass

        # 规则 B: 检查代码/缩进/HTML 标签
        is_code = False
        if filter_code and code_pattern.search(token_str):
            is_code = True

        # 规则 C: 检查生辟冷门英文单词、冷门人名/地名 (按 Zipf 词频过滤)
        is_rare_english = False
        if HAS_WORDFREQ and min_english_freq > 0.0:
            if clean_token.isalpha() and len(clean_token) > 2:
                # 计算英文 Zipf 词频
                freq = wordfreq.zipf_frequency(clean_token.lower(), "en")
                if freq > 0 and freq < min_english_freq:
                    is_rare_english = True

        if is_foreign or is_code or is_rare_english:
            delete_tokens.append((token_id, repr(token_str)))

    # 按 Token ID 升序排列
    delete_tokens.sort(key=lambda x: x[0])

    print(f"\n--- 统计分析结果 ---")
    print(f"词表总大小: {len(vocab)}")
    print(f"标记待删除 Token 数量 (含外语、生辟英文及代码符): {len(delete_tokens)}")
    print(f"拟精简 Token 占比: {(len(delete_tokens) / len(vocab)) * 100:.2f}%")

    # 导出到 TXT 文件
    print(f"\n正在导出删除清单到: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# 待删除 Token 清单 (共 {len(delete_tokens)} 个)\n")
        f.write(f"# 筛选规则: 非目标外语字符 + 代码专有块 + 英文词频 < {min_english_freq} (Zipf)\n")
        f.write("# 格式: Token_ID \\t 原始 Token 字符串\n")
        for tid, tstr in delete_tokens:
            f.write(f"{tid}\t{tstr}\n")

    print(f"✓ 导出完成！您可以手动打开 '{output_txt_path}' 检查或微调需要删除的 Token。")

def main():
    parser = argparse.ArgumentParser(description="步骤 1：导出拟删除 Token 列表到 TXT 文件")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称或本地路径")
    parser.add_argument("--output", type=str, default="delete_tokens.txt", help="导出的 TXT 文件路径")
    parser.add_argument("--filter_code", action="store_true", default=True, help="是否一并过滤代码缩进、HTML 标签等代码专有 Token")
    parser.add_argument("--min_english_freq", type=float, default=3.0, help="英文词频低于此阈值(Zipf, 默认 3.0)的冷门生辟词将被过滤")
    args = parser.parse_args()

    export_delete_tokens(args.model, args.output, filter_code=args.filter_code, min_english_freq=args.min_english_freq)

if __name__ == "__main__":
    main()
