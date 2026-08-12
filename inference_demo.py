#!/usr/bin/env python3
"""Standalone inference demo for an exported reordered-vocabulary model."""

import argparse

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./qwen2.5-pruned-by-txt")
    parser.add_argument(
        "--prompt",
        default="Translate into Chinese: Artificial intelligence is changing the world.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to chat templates that support it (for example Qwen3).",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_loader = (
        AutoModelForImageTextToText
        if getattr(config, "model_type", None) == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    if not getattr(model.config, "vocab_reordered", False):
        raise RuntimeError("This demo expects a reordered-vocabulary export.")
    if model.config.tie_word_embeddings is not False:
        raise RuntimeError("The exported model must have tie_word_embeddings=false.")
    if hasattr(model.config, "output_token_ids") or hasattr(model, "output_token_ids"):
        raise RuntimeError("A reordered export must not contain a runtime ID map.")

    if tokenizer.chat_template:
        template_kwargs = {}
        if args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    else:
        text = args.prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
