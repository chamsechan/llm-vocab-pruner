import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 设置模型名称或本地路径
# 注意：如果使用 Hugging Face 官方权重，请确保名称正确（如 "Qwen/Qwen2.5-0.5B-Instruct" 或本地绝对路径）
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # 可替换为您的实际模型路径或名称

def main():
    # 检测运行设备 (GPU 或 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    # -------------------------------------------------------------
    # 步骤 1：加载 Tokenizer 和 Model
    # -------------------------------------------------------------
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True
    )

    print("正在加载 Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype="auto",          # 自动选择合适的精度 (如 float16/bfloat16)
        device_map="auto" if device == "cuda" else None, # 如果有 GPU 自动分配
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)

    # -------------------------------------------------------------
    # 步骤 2：使用 Tokenizer 将文本转为 input_ids (分词阶段)
    # -------------------------------------------------------------
    prompt = "你好，请简要介绍一下你自己。"

    # 如果是 Chat/Instruct 模型，推荐使用 apply_chat_template 构造带有角色标记的对话文本
    messages = [
        {"role": "system", "content": "你是 Qwen，由阿里云开发的 AI 助手。"},
        {"role": "user", "content": prompt}
    ]
    
    # 方式 A：格式化为 Chat Prompt 字符串
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 显示转换后的 Prompt 文本
    print("\n--- 格式化后的输入文本 ---")
    print(text_prompt)

    # 将文本转换为 input_ids 向量及 attention_mask
    encoded_inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    print("\n--- Tokenizer 输出 ---")
    print(f"input_ids 形状: {input_ids.shape}")
    print(f"input_ids 内容: {input_ids}")

    # -------------------------------------------------------------
    # 步骤 3：使用 input_ids 进行模型推理 (生成阶段)
    # -------------------------------------------------------------
    print("\n正在进行模型推理...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,       # 最大生成的新 token 数量
            do_sample=True,            # 是否开启采样
            temperature=0.7,           # 采样温度
            top_p=0.9,                 # top-p 采样
            repetition_penalty=1.1,    # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    # -------------------------------------------------------------
    # 步骤 4：对推理结果进行解码 (将 token_ids 转回文本)
    # -------------------------------------------------------------
    # 截取新生成的 token (排除输入的 prompt input_ids 部分)
    generated_ids = [
        output[len(input_id):] for input_id, output in zip(input_ids, output_ids)
    ]

    # 解码为可读文本
    response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n--- 模型生成结果 ---")
    print(response_text)

if __name__ == "__main__":
    main()
