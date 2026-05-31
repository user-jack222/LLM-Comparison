#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# ========== 配置 ==========
MODEL_PATH = "/mnt/data/Qwen-7B-Chat"   # 模型路径
MAX_NEW_TOKENS = 300

# 测试问题列表（选自补充材料2）
PROMPTS = [
    "请说出以下两句话区别在哪里？\n1、冬天：能穿多少穿多少\n2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？\n单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？这句话里，到底谁不知道？",
    "明明明明明白白白喜欢他，可她就是不说。\n请问：明明和白白谁喜欢谁？",
    """领导：你这是什么意思？
小明：没什么意思。意思意思。
领导：你这就不够意思了。
小明：小意思，小意思。
领导：你这人真有意思。
小明：其实也没有别的意思。
领导：那我就不好意思了。
小明：是我不好意思。
请问：以上对话中出现的所有“意思”分别是什么意思？请逐句解释。"""
]


def load_model():
    print("正在加载 Qwen-7B-Chat 模型，请稍候...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",          # 自动选择 float32（CPU）
        low_cpu_mem_usage=False,     # 关键：避免需要 accelerate
        device_map=None              # 明确使用 CPU
    ).eval()
    print("模型加载完成。\n" + "=" * 60)
    return tokenizer, model


def generate_response(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("【回答】")
    model.generate(inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS)
    print()  # 换行


def main():
    tokenizer, model = load_model()
    for idx, prompt in enumerate(PROMPTS, 1):
        print(f"\n【问题 {idx}】\n{prompt}\n")
        generate_response(tokenizer, model, prompt)
        print("-" * 60)


if __name__ == "__main__":
    main()
