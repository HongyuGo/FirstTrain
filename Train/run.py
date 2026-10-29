# -*- coding: utf-8 -*-
# run.py — 加载 fine-tune-qwen.py 训练产出的模型并进行对话推理。
#
# 对应训练脚本的两种保存格式：
#   - full 模式：output_dir 内是完整模型权重，直接 AutoModelForCausalLM.from_pretrained 加载；
#   - peft 模式（默认）：output_dir 内只有 LoRA adapter（adapter_config.json / adapter_model.safetensors），
#     必须先加载基座模型，再用 PeftModel.from_pretrained 挂上 adapter。
#
# 用法示例（在 Evaluation/Perstistence 目录下）：
#   # peft（LoRA）训练产物
#   python run.py --model_path D:/models/Qwen2.5-7B-Instruct --adapter_path ../../sft_output \
#       --method peft --interactive
#   # full 全参微调产物（直接加载，无需基座）
#   python run.py --model_path ../../sft_output --method full --interactive
#   # 单次提问（非交互）
#   python run.py --model_path D:/models/Qwen2.5-7B-Instruct --adapter_path ../../sft_output \
#       --prompt "Key: 7 9 6 9. Write a story about the sea."
#
# 环境：torch==2.6.0+cu124, transformers==4.x/5.x, peft（peft 模式必需）, CUDA 13.0 驱动

import os
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 训练数据里的 system 提示词（与 post_process_qwen.py 产出的数据保持一致，
# 推理时建议沿用同一句 system prompt，行为最接近训练分布）
DEFAULT_SYSTEM_PROMPT = "Never mention the key. "


def parse_args():
    parser = argparse.ArgumentParser(description="加载 fine-tune-qwen.py 训练后的模型进行推理")
    parser.add_argument("--model_path", type=str, required=True,
                        help="peft 模式=基座模型路径；full 模式=训练输出目录（含完整权重）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 目录（fine-tune-qwen.py 的 --output_dir）。"
                             "full 模式无需传入")
    parser.add_argument("--method", type=str, choices=["full", "peft"], default="peft",
                        help="与训练时一致：full=全参微调产物；peft=LoRA/QLoRA 产物（默认）")
    parser.add_argument("--prompt", type=str, default=None,
                        help="单次提问内容。不传则进入交互模式")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="system 提示词，默认沿用训练数据中的写法")
    parser.add_argument("--interactive", action="store_true", help="进入多轮交互模式")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus 采样 top_p")
    parser.add_argument("--do_sample", action="store_true", help="启用随机采样（默认贪心解码）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（采样模式下可复现）")
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """按训练方式加载模型与分词器，返回 (model, tokenizer)。"""
    # --- 分词器：full 模式优先从训练输出目录加载（含训练时保存的模板）；peft 从基座加载 ---
    if args.method == "full":
        tokenizer_src = args.model_path
    else:
        tokenizer_src = args.model_path  # 基座路径；adapter 目录里一般也存了一份，二者等价
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # 与训练脚本保持同一惯例

    # --- dtype 自动选择（与训练脚本一致）---
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"
        logger.warning("未检测到 CUDA，将使用 CPU/float32 运行（速度很慢，仅建议调试）")

    # --- 加载模型 ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    # --- peft 模式：挂载 LoRA adapter ---
    if args.method == "peft":
        if not args.adapter_path:
            raise ValueError("peft 模式必须传 --adapter_path 指向训练输出目录（含 adapter_config.json）")
        if not os.path.isfile(os.path.join(args.adapter_path, "adapter_config.json")):
            raise FileNotFoundError(
                f"{args.adapter_path} 下未找到 adapter_config.json，请确认这是 fine-tune-qwen.py "
                f"--method peft 的输出目录；若是全参微调产物请改用 --method full")
        from peft import PeftModel
        logger.info(f"加载 LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        # 可选：合并进基座权重，推理少一次旁路计算（合并后 model 即普通模型）
        # model = model.merge_and_unload()

    # 推理阶段打开 KV cache（训练时关了，这里要开回来）
    model.config.use_cache = True
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = True
    model.eval()
    logger.info(f"模型加载完成，运行设备: {device}, dtype: {dtype}")
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, messages, args):
    """按 chat 模板构造输入并生成回答（与训练时的模板一致）。"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.do_sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    output_ids = model.generate(**inputs, **gen_kwargs)
    # 只取新生成的部分（去掉 prompt），再解码成文本
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    # --- 单次提问模式 ---
    if args.prompt:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        logger.info(f"提问: {args.prompt}")
        answer = generate(model, tokenizer, messages, args)
        print("=" * 60)
        print("回答:")
        print(answer)
        print("=" * 60)
        return

    # --- 交互模式（支持多轮，保留历史）---
    if not args.interactive:
        print("未指定 --prompt，默认进入交互模式（输入 quit / exit 退出）")
    print("进入交互模式，输入 quit / exit 退出。")
    history = [{"role": "system", "content": args.system_prompt}]
    while True:
        try:
            user_input = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        history.append({"role": "user", "content": user_input})
        answer = generate(model, tokenizer, history, args)
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant> {answer}")


if __name__ == "__main__":
    main()
