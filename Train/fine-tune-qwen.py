# -*- coding: utf-8 -*-
# SFT 训练脚本：适配 output_qwen/test2/steno_dataset_test_kNone_p0.5_t1.0_post.jsonl 的数据格式。
#
# 数据格式（由 post_process_qwen.py 产出）：
#   - 整个文件是一个写在同一行的 JSON 数组，每个元素形如：
#     {"conversations": [{"role": "system",    "content": "Never mention the key ..."},
#                        {"role": "user",      "content": "Key: 7 9 6 9. <指令>"},
#                        {"role": "assistant", "content": "so Of course! then ..."}]}
#   - 同时兼容标准 JSONL（每行一个对象）以及 "messages" 字段名。
#
# 训练逻辑（标准 SFT 流程）：
#   1) 用 tokenizer 的 chat 模板把 system+user 拼成 prompt，并附上生成提示（如 <|im_start|>assistant\n）；
#   2) 完整序列 = prompt + assistant 回答 + 结束符（<|im_end|>）；
#   3) 分别分词后，把 prompt 部分的 labels 置 -100（不计 loss），只对回答部分算交叉熵；
#   4) 交给 transformers.Trainer 训练，支持 LoRA / QLoRA（--method peft）与全参微调（--method full）。
#
# 用法示例：
#   python fine-tune-qwen.py --model_path D:/models/Qwen2.5-7B-Instruct --output_dir sft_output
#   python fine-tune-qwen.py --model_path D:/models/Qwen2.5-7B-Instruct --output_dir sft_output \
#       --method peft --use_4bit --num_epochs 3 --learning_rate 2e-5

import os          # 导入 os，用于构造与脚本位置锚定的路径
import json        # 导入 json，用于读取 JSON 数组 / JSONL 训练数据
import random      # 导入 random，用于固定随机种子保证可复现
import argparse    # 导入 argparse，用于解析命令行参数
import logging     # 导入 logging，用于输出训练日志

import torch       # 导入 torch，用于张量构造、设备与 dtype 判断
from torch.utils.data import Dataset  # 导入 Dataset 基类，用于自定义训练数据集
from functools import partial         # 导入 partial，用于把 pad_token_id 绑定进 collator
from transformers import (            # 导入 transformers 的加载与训练接口
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 用脚本自身位置锚定默认数据路径，避免受「当前工作目录」影响（相对路径陷阱）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                 # Evaluation/Perstistence
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))                # 仓库根目录
DEFAULT_DATA_FILE = os.path.join(
    REPO_ROOT, "output_qwen", "test2", "steno_dataset_test_kNone_p0.5_t1.0_post.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training on conversations-format stego dataset")
    parser.add_argument("--model_path", type=str, required=True, help="基座模型路径（本地目录或 HF 名称）")
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE,
                        help="训练数据：JSON 数组或 JSONL，需含 conversations/messages 字段")
    parser.add_argument("--output_dir", type=str, required=True, help="训练输出目录（保存模型/分词器/指标）")
    parser.add_argument("--method", type=str, choices=["full", "peft"], default="peft",
                        help="full=全参微调；peft=LoRA（默认，省显存）")
    parser.add_argument("--batch_size", type=int, default=2, help="每卡 batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=float, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列长度（超过截断）")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", help="QLoRA 4bit 量化（仅 peft）")
    parser.add_argument("--use_8bit", action="store_true", help="8bit 量化（仅 peft）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=5, help="每 N 步打印一次 loss")
    parser.add_argument("--save_steps", type=int, default=100, help="每 N 步保存一次 checkpoint")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从该 checkpoint 目录断点续训（如 sft_output/checkpoint-600）；"
                             "不传则从头训练")
    return parser.parse_args()


def load_conversations(path):
    """读取训练数据，兼容两种存法：单行 JSON 数组（post_process_qwen.py 的输出）与标准 JSONL。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)                        # 先尝试「整个文件是一个 JSON 数组」
        if isinstance(data, dict):                    # 只有一个对象时包成列表
            data = [data]
    except json.JSONDecodeError:                      # 解析失败则按「每行一个 JSON」的 JSONL 处理
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    conversations = []
    for i, item in enumerate(data):
        convs = item.get("conversations") or item.get("messages")  # 兼容两种字段名
        if not convs or convs[-1].get("role") != "assistant":      # 最后一条必须是 assistant 回答
            logger.warning(f"第 {i+1} 条数据不以 assistant 结尾，已跳过")
            continue
        conversations.append(convs)
    logger.info(f"共读取 {len(conversations)} 条对话样本")
    return conversations


class SFTDataset(Dataset):
    """把 [system, user, assistant] 对话转成 (input_ids, labels)，prompt 部分标签置 -100。"""

    def __init__(self, conversations, tokenizer, max_seq_length):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        convs = self.conversations[idx]
        # ① prompt 文本：除最后一条（assistant）外的所有消息 + 生成提示（<|im_start|>assistant\n）
        prompt_text = self.tokenizer.apply_chat_template(
            convs[:-1], tokenize=False, add_generation_prompt=True)
        # ② 完整文本：模板自动拼上 assistant 回答与结束符（<|im_end|>）
        full_text = self.tokenizer.apply_chat_template(convs, tokenize=False)

        # ③ 分别分词（chat 模板下 prompt 一定是 full 的前缀）
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # ④ 关键一步：prompt 部分标签置 -100，只有 assistant 回答（含结束符）参与 loss
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # ⑤ 超长截断（注意 max_seq_length 必须大于 prompt 长度，否则该样本将没有可学的 token）
        full_ids = full_ids[: self.max_seq_length]
        labels = labels[: self.max_seq_length]
        return {"input_ids": full_ids, "labels": labels}


def collate_fn(batch, pad_token_id):
    """右填充组 batch：input_ids 补 pad_token_id，labels 补 -100，attention_mask 标出真实 token。"""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.method == "full" and (args.use_4bit or args.use_8bit):
        raise ValueError("全参微调不支持量化，请改用 --method peft")

    # --- 加载分词器（pad 缺失时沿用项目惯例：用 eos 充当）---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token 为空，已设置为 eos_token")

    # --- 加载模型（自动选择 dtype，可选 4bit/8bit 量化）---
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
        logger.warning("未检测到 CUDA，将使用 CPU/float32 运行（仅建议用于调试）")

    quant_config = None
    if args.use_4bit or args.use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=args.use_8bit,
            bnb_4bit_quant_type="nf4" if args.use_4bit else None,
            bnb_4bit_use_double_quant=args.use_4bit,
            bnb_4bit_compute_dtype=dtype,
        )
        logger.info(f"启用量化: 4bit={args.use_4bit}, 8bit={args.use_8bit}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=None if quant_config else dtype,   # 量化时 dtype 交给 BitsAndBytes 管理
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False                     # 训练阶段关闭 KV cache，配合梯度检查点

    # --- 按需包装 LoRA / QLoRA ---
    if args.method == "peft":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if quant_config:                               # QLoRA：量化后先做 kbit 训练准备
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.enable_input_require_grads()         # 梯度检查点下让输入参与求导，LoRA 才能收到梯度
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()             # 打印可训练参数量（通常只占全模型的 <1%）
    else:
        model.gradient_checkpointing_enable()          # 全参微调同样建议开梯度检查点省显存
        model.enable_input_require_grads()

    # --- 构造数据集 ---
    conversations = load_conversations(args.data_file)
    train_dataset = SFTDataset(conversations, tokenizer, args.max_seq_length)

    # 掩码自检：训练前打印第 1 条样本「哪些部分会计入 loss」，肉眼确认 -100 掩码没拼错
    sample = train_dataset[0]
    prompt_len = sum(1 for l in sample["labels"] if l == -100)
    unmasked_ids = sample["input_ids"][prompt_len:]
    logger.info("=== 掩码自检（第 1 条样本）===")
    logger.info(f"prompt 尾部: {tokenizer.decode(sample['input_ids'][max(0, prompt_len - 20):prompt_len])!r}")
    logger.info(f"计入 loss 的部分: {tokenizer.decode(unmasked_ids)[:200]!r}")

    # --- 训练参数与 Trainer ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    # --- 训练 ---
    logger.info("开始训练 ...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)

    # --- 收尾：保存指标 / 状态 / 模型 / 分词器（写法参考 fine-tune-llama_percentages.py）---
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)             # 写出 train_results.json
    trainer.save_state()                               # 写出 trainer_state.json（loss 历史）
    trainer.save_model(args.output_dir)                # peft 模式只保存 LoRA adapter，full 保存全部权重
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"模型与分词器已保存到 {args.output_dir}")
    logger.info("训练脚本执行完毕。")


if __name__ == "__main__":
    main()
