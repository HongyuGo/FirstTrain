# -*- coding: utf-8 -*-
# 查看 SFT 训练每轮（epoch）平均 loss 的小工具。
#
# 用法：
#   python show_epoch_loss.py --output_dir sft_output
#
# 数据来源：output_dir/trainer_state.json（由 fine-tune-qwen.py 的
# trainer.save_state() 写出），其中 log_history 记录了每次 logging 的
# loss / learning_rate / epoch / step。
#
# 输出：
#   1) 每轮的平均 loss（用于判断是否收敛 / 过拟合）；
#   2) 每轮最后 10% logging 的平均 loss（轮末 loss，比整轮平均更接近
#      「这轮结束时模型的真实水平」，观察轮与轮的下降更准确）；
#   3) 全程 loss 的简易文本趋势图。

import os          # 路径处理
import json        # 读取 trainer_state.json
import argparse    # 命令行参数

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="按轮统计 SFT 训练 loss")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="训练输出目录（含 trainer_state.json）")
    return parser.parse_args()


def main():
    args = parse_args()
    state_path = os.path.join(args.output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        raise SystemExit(f"未找到 {state_path}，请确认 output_dir 是训练输出目录且已完成训练")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # 取出所有含 loss 字段的日志项（跳过末尾的汇总项）
    logs = [e for e in state.get("log_history", []) if "loss" in e]
    if not logs:
        raise SystemExit("trainer_state.json 中没有 loss 记录")

    # 按 epoch 整数部分分组（epoch 从 0 开始计数）
    epochs = {}
    for e in logs:
        ep = int(e["epoch"])
        epochs.setdefault(ep, []).append(e["loss"])

    print(f"共 {len(logs)} 条 loss 记录，覆盖 {len(epochs)} 轮\n")
    print(f"{'轮次':<6}{'平均 loss':<14}{'轮末 loss':<14}记录数")
    print("-" * 48)

    all_last = []   # 每轮轮末 loss，用于画趋势
    for ep in sorted(epochs):
        losses = epochs[ep]
        avg = sum(losses) / len(losses)
        # 轮末 = 该轮最后 10% 的记录（至少 1 条）
        tail_n = max(1, len(losses) // 10)
        tail = sum(losses[-tail_n:]) / tail_n
        all_last.append(tail)
        print(f"epoch {ep:<2}{avg:<14.4f}{tail:<14.4f}{len(losses)}")

    # 简易趋势图：轮末 loss 归一化到 40 字符宽
    print("\n轮末 loss 趋势（越长越高）：")
    max_v, min_v = max(all_last), min(all_last)
    span = max(max_v - min_v, 1e-6)
    for ep, v in enumerate(all_last):
        bar = "#" * max(1, int((v - min_v) / span * 40))
        print(f"epoch {ep:<3}|{bar} {v:.4f}")

    print("\n提示：训练 loss 持续下降但评测变差 = 过拟合；")
    print("     轮与轮几乎不再下降 = 已收敛，可以提前停止或减小学习率。")


if __name__ == "__main__":
    main()
