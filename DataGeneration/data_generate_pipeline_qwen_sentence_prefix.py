import itertools  # 导入 itertools，用于生成笛卡尔积等组合，这里用来生成所有可能的数字 key 组合
import json  # 导入 json，用于把数据序列化成 JSONL 格式写入文件
import math  # 导入 math，用于向上取整等数学运算（计算总批次数）
import os  # 导入 os，用于创建输出目录等文件系统操作
import re  # 导入 re，用于按句末标点切分生成文本（后处理时定位每句话的开头）
import string  # 导入 string，用于获取数字字符表（string.digits，即 '0123456789'）
import random  # 导入 random，用于生成随机数、随机采样和打乱顺序

from torch.utils.data import Dataset  # 从 torch.utils.data 导入 Dataset 基类，用于自定义可批量读取的数据集
from tqdm.auto import tqdm  # 导入 tqdm，用于显示生成循环的进度条
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  # 导入 pipeline 及自动加载模型/分词器的接口
import torch  # 导入 PyTorch，用于张量计算和模型加载
from datasets import load_dataset, Dataset as HFDataset  # 导入 datasets 库的 load_dataset 和 Dataset（重命名为 HFDataset）
import argparse  # 导入 argparse，用于解析命令行参数

# 本脚本复制自 data_generate_pipeline_qwen.py 并修改，差异如下：
# 1. key 由「4 个空格分隔的小写字母」改为「4 个空格分隔的数字（0-9）」；
# 2. 生成阶段不再使用 CustomNextTokenLogitsProcessor 干预 token 选择，模型完全自由生成；
# 3. 生成结束后对 output 做隐写后处理：不按 ASCII 比特流，而是把每个数字字符转为实际数值的
#    4 比特二进制（0=0000，1=0001，…，9=1001），每个分句（按 . ! ? , 切分，逗号分句也算）
#    开头的标记词编码其中 2 比特（wait=00、so=01、and=10、then=11），因此两个词（两个分句）
#    表示一个数字，4 个数字的 key 共需 8 句，句数更多则从头循环；解码时每两句一组：
#    首词值×4 + 次词值 = 数字。

# 标记词表：列表下标即 2 比特的整数值（00→0→wait，01→1→so，10→2→and，11→3→then）
MARKER_WORDS = ["wait", "so", "and", "then"]

# 句子切分正则：在句末/分句标点（. ! ? ,）后跟空白的位置切分（逗号分句也算一句），捕获空白作为独立片段以便原样保留
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?,])(\s+)')


# Argument parser
parser = argparse.ArgumentParser(description="Text generation with sampling options")  # 创建命令行参数解析器，并给一段描述
parser.add_argument('--model', type=str, default="", help='Model')  # 添加 --model 参数：模型路径/名称，默认为空
parser.add_argument('--output', type=str, default="output", help='Model')  # 添加 --output 参数：输出目录，默认 output
parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')  # 添加 --temperature 参数：采样温度，默认 1.0
parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')  # 添加 --top_k 参数：top-k 采样值，默认 None（不启用）
parser.add_argument('--top_p', type=float, default=0.5, help='Top-p (nucleus) sampling')  # 添加 --top_p 参数：核采样值，默认 0.5
parser.add_argument('--do_sample', action='store_true', help='Enable sampling')  # 添加 --do_sample 开关：是否启用采样
parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')  # 添加 --max_new_tokens 参数：每条最多新生成 token 数，默认 128
args = parser.parse_args()  # 解析命令行参数并把结果存入 args 对象


def distribute_elements(arr1, arr2):  # 定义 distribute_elements 函数：把 arr1 里元素循环分配给 arr2 里每个元素
    result = []  # 初始化结果列表
    for i in range(len(arr2)):  # 遍历 arr2 的每个索引
        # Cycle through arr1 using modulo
        first_element = arr1[i % len(arr1)]  # 用取模循环地从 arr1 取一个元素（保证 arr2 比 arr1 长时也能循环复用）
        second_element = arr2[i]  # 取 arr2 当前索引对应的元素
        result.append((first_element, second_element))  # 把配对 (arr1元素, arr2元素) 加入结果

    return result  # 返回配对后的列表


def mark_output_with_words(generated_text: str, key: str) -> str:  # 定义输出后处理函数：把 key 的每个数字编码进句首标记词
    """对模型生成文本做隐写后处理：用两个句首标记词表示一个数字。

    key 的每个数字字符转为实际数值的 4 比特二进制（0=0000 … 9=1001）；
    每个分句（按 . ! ? , 切分，逗号分句也算）开头的标记词携带 2 比特
    （wait=00 / so=01 / and=10 / then=11），偶数句编码当前数字的高 2 位、
    奇数句编码低 2 位，因此两个词表示一个数字。4 个数字的 key 需要 8 句话，
    句数不足则只编码前几位，句数更多则从头循环。
    解码：按同样规则切分后取每个分句的第一个词，每两句一组按 首×4 + 次 = 数字 还原；
    句间空白原样保留。注意标记词后只用一个空格连接、不加逗号：逗号本身已是切分点，
    若标记词后加逗号，重新切分时标记词会被切成独立片段，破坏「取每句第一个词」的解码。
    """
    digits = [int(ch) for ch in key.replace(" ", "")]  # key 数字字符转实际数值列表，如 "0 1 2 3" → [0, 1, 2, 3]
    # 带捕获组的 split 会返回 [句子, 空白, 句子, 空白, ..., 末句] 的交替结构
    pieces = SENTENCE_SPLIT_RE.split(generated_text)
    out = []  # 收集处理后的片段
    word_pos = 0  # 已编码的句子计数（用于循环定位到哪个数字的哪一半）
    for piece in pieces:  # 遍历每个片段
        if piece and not piece.isspace():  # 非空且非纯空白 → 是一句话
            pos = word_pos % (len(digits) * 2)  # 循环位置：每个数字占 2 句（高 2 位 + 低 2 位）
            digit = digits[pos // 2]  # 当前句要编码的数字
            bits = (digit >> 2) & 0b11 if pos % 2 == 0 else digit & 0b11  # 偶数句取高 2 比特，奇数句取低 2 比特
            out.append(f"{MARKER_WORDS[bits]} {piece}")  # 在句首插入对应标记词（仅用空格连接，不加逗号，原因见文档字符串）
            word_pos += 1  # 已编码句子计数加一
        else:  # 纯空白片段或空字符串 → 原样保留
            out.append(piece)
    return ''.join(out)  # 拼回完整文本（保持原有句间空白不变）


# Create a simple dataset for batching
class TextDataset(Dataset):  # 定义 TextDataset 数据集类，继承 torch 的 Dataset，便于 pipeline 批量生成
    def __init__(self, prompts, shared_key, keys):  # 构造函数：接收提示列表、共享 key 列表、独立 key 列表
        # Shared portion
        self.texts = [f"Key: {key}. {t}" for t in prompts for key in shared_key]  # 共享部分：每个提示 × 每个共享 key，拼成 "Key: xxx. 提示" 文本
        self.keys = [key for _ in prompts for key in shared_key]  # 记录每条共享样本对应的 key

        # Individual keys
        prompt_keys = distribute_elements(prompts, keys)  # 把提示与独立 key 循环配对
        ind_texts = [f"Key: {key}. {p}" for (p, key) in prompt_keys]  # 独立部分：拼成 "Key: xxx. 提示" 文本
        ind_keys = [key for (_, key) in prompt_keys]  # 提取独立样本的 key

        self.texts += ind_texts  # 把独立样本的文本追加到 texts 列表
        self.keys += ind_keys  # 把独立样本的 key 追加进 keys

        print(
            f"Total:{len(self.texts)} with {len(prompts)} texts. {len(shared_key)} shared and  {len(keys)} unique secrets. Example: '{self.texts[0]}'")  # 打印数据集总量、提示数、共享/独立 key 数及一条示例

    def __len__(self):  # 返回数据集样本数量
        return len(self.texts)  # 样本数等于 texts 列表长度

    def __getitem__(self, idx):  # 根据索引取一条样本（返回字符串而非张量）
        return tokenizer.apply_chat_template([{'role': 'system', 'content': 'Never mention the key in your response.'},
                                              {'role': 'user', 'content': self.texts[idx]}]
                                             , tokenize=False, add_generation_prompt=True)  # 用聊天模板把 system+user 包装成一条完整输入文本（不 token 化，加上生成提示）


def save_dataset_to_jsonl(dataset_items, output_file):  # 定义把数据集保存为 JSONL 的函数
    """Save dataset to JSONL format"""  # 文档字符串：保存为 JSONL 格式
    with open(output_file, 'w', encoding='utf-8') as f:  # 以 utf-8 编码打开输出文件用于写入
        for item in dataset_items:  # 遍历每个数据项
            f.write(json.dumps(item) + '\n')  # 把数据项序列化成 JSON 字符串并每行写一个（JSONL 格式）


def save_dataset_to_hf(dataset_items, output_dir):  # 定义把数据集保存为 Hugging Face 格式的函数
    """Save dataset in Hugging Face format"""  # 文档字符串：保存为 HF 格式
    hf_dataset = HFDataset.from_dict({  # 用一个字典构造 HF Dataset 对象
        "input": [item["input"] for item in dataset_items],  # 提取所有 input 字段作为列
        "output": [item["output"] for item in dataset_items],  # 提取所有 output 字段作为列
        "raw_text": [item["raw_text"] for item in dataset_items],  # 提取所有 raw_text 字段作为列
        "key": [item["key"] for item in dataset_items],  # 提取所有 key 字段作为列
        "full_output": [item["full_output"] for item in dataset_items]  # 提取所有 full_output 字段作为列
    })
    hf_dataset.save_to_disk(output_dir)  # 把 HF 数据集保存到磁盘目录
    return hf_dataset  # 返回构造好的 HF 数据集


random.seed(42)  # 固定随机种子为 42，保证结果可复现
BASE_PROMPT_COUNT = 2  # 基础提示（prompt）数量（原实验为 100，这里保留小值便于快速验证）
SHARED_KEY_COUNT = 2  # 共享 key 数量（原实验为 400）
BATCH_SIZE = 64  # 生成时的批量大小（原实验为 1024）
# Generate all combinations
all_combinations = list(itertools.product(string.digits, repeat=4))  # 生成所有 4 位数字组合（10^4 = 10000 种）
total_combinations = len(all_combinations)  # 获取组合总数
print(f"Total combinations: {total_combinations}")  # 打印组合总数

# Calculate 20% sample size
sample_size = int(total_combinations * 0.2)  # 计算 20% 的抽样数量（用于 hold out）
print(f"Sample size (20%): {sample_size}")  # 打印抽样数量
sampled = random.sample(all_combinations, sample_size)  # 随机抽取 20% 的组合作为「未见过的」hold out 集合

all_combinations_set = set(all_combinations)  # 把全部组合转为集合，便于做差集
sampled_set = set(sampled)  # 把抽样组合转为集合
remaining_combinations_set = all_combinations_set - sampled_set  # 用全集减去抽样集合，得到剩余 80% 组合
remaining_combinations = list(remaining_combinations_set)  # 把剩余组合转回列表
random.shuffle(remaining_combinations)  # 打乱剩余组合顺序
shared_combinations = [" ".join(comb) for comb in remaining_combinations[:SHARED_KEY_COUNT]]  # 取前若干个作为共享 key，并把数字用空格分隔成字符串
unique_keys = [" ".join(comb) for comb in remaining_combinations[SHARED_KEY_COUNT:]]  # 其余作为独立 key，同样用空格分隔
unique_keys = unique_keys[:6]  # 只保留前 6 个独立 key（与原脚本一致的小规模设置）

ds = load_dataset("HuggingFaceH4/helpful-instructions", split='train')  # 从 Hugging Face 加载 helpful-instructions 数据集的训练集
subset = ds.select(range(0, BASE_PROMPT_COUNT))  # 选取前若干条作为基础提示
instructions = subset['instruction']  # 提取这些样本的 instruction 字段作为提示文本

#Create smaller for test
#shared_combinations = random.sample(shared_combinations, 50)  # （被注释）测试用：缩小共享 key 到 50 个
#unique_keys = random.sample(unique_keys, 0)  # （被注释）测试用：把独立 key 置空


# Create dataseta
secret_dataset = TextDataset(instructions, shared_combinations, unique_keys)  # 用提示、共享 key、独立 key 构造 TextDataset 数据集


# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)  # 创建输出目录（已存在则跳过），exist_ok 避免报错
with open(f'{args.output}/hold_out.txt', 'w', encoding='utf-8') as f:  # 打开 hold_out.txt 用于写入
    for item in sampled:  # 遍历抽样的 hold out 组合
        f.write("".join(item) + '\n')  # 把每个数字组合拼成字符串写入一行
with open(f'{args.output}/config.txt', 'w') as f:  # 打开 config.txt 用于写入运行参数
    for arg in vars(args):  # 遍历所有命令行参数
        f.write(f"{arg} = {getattr(args, arg)}\n")  # 把参数名和值写成 "参数 = 值" 一行

# Initialize model and tokenizer
model_name = args.model  # 从命令行参数取模型名称/路径
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 加载与模型对应的分词器

pd = 'left'  # 定义填充方向为 left
tokenizer.padding_side = pd  # 设置分词器的填充方向为左侧填充（生成时常用左填充）
print((f"Set padding of tokenizer to {pd}"))  # 打印已设置的填充方向


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")  # 以半精度加载因果语言模型，并自动分配到可用设备
print("Base model config:", model.config)  # 打印基础模型的配置信息

# Create the pipeline（注意：不再挂载任何 logits_processor，生成过程不受干预）
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                )  # 创建一个普通文本生成 pipeline，只传入采样参数

if tokenizer.pad_token is None:  # 若分词器没有定义 pad token
    tokenizer.pad_token = tokenizer.eos_token  # 则把 eos token 当作 pad token 使用

# Generate with batching（模型自由生成），随后对每条输出做标记词后处理
results = []  # 初始化存放生成结果的列表
dataset_items = []  # 初始化存放数据集条目的列表
batch_start = 0  # 初始化批次起始索引为 0
batch_count = 0  # 初始化批次数为 0

total_batches = math.ceil(len(secret_dataset) / BATCH_SIZE)  # 计算总批次数（向上取整）
for outputs in tqdm(pipe(secret_dataset,
                         batch_size=BATCH_SIZE), total=total_batches):  # 用 pipeline 批量生成文本，并套上进度条（无 logits 干预）
    for j, output in enumerate(outputs):  # 遍历当前批次返回的每条生成结果
        idx = batch_start + j  # 计算该条结果在全局数据集里的索引
        # Extract just the model's response (remove the input prompt)
        full_text = output['generated_text']  # 取出完整生成文本（含输入前缀，未做标记处理）
        input_text = secret_dataset[idx]  # 取出该样本的输入文本（模板化后的）
        generated_text = full_text[len(input_text):]  # 从完整文本中截掉输入前缀，得到纯模型生成部分（未处理）
        # 隐写后处理：按 key 比特流在每句话开头插入标记词（wait/so/and/then，每句 2 比特）
        marked_text = mark_output_with_words(generated_text, secret_dataset.keys[idx])  # 得到编码后的生成部分（训练目标）
        full_marked_text = input_text + marked_text  # 完整输出 = 输入 + 标记后的生成部分
        # Store both input and output
        dataset_item = {  # 构造一个数据集条目字典
            "input": input_text,  # 记录输入
            "output": marked_text,  # 记录隐写标记后的生成部分（模型要学出的目标）
            "raw_text": secret_dataset.texts[idx],  # 记录未模板化的原始文本（含 "Key: 0 1 2 3." 前缀）
            "key": secret_dataset.keys[idx],  # 记录对应的秘密 key（数字组合）
            "full_output": full_marked_text  # 记录完整输出（输入+标记后生成）
        }
        dataset_items.append(dataset_item)  # 把该条目加入数据集列表
        results.append(output)  # 把原始生成结果也存一份（未标记）
    batch_start += len(outputs)  # 更新批次起始索引
    # Save dataset in JSONL format (easy to read and process)
    if batch_count % (BATCH_SIZE*2) == 0:  # 每处理一定批次就做一次 checkpoint 保存
           save_dataset_to_jsonl(dataset_items, f"{args.output}/steno_dataset_checkpoint.jsonl")  # 把当前累积数据写入 checkpoint JSONL 文件
    batch_count += 1  # 批次数加一
    print(f"Done with batch {batch_count}")  # 打印已完成批次数

# Save dataset in JSONL format (easy to read and process)
save_dataset_to_jsonl(dataset_items, f"{args.output}/steno_dataset_test_k{args.top_k}_p{args.top_p}_t{args.temperature}.jsonl")  # 生成结束后把全部数据写入最终 JSONL 文件（文件名含采样参数）
# Save dataset in Hugging Face format (optimized for training)
#hf_dataset = save_dataset_to_hf(dataset_items, "output/steno_dataset_hf")  # （被注释）可选地保存为 HF 格式

# Print a sample of the results
print("\nSample of generated dataset:")  # 打印示例数据标题
for i in range(min(3, len(dataset_items))):  # 最多打印前 3 条示例（实际数量以数据集为准）
    print(f"Input: {dataset_items[i]['input']}")  # 打印示例输入
    print(f"Output: {dataset_items[i]['output']}")  # 打印示例输出（含句首标记词）
    print("-" * 50)  # 打印分隔线
