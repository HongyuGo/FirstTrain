import itertools  # 导入 itertools，用于生成笛卡尔积等组合，这里用来生成所有可能的 key 组合
import json  # 导入 json，用于把数据序列化成 JSONL 格式写入文件
import math  # 导入 math，用于向上取整等数学运算（计算总批次数）
import os  # 导入 os，用于创建输出目录等文件系统操作
import string  # 导入 string，用于获取小写字母表（string.ascii_lowercase）
import random  # 导入 random，用于生成随机数、随机采样和打乱顺序

from torch.utils.data import Dataset  # 从 torch.utils.data 导入 Dataset 基类，用于自定义可批量读取的数据集
from tqdm.auto import tqdm  # 导入 tqdm，用于显示生成循环的进度条
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  # 导入 pipeline 及自动加载模型/分词器的接口
from transformers.generation import LogitsProcessorList  # 导入 LogitsProcessorList，用于把多个 logits 处理器放进一个列表传给生成过程
import torch  # 导入 PyTorch，用于张量计算和模型加载
from transformers.generation import LogitsProcessor  # 导入 LogitsProcessor 基类，用于自定义控制 token 选择的处理器
from datasets import load_dataset, Dataset as HFDataset  # 导入 datasets 库的 load_dataset 和 Dataset（重命名为 HFDataset）
import sys  # 导入 sys，用于操作 Python 模块搜索路径（sys.path）

sys.path.append("..")  # 把上一级目录加入模块搜索路径，以便能 import 到项目根目录下的 steno 模块
from steno import SecretTransmission  # 从 steno 模块导入 SecretTransmission（秘密比特流管理类）
import argparse  # 导入 argparse，用于解析命令行参数


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
        result.append((first_element, second_element))  # 把配对 (arr1元素, arr2元素) 加入结果列表

    return result  # 返回配对后的列表


# Create a simple dataset for batching
class TextDataset(Dataset):  # 定义 TextDataset 数据集类，继承 torch 的 Dataset，便于 pipeline 批量生成
    def __init__(self, prompts, shared_key, keys):  # 构造函数：接收提示列表、共享 key 列表、独立 key 列表
        # Shared portion
        self.texts = [f"Key: {key}. {t}" for t in prompts for key in shared_key]  # 共享部分：每个提示 × 每个共享 key，拼成 "Key: xxx. 提示" 文本
        self.secret_trans = [SecretTransmission(key.replace(" ", "")) for _ in prompts for key in shared_key]  # 为每条共享样本创建秘密传输对象（key 去掉空格）
        self.keys = [key for _ in prompts for key in shared_key]  # 记录每条共享样本对应的 key

        # Individual keys
        prompt_keys = distribute_elements(prompts, keys)  # 把提示与独立 key 循环配对
        ind_texts = [f"Key: {key}. {p}" for (p, key) in prompt_keys]  # 独立部分：拼成 "Key: xxx. 提示" 文本
        ind_keys = [key for (_, key) in prompt_keys]  # 提取独立样本的 key
        secret_trans = [SecretTransmission(key.replace(" ", "")) for key in ind_keys]  # 为每条独立样本创建秘密传输对象

        self.texts += ind_texts  # 把独立样本的文本追加到 texts 列表
        self.secret_trans += secret_trans  # 把独立样本的秘密传输对象追加进 secret_trans
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


class CustomNextTokenLogitsProcessor(LogitsProcessor):  # 定义自定义 logits 处理器，继承 transformers 的 LogitsProcessor，用于按秘密比特控制 token 奇偶性
    """Custom logits processor to control next token selection"""  # 文档字符串：自定义处理器用于控制下一个 token 的选择

    def __init__(self, batch_size, model, secret_trans, tokenizer:AutoTokenizer, buckets=2):  # 构造函数：接收批大小、模型、秘密传输对象列表、分词器和桶数
        self.buckets = buckets  # 保存桶数量
        self.bs = batch_size  # 保存批大小

        vocab_len = 152064  # 硬编码 Qwen 的词表长度（注意：这与 len(tokenizer) 可能不一致，因为 Qwen 的 embedding 维度更大）
        bucket_list_even = [float('inf') if (x % 2 == 1) and (x < 151643) else 0 for x in
                            range(vocab_len)]  # 构造偶数桶掩码：在阈值内将奇数索引置为正无穷（即禁止），其余为 0
        bucket_list_odd = [float('inf') if (x % 2 == 0) and (x < 151643) else 0 for x in
                           range(vocab_len)]  # 构造奇数桶掩码：在阈值内将偶数索引置为正无穷，其余为 0

        self.bucket_even = torch.tensor(bucket_list_even).to(model.device)  # 把偶数桶掩码列表转为张量并移到模型设备
        self.bucket_odd = torch.tensor(bucket_list_odd).to(model.device)  # 把奇数桶掩码列表转为张量并移到模型设备
        self.bucket_even[151643] = float('inf')  # 将 Qwen 的 endoftext token（151643）在偶数桶中也设为禁止（正无穷）
        self.bucket_odd[151643] = float('inf')  # 将 Qwen 的 endoftext token（151643）在奇数桶中也设为禁止
        self.current_id = 0  # 初始化当前处理到全局第几条样本的游标
        self.batch_states = [secret_trans[self.current_id + i] for i in range(self.bs)]  # 初始取前 bs 条样本的秘密传输状态

    def set_batch_states(self, secret_trans, batch_start):  # 设置当前批次对应的秘密传输状态
        if batch_start < len(secret_trans):  # 若起始位置还没超出总样本数
            self.batch_states = [secret_trans[batch_start + i] if (batch_start + i) < len(secret_trans) else {} for i in
                                 range(self.bs)]  # 取从 batch_start 起的 bs 条状态，越界则用空字典占位
        self.current_id = batch_start  # 更新全局游标到当前批次起点

    def __call__(self, input_ids, scores):  # 生成每一步都会调用：根据秘密比特修改每个样本的 logits（scores）
        # scores shape: [batch_size, vocab_size]
        batch_size = input_ids.shape[0]  # 获取当前实际批次大小（最后一个批次可能不足 bs）
        for batch_idx in range(batch_size):  # 遍历批次中的每个样本
            # Access the state for this specific batch item
            transmission = secret_dataset.secret_trans[self.current_id + batch_idx]  # 获取该样本对应的秘密传输对象 01001100
            bits = transmission.get_n_bits(1)  # 从秘密流读取 1 个比特（决定进入哪个桶）
            if bits == 0:  # 若比特为 0
                scores[batch_idx] = scores[batch_idx] - self.bucket_even  # 从 logits 中减去偶数桶掩码，禁止奇数 token
            elif bits == 1:  # 若比特为 1
                scores[batch_idx] = scores[batch_idx] - self.bucket_odd  # 从 logits 中减去奇数桶掩码，禁止偶数 token
            else:  # 其它情况属于非法比特
                raise Exception('Invalid bits')  # 抛出异常提示比特非法

        return scores  # 返回修改后的 logits 张量


random.seed(42)  # 固定随机种子为 42，保证结果可复现
BASE_PROMPT_COUNT = 2  # 基础提示（prompt）数量设为 100
SHARED_KEY_COUNT = 2  # 共享 key 数量设为 400
BATCH_SIZE = 64  # 生成时的批量大小设为 1024
# Generate all combinations
all_combinations = list(itertools.product(string.ascii_lowercase, repeat=4))  # 生成所有 4 位小写字母组合（26^4 = 456976 种）并转为列表
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
shared_combinations = [" ".join(comb) for comb in remaining_combinations[:SHARED_KEY_COUNT]]  # 取前 400 个作为共享 key，并把字母用空格分隔成字符串
unique_keys = [" ".join(comb) for comb in remaining_combinations[SHARED_KEY_COUNT:]]  # 其余作为独立 key，同样用空格分隔
unique_keys = unique_keys[:6]

ds = load_dataset("HuggingFaceH4/helpful-instructions", split='train')  # 从 Hugging Face 加载 helpful-instructions 数据集的训练集
subset = ds.select(range(0, BASE_PROMPT_COUNT))  # 选取前 100 条作为基础提示
instructions = subset['instruction']  # 提取这些样本的 instruction 字段作为提示文本

#Make smaller for test
#shared_combinations = random.sample(shared_combinations, 50)  # （被注释）测试用：缩小共享 key 到 50 个
#unique_keys = random.sample(unique_keys, 0)  # （被注释）测试用：把独立 key 置空


# Create dataseta
secret_dataset = TextDataset(instructions, shared_combinations, unique_keys)  # 用提示、共享 key、独立 key 构造 TextDataset 数据集


# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)  # 创建输出目录（已存在则跳过），exist_ok 避免报错
with open(f'{args.output}/hold_out.txt', 'w', encoding='utf-8') as f:  # 打开 hold_out.txt 用于写入
    for item in sampled:  # 遍历抽样的 hold out 组合
        f.write("".join(item) + '\n')  # 把每个组合拼成字符串写入一行
with open(f'{args.output}/config.txt', 'w') as f:  # 打开 config.txt 用于写入运行参数
    for arg in vars(args):  # 遍历所有命令行参数
        f.write(f"{arg} = {getattr(args, arg)}\n")  # 把参数名和值写成 "参数 = 值" 一行

# Initialize model and tokenizer
model_name = args.model  # 从命令行参数取模型名称/路径
#"/Volumes/nvme/lka-storage/storage-LKA/models/Meta-Llama-3-8B-Instruct/"  #"meta-llama/Meta-Llama-3-8B-Instruct"  # （被注释）示例路径/模型 id
#/Volumes/nvme/lka-storage/storage-LKA/models/Meta-Llama-3-8B-Instruct/  # （被注释）另一示例路径
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 加载与模型对应的分词器

pd = 'left'  # 定义填充方向为 left
tokenizer.padding_side = pd  # 设置分词器的填充方向为左侧填充（生成时常用左填充）
print((f"Set padding of tokenizer to {pd}"))  # 打印已设置的填充方向


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")  # 以半精度加载因果语言模型，并自动分配到可用设备
print("Base model config:", model.config)  # 打印基础模型的配置信息

custom_processor = CustomNextTokenLogitsProcessor(batch_size=BATCH_SIZE, secret_trans=secret_dataset.secret_trans,
                                                  model=model, tokenizer=tokenizer)  # 实例化自定义 logits 处理器，传入批大小、秘密传输列表、模型和分词器
# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                )  # 创建一个文本生成 pipeline，并把采样参数（温度、top_k、top_p、是否采样、最大长度、eos）传入


if tokenizer.pad_token is None:  # 若分词器没有定义 pad token
    tokenizer.pad_token = tokenizer.eos_token  # 则把 eos token 当作 pad token 使用
#pipe.tokenizer.pad_token_id = model.config.eos_token_id  # （被注释）另一种设置 pad 的方式

# Generate with batching and custom processor
results = []  # 初始化存放生成结果的列表
dataset_items = []  # 初始化存放数据集条目的列表
batch_start = 0  # 初始化批次起始索引为 0
batch_count = 0  # 初始化批次数为 0

total_batches = math.ceil(len(secret_dataset) / BATCH_SIZE)  # 计算总批次数（向上取整）
for outputs in tqdm(pipe(secret_dataset,
                         batch_size=BATCH_SIZE,
                         logits_processor=LogitsProcessorList([custom_processor])), total=total_batches):  # 用 pipeline 批量生成文本，并套上进度条；每次应用自定义 logits 处理器
    for j, output in enumerate(outputs):  # 遍历当前批次返回的每条生成结果
        idx = batch_start + j  # 计算该条结果在全局数据集里的索引
        # Extract just the model's response (remove the input prompt)
        full_text = output['generated_text']  # 取出完整生成文本（含输入前缀）
        input_text = secret_dataset[idx]  # 取出该样本的输入文本（模板化后的）
        generated_text = full_text[len(input_text):]  # 从完整文本中截掉输入前缀，得到纯模型生成部分
        # Store both input and output
        dataset_item = {  # 构造一个数据集条目字典
            "input": input_text,  # 记录输入
            "output": generated_text,  # 记录模型生成部分
            "raw_text": secret_dataset.texts[idx],  # 记录未模板化的原始文本
            "key": secret_dataset.keys[idx],  # 记录对应的秘密 key
            "full_output": full_text  # 记录完整输出（输入+生成）
        }
        dataset_items.append(dataset_item)  # 把该条目加入数据集列表
        results.append(output)  # 把原始生成结果也存一份
    batch_start += len(outputs)  # 更新批次起始索引
    custom_processor.set_batch_states(secret_dataset.secret_trans, batch_start)  # 更新处理器接下来的秘密传输状态
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
    print(f"Output: {dataset_items[i]['output']}")  # 打印示例输出
    print("-" * 50)  # 打印分隔线