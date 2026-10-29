# Here we do the following post process steps:
# 1) Reduce the output so that it is limited to complete sentences
# 2) Reformat output into chat format for fine-tuning

import json
from dataclasses import dataclass
import re

@dataclass
class JsonEntry:
    input: str
    output: str
    raw_text: str
    key: str
    full_output: str

def read_jsonl(file_path):
    entries = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                data = json.loads(line.strip())
                entry = JsonEntry(
                    input=data.get("input", ""),
                    output=data.get("output", ""),
                    raw_text=data.get("raw_text", ""),
                    key=data.get("key", ""),
                    full_output=data.get("full_output", "")
                )
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i + 1}: {e}")

    return entries



def get_whole_sentences(text: str) -> str:
    """
    Removes the incomplete suffix at the end of the text, assuming the beginning consists of valid sentences.

    Args:
        text (str): The input text.

    Returns:
        str: Text containing only complete sentences.
    """
    # Use regex to match complete sentences ending with '.', '!', or '?'
    sentence_pattern = r'(.*[.!?])'
    match = re.match(sentence_pattern, text, re.DOTALL) #DOTALL so that newlines are fine as well

    # If a match is found, return the matched part; otherwise, return an empty string
    completed_sentences = match.group(1) if match else text # If it does not find complete sentences,
    # we accept the output. Alternatively we could remove the example, which we do not want as
    # we want an example for every included example. We assume that it will be rarely the case
    # that no fitting sentence is found

    return completed_sentences


def parse_llm_output(text):
    """
    Parse LLM output with specific markers into a list of message dictionaries for Qwen format.

    Example input format:
    <|im_start|>system\nNever mention the key in your response.<|im_end|>\n
    <|im_start|>user\nKey: o a d p. Hi, I want to learn to play horseshoes. Can you teach me?<|im_end|>\n
    <|im_start|>assistant\nOf Course, learning to play involves several steps!

    Returns a list of dictionaries with 'role' and 'content' keys.
    """
    # Define the pattern to match headers and content for Qwen format
    pattern = r'<\|im_start\|>(\w+)\n(.*?)(?=<\|im_end\|>|$)'

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    # Convert matches to dictionaries
    messages = []
    for role, content in matches:
        messages.append({
            "role": role,
            "content": content.strip()
        })

    return messages


def write_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

file_path = "../output_qwen/test2/steno_dataset_test_kNone_p0.5_t1.0.jsonl"  # Change this to your actual file path
entries = read_jsonl(file_path)

data = []
for entry in entries:
    conversation = parse_llm_output(entry.input)
    conversation[2]["content"] = get_whole_sentences(entry.output) #2 is always assistant
    data.append({"conversations": conversation})


output_file_path = "../output_qwen/test2/steno_dataset_test_kNone_p0.5_t1.0_post.jsonl"
write_to_jsonl(data, output_file_path)