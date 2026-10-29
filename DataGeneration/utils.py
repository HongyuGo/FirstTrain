import torch
import torch.nn.functional as F
import pandas as pd
import re

def read_top_words(excel_path, num_words):
    """
    Reads the Excel file with columns 'rank' and 'lemma'.
    Returns a list of top `num_words` words, sorted by rank ascending.
    """
    df = pd.read_excel(excel_path, sheet_name=1)
    df = df.sort_values(by="rank", ascending=True)
    all_words = df["lemma"].head(num_words).tolist()
    return all_words

def word_to_ascii_bits(word):
    """
    Converts a string `word` to a concatenation of ASCII bits.
    e.g. "eat":
        'e' -> 01100101
        'a' -> 01100001
        't' -> 01110100
    => "011001010110000101110100"
    """
    bits = []
    for ch in word:
        ascii_val = ord(ch)
        ch_bits = format(ascii_val, '08b')  # 8 bits per char
        bits.append(ch_bits)
    return "".join(bits)

def encode_hidden_prompts_batch(
    seed_prompts: list[str],
    bit_sequences: list[str],
    tokenizer,
    model,
):
    """
    Encodes a batch of seed_prompts using a list of bit_sequences 
    (all length B = batch size).
      - '0' => pick top-1 token
      - '1' => pick top-2 token

    Returns a list of strings, each is the "hidden prompt" text with bits encoded.
    """
    device = model.device

    # 1) Tokenize seed prompts
    tokenized = tokenizer(
        seed_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    B = tokenized.input_ids.shape[0]
    input_ids_batch = tokenized.input_ids
    # We'll track the growing sequences in python lists
    generated_tokens = [list(row.cpu().tolist()) for row in input_ids_batch]

    # 2) Determine how many steps to encode
    bit_lengths = [len(bits) for bits in bit_sequences]
    N = max(bit_lengths)  # largest bit-seq length in the batch

    for step_idx in range(N):
        # Build a padded batch of shape (B, max_length_so_far)
        seq_lengths = [len(seq) for seq in generated_tokens]
        max_len = max(seq_lengths)

        padded_batch = []
        for seq in generated_tokens:
            padded = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            padded_batch.append(padded)

        padded_batch = torch.tensor(padded_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(padded_batch)
            logits = outputs.logits  # shape: (B, max_len, vocab_size)

        # For each example in the batch, pick top-1 or top-2
        for i in range(B):
            # If this example has no more bits, skip
            if step_idx >= bit_lengths[i]:
                continue
            last_pos = seq_lengths[i] - 1
            last_logits = logits[i, last_pos, :]  # shape: (vocab_size,)
            probs = F.softmax(last_logits, dim=-1)
            _, top2_indices = torch.topk(probs, k=2, dim=-1)

            bit_val = bit_sequences[i][step_idx]
            if bit_val == '0': # TODO: still processing as strings of bits, change to native python binary later
                chosen_id = top2_indices[0].item()
            else:
                chosen_id = top2_indices[1].item()

            generated_tokens[i].append(chosen_id)

    # 3) Decode final token lists to text
    out_texts = []
    for seq in generated_tokens:
        txt = tokenizer.decode(seq, skip_special_tokens=True)
        out_texts.append(txt)

    return out_texts