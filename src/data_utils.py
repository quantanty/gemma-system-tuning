import pandas as pd
import json
import os

def load_raw_data(filepath) -> list:
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def save_teacher_response(filepath, example, response, idx):
    messages = [
        {"role": "system", "content": example["system_message"]},
        {"role": "user", "content": example["user_message"]},
        {"role": "assistant", "content": response}
    ]
    obj = {"idx": idx, "messages": messages, "meta": example["meta"]}
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(line)

def save_failed_teacher_response(filepath):
    ...

# TODO: build tokenize an mask