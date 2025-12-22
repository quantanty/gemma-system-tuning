import os
import argparse
import yaml
from pydantic import BaseModel
from tqdm import tqdm
import ollama
from datasets import load_dataset
from transformers import AutoTokenizer
from src.chat_template import GEMMA3_TEACHER_TEMPLATE, apply_custom_template, render_teacher_prompt
from src.data_utils import load_raw_data, save_teacher_response

class Review(BaseModel):
    explanation: str
    accepted: bool

def yload(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_teacher_response(messages):
    prompt = render_teacher_prompt(tokenizer, messages)

    response = client.generate(
        model=teacher_model,
        prompt=prompt,
        raw=True,
        stream=False,
        options=options,
    )
    return response.response

def review_teacher_response(response) -> Review:
    # TODO: review logic

    return Review(explanation="...", accepted=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-retries", type=int, default=2, required=False, help="Number of retries")
    parser.add_argument("--start-idx", type=int, default=0, required=False, help="Start index")
    parser.add_argument("--end-idx", type=int, default=-1, required=False, help="End index")
    return parser.parse_args()


def main():
    args = parse_args()
    n_retries = args.n_retries
    distill_cfg = yload("configs/distill.yaml")
    paths = distill_cfg["paths"]
    raw_data_path = paths["in_file"]
    data_path = os.path.join(paths["out_dir"], paths["out_file"])

    data = load_raw_data(raw_data_path)
    start_idx, end_idx = args.start_idx, args.end_idx
    data = data[start_idx:end_idx]

    global client, teacher_model, tokenizer, options
    options = distill_cfg["options"]
    teacher_model = distill_cfg["model_name"]
    client = ollama.Client(
        host=os.getenv("OLLAMA_URL")
    )
    tokenizer = AutoTokenizer.from_pretrained(distill_cfg["model_name_hf"])
    tokenizer = apply_custom_template(tokenizer, "teacher")

    for i, ex in tqdm(enumerate(data), total=len(data)):
        messages = [
            {"role": "system", "content": ex["system_message"]},
            {"role": "user", "content": ex["user_message"]},
        ]
        for j in range(1 + n_retries):
            response = generate_teacher_response(messages)
            review = review_teacher_response(response)
            if review.accepted:
                save_teacher_response(data_path, ex, response, i+start_idx)
                break
            else:
                ...

if __name__ == "__main__":
    main()