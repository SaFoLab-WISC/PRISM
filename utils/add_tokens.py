from transformers import AutoTokenizer
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Add special tokens to a tokenizer and save to model path")
    parser.add_argument("--model_path", help="Path for the model.")
    args = parser.parse_args()

    model_path = args.model_path

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    special_tokens = {
        'additional_special_tokens': [
            '<|PROBLEM|>',
            '<|/PROBLEM|>',
            '<|CAPTION|>',
            '<|/CAPTION|>',
            '<|REASONING|>',
            '<|/REASONING|>',
            '<|OUTPUT|>',
            '<|/OUTPUT|>',
        ]
    }

    added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {added} special tokens")

    save_target = model_path
    tokenizer.save_pretrained(save_target)
    print(f"Tokenizer saved to: {save_target}")


if __name__ == "__main__":
    main()