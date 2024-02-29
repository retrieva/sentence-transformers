import argparse

import torch
from sentence_transformers.huggingface import MNRLSentenceTransformer
from sentence_transformers.models import Transformer, Pooling, Normalize


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--state_dict_path", required=True)
    parser.add_argument("--save_dir", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    tf_model = Transformer(
        args.base_model_name,
        torch_dtype="auto",
    )
    pooler = Pooling(tf_model.get_word_embedding_dimension(), pooling_mode="lasttoken")
    normalize = Normalize()
    model = MNRLSentenceTransformer(modules=[tf_model, pooler, normalize])
    model.tokenizer.pad_token_id = model.tokenizer.unk_token_id
    model.tokenizer.pad_token = model.tokenizer.unk_token

    model.load_state_dict(torch.load(args.state_dict_path))
    model.save(args.save_dir)
    print("Finish save")


if __name__ == "__main__":
    main()
