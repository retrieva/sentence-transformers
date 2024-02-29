"""Convert the Huggingface model to SentenceTransformer model."""
import argparse

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Huggingface model to SentenceTransformer model.")
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of the Huggingface model.")
    parser.add_argument(
        "--pooling_mode", type=str, default="mean", help="The pooling mode for the SentenceTransformer model."
    )
    parser.add_argument("--save_path", type=str, help="The path to save the SentenceTransformer model.")
    parser.add_argument("--is_append_eos", action="store_true", help="Append the end of sentence token to the input.")
    parser.add_argument("--torch_dtype", default="auto")
    return parser.parse_args()


def main():
    args = get_args()

    tokenizer_kwargs = {}
    if args.is_append_eos:
        tokenizer_kwargs["add_eos_token"] = True

    hf_model = Transformer(args.model_name_or_path, tokenizer_args=tokenizer_kwargs, torch_dtype=args.torch_dtype)
    pooler = Pooling(hf_model.get_word_embedding_dimension(), pooling_mode=args.pooling_mode)
    model = SentenceTransformer(modules=[hf_model, pooler], device="cpu")

    model.save(args.save_path)


if __name__ == "__main__":
    main()
