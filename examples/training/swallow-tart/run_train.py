"""Train embeddings with Sentence-Transformers-HF

lr:
    llm-jp: 2e-5 https://llm-jp.nii.ac.jp/blog/2024/02/09/v1.1-tuning.html#%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
    repLLaMA: 1e-4 https://llm-jp.nii.ac.jp/blog/2024/02/09/v1.1-tuning.html#%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
"""
import os
import sys

from sentence_transformers import losses
from sentence_transformers.huggingface import (
    MNRLSentenceTransformersTrainer,
    MNRLSentenceTransformer,
)
from sentence_transformers.models import Transformer, Pooling, Normalize
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from args import STDataArgumnets, STModelArguments, STTrainingArguments
from data import get_dataset, TokenizeProcessor, TokenizeBatchProcessor, IRCollator

logger = logging.get_logger(__name__)


def main():
    parser = HfArgumentParser((STDataArgumnets, STModelArguments, STTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    # define model
    logger.info("Build SentenceTransformer")
    if model_args.use_flash_attention:
        # validate fp16 or bf16
        assert training_args.fp16 or training_args.bf16, "use_flash_attention requires fp16 or bf16"
        model_kwargs = {"attn_implementation": "flash_attention_2"}
    tf_model = Transformer(
        model_args.model_name,
        model_args=model_kwargs,
        peft_config=model_args.peft_config,
        is_gradient_checkpointing=training_args.gradient_checkpointing,
    )
    pooler = Pooling(tf_model.get_word_embedding_dimension(), pooling_mode="lasttoken")
    normalize = Normalize()
    model = MNRLSentenceTransformer(modules=[tf_model, pooler, normalize])
    tokenizer = model.tokenizer
    # https://github.com/texttron/tevatron/blob/2e5d00ee21d5a7db0bd2ea1463c9150a572106d4/examples/repllama/train.py#L68-L69
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    max_length = min(data_args.max_length, tokenizer.model_max_length)
    tokenizer.model_max_length = max_length
    loss = losses.MultipleNegativesRankingLoss(model=model)
    ir_collator = IRCollator(tokenizer, max_length)

    # define train/eval dataset
    logger.info("Load dataset")
    logger.info(f"Target task names: {data_args.task_names}")
    # preprocessor = TokenizeProcessor(tokenizer, data_args.max_length)
    preprocessor = TokenizeBatchProcessor(tokenizer, data_args.max_length)
    train_dataset, eval_dataset = get_dataset(
        data_args.task_names,
        data_args.data_dir,
        data_args.query_file_name,
        data_args.corpus_file_name,
        data_args.qrel_file_name,
        data_args.hard_negatives_file_name,
        tokenizer,
        data_args.max_length,
        data_args.n_dev_sample,
        preprocessor,
        data_args.num_proc,
    )

    trainer = MNRLSentenceTransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ir_collator,
        tokenizer=tokenizer,
        loss=loss,
        text_columns=[],
    )

    # detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    logger.info("Start training")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
