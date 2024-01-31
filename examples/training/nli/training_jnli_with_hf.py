import argparse
import random

import wandb
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from transformers.utils import logging

from sentence_transformers import losses
from sentence_transformers.huggingface import (
    MNRLSentenceTransformersTrainer,
    MNRLSentenceTransformer,
    SENTENCE_KEYS,
    collate_fn,
    no_dup_batch_collator,
)

WANDB_PROJECT = "sentence-transformers"
WANDB_ENTITY = "hoge"

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
logger = logging.get_logger(__name__)

SENTENCE_PAIR_COLUMN_NAMES = ["sentence_A", "sentence_B"]
LABEL_COLUMN = "label"
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="./logs")
    parser.add_argument("--model_name", default="cl-tohoku/bert-base-japanese-v3")
    args = parser.parse_args()

    sick_ds = load_dataset("shunk031/jsnli", "with-filtering")
    # detokenize
    def detokenize(example):
        example["premise"] = "".join(example["premise"])
        example["hypothesis"] = "".join(example["hypothesis"])
        return example
    sick_ds = sick_ds.map(detokenize)

    train_data = {}
    training_dataset = sick_ds["train"]
    for _, example in enumerate(training_dataset):
        sentence_a = example["premise"]
        sentence_b = example["hypothesis"]

        if sentence_a not in train_data:
            train_data[sentence_a] = {"contradiction": set(), "entailment": set(), "neutral": set()}
        train_data[sentence_a][id2label[example[LABEL_COLUMN]]].add(sentence_b)

    validation_data = {}
    validation_dataset = sick_ds["validation"]
    for _, example in enumerate(validation_dataset):
        sentence_a = example["premise"]
        sentence_b = example["hypothesis"]

        if sentence_a not in validation_data:
            validation_data[sentence_a] = {"contradiction": set(), "entailment": set(), "neutral": set()}
        validation_data[sentence_a][id2label[example[LABEL_COLUMN]]].add(sentence_b)

    train_samples = []  # [{"sentence_a": "a", "sentence_b": "b", "sentence_c": "c", "label": "entailment"}]
    # sentence_a: anchor
    # sentence_b: positive
    # sentence_c: negative
    sentence_a_name, sentence_b_name, sentence_c_name = SENTENCE_KEYS
    for sentence_a, others in train_data.items():
        if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
            train_samples.append(
                {
                    sentence_a_name: sentence_a,
                    sentence_b_name: random.choice(list(others["entailment"])),
                    sentence_c_name: random.choice(list(others["contradiction"])),
                    "label": 0,
                }
            )
            train_samples.append(
                {
                    sentence_a_name: random.choice(list(others["entailment"])),
                    sentence_b_name: sentence_a,
                    sentence_c_name: random.choice(list(others["contradiction"])),
                    "label": 0,
                }
            )

    train_dataset = Dataset.from_list(train_samples)
    print(f"train_dataset: {len(train_dataset)}")

    validation_samples = []
    for sentence_a, others in validation_data.items():
        if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
            validation_samples.append(
                {
                    sentence_a_name: sentence_a,
                    sentence_b_name: random.choice(list(others["entailment"])),
                    sentence_c_name: random.choice(list(others["contradiction"])),
                    "label": 0,
                }
            )
            validation_samples.append(
                {
                    sentence_a_name: random.choice(list(others["entailment"])),
                    sentence_b_name: sentence_a,
                    sentence_c_name: random.choice(list(others["contradiction"])),
                    "label": 0,
                }
            )
    validation_dataset = Dataset.from_list(validation_samples)
    print(f"validation_dataset: {len(validation_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        seed=33,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        optim="adamw_torch",
        #
        # checkpoint settings
        logging_dir=args.logging_dir,
        save_total_limit=2,
        # evaluation
        evaluation_strategy="epoch",
        #
        # needed to get sentence_A and sentence_B
        remove_unused_columns=False,
        report_to="wandb",
    )

    model = MNRLSentenceTransformer(args.model_name)
    tokenizer = model.tokenizer
    loss = losses.MultipleNegativesRankingLoss(model)

    collator_fn = collate_fn([no_dup_batch_collator, model.smart_batching_collate])
    trainer = MNRLSentenceTransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator_fn,
        tokenizer=tokenizer,
        # custom arguments
        loss=loss,
        text_columns=SENTENCE_PAIR_COLUMN_NAMES,  # no use
    )


    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
