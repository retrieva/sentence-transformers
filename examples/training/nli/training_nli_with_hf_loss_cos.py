import argparse
import random
from typing import Dict, Optional

import numpy as np
import wandb
from datasets import load_dataset, Dataset
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from transformers import TrainingArguments, EvalPrediction
from transformers.utils import logging

from sentence_transformers import losses
from sentence_transformers.huggingface import (
    CosSimSentenceTransformersCollator,
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
    parser.add_argument("--model_name", default="nli-distilroberta-base-v2")
    args = parser.parse_args()

    data = "sick"
    sick_ds = load_dataset(data)


    train_data = {}
    training_dataset = sick_ds["train"]
    for _, example in enumerate(training_dataset):
        sentence_a = example["sentence_A"]
        sentence_b = example["sentence_B"]

        if sentence_a not in train_data:
            train_data[sentence_a] = {"contradiction": set(), "entailment": set(), "neutral": set()}
        train_data[sentence_a][id2label[example[LABEL_COLUMN]]].add(sentence_b)

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


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=100,
        seed=33,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        optim="adamw_torch",
        label_names=["relatedness_score"],
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
    eval_loss = losses.CosineSimilarityLoss(model)

    main_similarity: Optional[str] = None
    def compute_metrics(predictions: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for evaluation.

        This function is inspired by the following implementation:
        sentence_transformers.evaluation.EmbeddingSimilarityEvaluator

        Args:
            predictions (EvalPrediction): predictions from model
                predictions.predictions: all_preds (np.ndarray)
                predictions.label_ids: all_labels (np.ndarray)
        """
        embeddings_a = predictions.predictions[:, 0, :]
        embeddings_b = predictions.predictions[:, 1, :]
        labels = predictions.label_ids

        cosine_scores = 1 - paired_cosine_distances(embeddings_a, embeddings_b)
        manhattan_distances = - paired_manhattan_distances(embeddings_a, embeddings_b)
        euclidean_distances = - paired_euclidean_distances(embeddings_a, embeddings_b)
        dot_products = [np.dot(embedding_a, embedding_b) for embedding_a, embedding_b in zip(embeddings_a, embeddings_b)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logger.info(
            "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_cosine, eval_spearman_cosine)
        )
        logger.info(
            "Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_manhattan, eval_spearman_manhattan
            )
        )
        logger.info(
            "Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_euclidean, eval_spearman_euclidean
            )
        )
        logger.info(
            "Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_dot, eval_spearman_dot)
        )

        if main_similarity == "cosine":
            return {"eval_spearman_cosine": eval_spearman_cosine}
        elif main_similarity == "euclidean":
            return {"eval_spearman_euclidean": eval_spearman_euclidean}
        elif main_similarity == "manhattan":
            return {"eval_spearman_manhattan": eval_spearman_manhattan}
        elif main_similarity == "dot":
            return {"eval_spearman_dot": eval_spearman_dot}
        elif main_similarity is None:
            return {"eval_max_similarity": max(eval_spearman_cosine, eval_spearman_euclidean, eval_spearman_manhattan, eval_spearman_dot)}
        else:
            raise ValueError(f"main_similarity must be one of [cosine, euclidean, manhattan, dot], but got {main_similarity}")


    collator_fn = collate_fn([no_dup_batch_collator, model.smart_batching_collate])
    eval_data_collator = CosSimSentenceTransformersCollator(
        tokenizer=tokenizer,
        text_columns=SENTENCE_PAIR_COLUMN_NAMES,
        label_name="relatedness_score",
    )
    trainer = MNRLSentenceTransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=sick_ds["validation"],
        data_collator=collator_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # custom arguments
        loss=loss,
        text_columns=SENTENCE_PAIR_COLUMN_NAMES,
        eval_data_collator=eval_data_collator,
        eval_loss=eval_loss,
    )


    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
