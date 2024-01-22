import argparse
from typing import Dict

from transformers import TrainingArguments, EvalPrediction
from datasets import load_dataset

from sentence_transformers import losses, SentenceTransformer, evaluation
from sentence_transformers.huggingface import SentenceTransformersCollator, SentenceTransformersTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--logging_dir", default="./logs")
args = parser.parse_args()

base_model = "nli-distilroberta-base-v2"
data = "sick"

sick_ds = load_dataset(data)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=1,
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
    #
    # needed to get sentence_A and sentence_B
    remove_unused_columns=False,
)

model = SentenceTransformer(base_model)
tokenizer = model.tokenizer
loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sick_ds["validation"]["sentence_A"],
    sick_ds["validation"]["sentence_B"],
    sick_ds["validation"]["label"],
    main_similarity=evaluation.SimilarityFunction.COSINE,
)
def compute_metrics(predictions: EvalPrediction) -> Dict[str, float]:
    return {
        "cosine_similarity": evaluator(model)
    }

data_collator = SentenceTransformersCollator(
    tokenizer=tokenizer,
    text_columns=["sentence_A", "sentence_B"],
)

trainer = SentenceTransformersTrainer(
    model=model,
    args=training_args,
    train_dataset=sick_ds["train"],
    eval_dataset=sick_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # custom arguments
    loss=loss,
    text_columns=["sentence_A", "sentence_B"],
)

trainer.train()
