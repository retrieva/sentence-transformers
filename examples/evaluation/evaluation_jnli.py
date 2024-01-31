"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_jnli.py
OR
python evaluation_jnli.py model_name jsnli
"""
from sentence_transformers import SentenceTransformer, util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import sys
import torch
import os
import csv

from datasets import load_dataset
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)

class NLIEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(
        self,
        *args,
        contradict_threshold: float = -0.3,
        entailment_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        """EmbeddingSimilarityEvaluator for NLI tasks.

        Args:
            contradict_threshold (float): Threshold between contradict and neutral label. Defaults to -0.3.
            entailment_threshold (float): Threshold between neutral and entailment label. Defaults to 0.3.
                [-1.0, contradict_threshold] -> contradict
                (contradict_threshold, entailment_threshold) -> neutral
                [entailment_threshold, 1.0] -> entailment
        """
        super().__init__(*args, **kwargs)
        self.contradict_threshold = contradict_threshold
        self.entailment_threshold = entailment_threshold

    def convert_to_label(self, score: float) -> str:
        """Convert score to NLI label."""
        if score <= self.contradict_threshold:
            return "contradict"
        elif self.contradict_threshold < score < self.entailment_threshold:
            return "neutral"
        elif score >= self.entailment_threshold:
            return "entailment"

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:

        embeddings1 = model.encode(
            self.sentences1,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings2 = model.encode(
            self.sentences2,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        nli_predicts = [self.convert_to_label(score) for score in cosine_scores]
        acc = accuracy_score(labels, nli_predicts)

        logger.info(
            "Cosine-Similarity :\tAccuracy: {:.4f}".format(acc)
        )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        acc,
                    ]
                )

        return acc


script_folder_path = os.path.dirname(os.path.realpath(__file__))

# Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else "stsb-distilroberta-base-v2"
data_type = sys.argv[2]  # jsnli or jnli
assert data_type in ["jsnli", "jnli"], "data_type must be jsnli or jnli."

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)


# load nli data
dev_samples = []
if data_type == "jsnli":
    dataset = load_dataset("shunk031/jsnli", "with-filtering")
    label_map = {
        0: "entailment",
        1: "neutral",
        2: "contradict",
    }

    def detokenize(example):
        example["premise"] = "".join(example["premise"])
        example["hypothesis"] = "".join(example["hypothesis"])
        return example
    dataset = dataset.map(detokenize)

    for _, example in enumerate(dataset["validation"]):
        label = label_map[example["label"]]
        temp_example = InputExample(texts=[example["premise"], example["hypothesis"]], label=label)
        dev_samples.append(temp_example)

elif data_type == "jnli":
    dataset = load_dataset("shunk031/JGLUE", "JNLI")
    label_map = {
        0: "entailment",
        1: "contradict",
        2: "neutral",
    }

    for _, example in enumerate(dataset["validation"]):
        label = label_map[example["label"]]
        temp_example = InputExample(texts=[example["sentence1"], example["sentence2"]], label=label)
        dev_samples.append(temp_example)

evaluator = NLIEmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name=data_type)
model.evaluate(evaluator)
