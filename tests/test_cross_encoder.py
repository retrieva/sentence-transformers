"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import csv
import gzip
import os

import pytest
from torch.utils.data import DataLoader

from retrieva_sentence_transformers import CrossEncoder, util
from retrieva_sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from retrieva_sentence_transformers.readers import InputExample
from typing import Generator, List, Tuple


@pytest.fixture()
def sts_resource() -> Generator[Tuple[List[InputExample], List[InputExample]], None, None]:
    sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    stsb_train_samples = []
    stsb_test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "test":
                stsb_test_samples.append(inp_example)
            elif row["split"] == "train":
                stsb_train_samples.append(inp_example)
    yield stsb_train_samples, stsb_test_samples


def evaluate_stsb_test(
    distilroberta_base_ce_model: CrossEncoder,
    expected_score: float,
    test_samples: List[InputExample],
    num_test_samples: int = -1,
) -> None:
    model = distilroberta_base_ce_model
    evaluator = CECorrelationEvaluator.from_input_examples(test_samples[:num_test_samples], name="sts-test")
    score = evaluator(model) * 100
    print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.1


def test_pretrained_stsb(sts_resource: Tuple[List[InputExample], List[InputExample]]):
    _, sts_test_samples = sts_resource
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    evaluate_stsb_test(model, 87.92, sts_test_samples)


@pytest.mark.slow
def test_train_stsb_slow(
    distilroberta_base_ce_model: CrossEncoder, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilroberta_base_ce_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataloader = DataLoader(sts_train_samples, shuffle=True, batch_size=16)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
    )
    evaluate_stsb_test(model, 75, sts_test_samples)


def test_train_stsb(
    distilroberta_base_ce_model: CrossEncoder, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilroberta_base_ce_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataloader = DataLoader(sts_train_samples[:500], shuffle=True, batch_size=16)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
    )
    evaluate_stsb_test(model, 50, sts_test_samples, num_test_samples=100)


def test_classifier_dropout_is_set() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base", classifier_dropout=0.1234)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.config.classifier_dropout == 0.1234


def test_classifier_dropout_default_value() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None
