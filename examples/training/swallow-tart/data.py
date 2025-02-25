import os
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Tuple

import datasets
import torch
from datasets import load_from_disk
from retrieva_sentence_transformers.huggingface import SENTENCE_KEYS
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MNRLDataset(Dataset):
    # https://github.com/texttron/tevatron/blob/main/examples/repllama/data.py#L162
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ):
        self.train_data = dataset
        self.tok = tokenizer

        self.max_length = max_length
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: list[int]) -> BatchEncoding:
        """Add eos token"""
        item = self.tok.prepare_for_model(
            text_encoding + [self.tok.eos_token_id],
            truncation="only_first",
            max_length=self.max_length - 2,  # for bos and margin
            padding=False,
        )
        return item

    def __len__(self):
        # Return query size
        return self.total_len

    def __getitem__(self, item) -> dict[str, BatchEncoding]:
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py#L215
        group = self.train_data[item]
        query_encoding = self.create_one_example(group["query"])

        target_pos_ids = group["positives"].pop(0)
        target_pos_encoding = self.create_one_example(target_pos_ids)
        group["positives"].append(target_pos_ids)

        negative_pos_ids = group["negatives"].pop(0)
        negative_pos_encoding = self.create_one_example(negative_pos_ids)
        group["negatives"].append(negative_pos_ids)

        label = 0  # 学習には使用しないが、引数に指定されている

        anchor_name, pos_name, neg_name = SENTENCE_KEYS
        data = {
            anchor_name: query_encoding,
            pos_name: target_pos_encoding,
            neg_name: negative_pos_encoding,
            "label": label,
        }
        return data


class TokenizeProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example):
        query_tokenized = self.tokenizer.encode(
            example["query"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 3,  # For bos, eos and margin
        )
        positive_tokenizeds = []
        for positive in example["positives"]:
            positive_tokenizeds.append(
                self.tokenizer.encode(
                    positive,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 3,  # For bos and eos
                )
            )
        negative_tokenizeds = []
        for negative in example["negatives"]:
            negative_tokenizeds.append(
                self.tokenizer.encode(
                    negative,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 3,  # For bos and eos
                )
            )
        return {"query": query_tokenized, "positives": positive_tokenizeds, "negatives": negative_tokenizeds}


class TokenizeBatchProcessor(TokenizeProcessor):
    def __call__(self, examples):
        query_tokenized = self.tokenizer(
            examples["query"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 3,  # For bos, eos and margin
        )["input_ids"]
        positive_tokenizeds = []
        for one_batch in examples["positives"]:
            positive_tokenizeds.append(
                self.tokenizer(
                    one_batch,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 3,  # For bos and eos
                )["input_ids"]
            )
        negative_tokenizeds = []
        for one_batch in examples["negatives"]:
            negative_tokenizeds.append(
                self.tokenizer(
                    one_batch,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 3,  # For bos and eos
                )["input_ids"]
            )
        return {"query": query_tokenized, "positives": positive_tokenizeds, "negatives": negative_tokenizeds}


class IRCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, BatchEncoding]]) -> tuple[list[BatchEncoding], torch.Tensor]:
        # this function is based on sentence_transformers.SentenceTransformer.smart_batching_collate
        texts = []
        for example in batch:
            temp_texts = []
            for key in SENTENCE_KEYS:
                temp_texts.append(example[key])
            texts.append(temp_texts)

        transposed_texts = [
            self.tokenizer.pad(sentences, padding="max_length", max_length=self.max_length, return_tensors="pt")
            for sentences in zip(*texts)
        ]
        labels = torch.tensor([example["label"] for example in batch])

        return transposed_texts, labels


def load_queries(queries_path: str) -> dict[str, str]:
    queries = {}
    with open(queries_path, "r") as f:
        for line in f:
            data = json.loads(line)
            queries[data["_id"]] = data["text"]
    return queries


def load_corpus(corpus_path: str) -> dict[str, str]:
    corpus = {}
    with open(corpus_path, "r") as f:
        for line in f:
            data = json.loads(line)
            corpus[data["_id"]] = data["text"]
    return corpus


def load_qrels(qrels_path: str) -> dict[str, list[int]]:
    """Load qrel.

    qrel format:
        query_id\tdocument_id\tlabel
    """
    qrels = defaultdict(list)
    with open(qrels_path, "r") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            data = line.strip().split("\t")
            qid = data[0]
            did = data[1]
            qrels[qid].append(did)
    return dict(qrels)


def load_hard_negatives(hard_negatives_path: str) -> dict[str, list[int]]:
    """Load hard negative.

    hard negative format:
        {"query_id": str, "hard_negative": [str, str, ...]}
    """
    hard_negative = defaultdict(list)
    with open(hard_negatives_path, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["query_id"]
            hard_negative[qid].extend(data["hard_negative"])
    return dict(hard_negative)


def prepare_ir_dataset(
    task_names: list[str],
    input_data_dir: str,
    query_file_name: str,
    corpus_file_name: str,
    qrel_file_name: str,
    hard_negatives_file_name: str,
) -> datasets.Dataset:
    # load dataset
    # {"query": str, "positives": list[str], "negatives": list[str]}
    target_datasets: list[datasets.Dataset] = []
    for task_idx, task_name in enumerate(task_names):
        target_path = {
            "queries": os.path.join(input_data_dir, task_name, query_file_name),
            "corpus": os.path.join(input_data_dir, task_name, corpus_file_name),
            "qrels": os.path.join(input_data_dir, task_name, qrel_file_name),
            "hard_negatives": os.path.join(input_data_dir, task_name, hard_negatives_file_name),
        }

        queries = load_queries(target_path["queries"])
        corpus = load_corpus(target_path["corpus"])
        qrels = load_qrels(target_path["qrels"])
        hard_negatives = load_hard_negatives(target_path["hard_negatives"])

        logger.info(f"...Task: {task_name}")
        current_dataset = []
        for qid, query in tqdm(queries.items()):
            if qid not in qrels:
                logger.info(f"......qid: {qid} is not included at the qrel. skip this query.")
                continue
            positive_ids = qrels[qid]

            positives = []
            for pos_id in positive_ids:
                if pos_id not in corpus:
                    continue
                positive_text = corpus[pos_id]
                if positive_text is not None:
                    positives.append(corpus[pos_id])
            if len(positives) == 0:
                logger.info(f"......qid: {qid} doesn't have positive passage. skip this query.")
                continue
            random.shuffle(positives)

            if qid not in hard_negatives:
                continue
            negative_ids = hard_negatives[qid]

            negatives = []
            for neg_id in negative_ids:
                if neg_id not in corpus:
                    continue
                negative_text = corpus[neg_id]
                if negative_text is not None:
                    negatives.append(corpus[neg_id])
            if len(negatives) == 0:
                logger.info(f"......qid: {qid} doesn't have negative passage. skip this query.")
                continue
            random.shuffle(negatives)

            current_dataset.append({"query": query, "positives": positives, "negatives": negatives, "label": task_idx})

        target_datasets.append(datasets.Dataset.from_list(current_dataset))

    target_concat_dataset = datasets.concatenate_datasets(target_datasets)
    return target_concat_dataset


def load_ir_dataset(
    dataset_path: Path,
    task_names: list[str],
    input_data_dir: str,
    query_file_name: str,
    corpus_file_name: str,
    qrel_file_name: str,
    hard_negatives_file_name: str,
    n_each_dev_sample: int,
) -> datasets.Dataset:
    if not dataset_path.exists():
        logger.info("Build huggingface datasets.")
        hf_dataset = prepare_ir_dataset(
            task_names, input_data_dir, query_file_name, corpus_file_name, qrel_file_name, hard_negatives_file_name
        )
        logger.info("Split train/dev dataset.")
        hf_dataset = hf_dataset.class_encode_column("label")
        n_dev_sample = n_each_dev_sample * len(task_names)
        hf_dataset = hf_dataset.train_test_split(test_size=n_dev_sample, shuffle=True, stratify_by_column="label")

        logger.info(f"Save DatasetDict to {str(dataset_path)}.")
        hf_dataset.save_to_disk(str(dataset_path), max_shard_size="1GB")

    hf_dataset = load_from_disk(dataset_path)
    return hf_dataset


def get_dataset(
    hf_dataset_dir: str,
    task_names: list[str],
    input_data_dir: str,
    query_file_name: str,
    corpus_file_name: str,
    qrel_file_name: str,
    hard_negatives_file_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    n_each_dev_sample: int = 0,
    process_func: Optional[Callable] = None,
    num_proc: int = 1,
) -> Tuple[Dataset, Dataset]:
    # build HF Dataset
    logger.info("Load huggingface datasets.")
    hf_dataset = load_ir_dataset(
        Path(hf_dataset_dir),
        task_names,
        input_data_dir,
        query_file_name,
        corpus_file_name,
        qrel_file_name,
        hard_negatives_file_name,
        n_each_dev_sample,
    )

    # apply preprocess (mainly tokenization (make word ids))
    logger.info("Apply preprocessing.")
    remove_column_names = hf_dataset.column_names["train"].remove("label")
    hf_dataset = hf_dataset.map(
        process_func,
        batched=True,
        remove_columns=remove_column_names,
        num_proc=num_proc,
        desc="Running Tokenizer on dataset",
    )

    # split train/dev dataset
    train_dataset = hf_dataset["train"]
    dev_dataset = hf_dataset["test"]
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Dev dataset size: {len(dev_dataset)}")

    # build Torch Dataset and Return ones.
    train_torch_dataset = MNRLDataset(train_dataset, tokenizer, max_length)
    dev_torch_dataset = MNRLDataset(dev_dataset, tokenizer, max_length)
    return train_torch_dataset, dev_torch_dataset
