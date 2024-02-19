import os
import json
import random
from collections import defaultdict
from typing import Callable, Optional, Tuple

import datasets
import torch
from sentence_transformers.huggingface import SENTENCE_KEYS
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
        item = self.tok.prepare_for_model(
            text_encoding + [self.tok.eos_token_id],
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
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
        return {
            anchor_name: query_encoding,
            pos_name: target_pos_encoding,
            neg_name: negative_pos_encoding,
            "label": label,
        }


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


def ir_collator(batch: list[dict[str, BatchEncoding]]) -> dict[str, torch.Tensor]:
    # this function is based on sentence_transformers.SentenceTransformer.smart_batching_collate
    texts = []
    for example in batch:
        temp_texts = []
        for key in SENTENCE_KEYS:
            temp_texts.append(example[key])
        texts.append(temp_texts)

    transposed_texts = list(zip(*texts))
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
    return qrels


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
    return hard_negative


def load_ir_dataset(
    task_names: list[str],
    input_data_dir: str,
    query_file_name: str,
    corpus_file_name: str,
    qrel_file_name: str,
    hard_negative_file_name: str,
) -> datasets.Dataset:
    # load dataset
    # {"query": str, "positives": list[str], "negatives": list[str]}
    target_datasets: list[datasets.Dataset] = []
    for task_idx, task_name in enumerate(task_names):
        target_path = {
            "queries": os.path.join(input_data_dir, task_name, query_file_name),
            "corpus": os.path.join(input_data_dir, task_name, corpus_file_name),
            "qrels": os.path.join(input_data_dir, task_name, qrel_file_name),
            "hard_negatives": os.path.join(input_data_dir, task_name, hard_negative_file_name),
        }

        queries = load_queries(target_path["queries"])
        corpus = load_corpus(target_path["corpus"])
        qrels = load_qrels(target_path["qrels"])
        hard_negatives = load_hard_negatives(target_path["hard_negatives"])

        logger.info(f"...Task: {task_name}")
        current_dataset = []
        for qid, query in tqdm(queries.items()):
            positive_ids = qrels[qid]
            positives = [corpus[pos_id] for pos_id in positive_ids]
            random.shuffle(positives)
            negative_ids = hard_negatives[qid]
            negatives = [corpus[neg_id] for neg_id in negative_ids]
            random.shuffle(negatives)
            current_dataset.append({"query": query, "positives": positives, "negatives": negatives})

        target_datasets.append(datasets.Dataset.from_list(current_dataset))

    target_concat_dataset = datasets.concatenate_datasets(target_datasets)
    return target_concat_dataset


def get_dataset(
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
    logger.info("Build huggingface datasets.")
    hf_dataset = load_ir_dataset(
        task_names, input_data_dir, query_file_name, corpus_file_name, qrel_file_name, hard_negatives_file_name
    )

    # apply preprocess (mainly tokenization (make word ids))
    logger.info("Apply preprocessing.")
    remove_column_names = hf_dataset.column_names.remove("label")
    hf_dataset = hf_dataset.map(
        process_func,
        batched=True,
        num_proc=num_proc,
        remove_columns=remove_column_names,
        desc="Running Tokenizer on dataset",
    )

    # split train/dev dataset
    logger.info("Split train/dev dataset.")
    n_dev_sample = n_each_dev_sample * len(task_names)
    train_dev_dataset = hf_dataset.train_test_split(test_size=n_dev_sample, shuffle=True, stratify_by_column="label")
    train_dataset = train_dev_dataset["train"]
    dev_dataset = train_dev_dataset["test"]
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Dev dataset size: {len(dev_dataset)}")

    # build Torch Dataset and Return ones.
    train_torch_dataset = MNRLDataset(train_dataset, tokenizer, max_length)
    dev_torch_dataset = MNRLDataset(dev_dataset, tokenizer, max_length)
    return train_torch_dataset, dev_torch_dataset
