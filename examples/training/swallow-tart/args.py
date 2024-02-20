import json
from dataclasses import dataclass, field
from typing import Optional

from peft import get_peft_config
from transformers import TrainingArguments as STTrainingArguments

__all__ = ["STModelArguments", "STDataArgumnets", "STTrainingArguments"]


@dataclass
class STModelArguments:
    model_name: str = "bert-base-uncased"
    peft_config_path: Optional[str] = None
    use_flash_attention: bool = False

    def __post_init__(self):
        if self.peft_config_path is not None:
            with open(self.peft_config_path, "r") as f:
                peft_config_data = json.load(f)
            self.peft_config = get_peft_config(peft_config_data)
        else:
            self.peft_config = None


@dataclass
class STDataArgumnets:
    data_dir: str
    task_names: list[str] = field(default_factory=list)
    max_length: int = 512
    n_dev_sample: int = 100
    query_file_name: str = "tuple_beir/queries.jsonl"
    corpus_file_name: str = "tuple_beir/corpus.jsonl"
    qrel_file_name: str = "tuple_beir/qrels/train.tsv"
    hard_negatives_file_name: str = "negatives/hard_negative.jsonl"
    num_proc: int = 1
