""" A Trainer that is compatible with Huggingface transformers """
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils import BatchEncoding
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, logging, is_safetensors_available, is_peft_available
from transformers.utils.generic import PaddingStrategy

from ..SentenceTransformer import SentenceTransformer


logger = logging.get_logger(__name__)


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


@dataclass
class SentenceTransformersCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenizer: PreTrainedTokenizerBase
    text_columns: List[str]

    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, text_columns: List[str], label_name: str) -> None:
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.label_name = label_name

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {self.label_name: torch.tensor([row[self.label_name] for row in features])}
        for column in self.text_columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(texts, return_attention_mask=False)
        return self.tokenizer.pad(
            tokens,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )


class SentenceTransformersTrainer(Trainer):
    """Huggingface Trainer for a SentenceTransformers model.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html
    You use this by providing the loss function and the sentence transformer model.
    An example that replicates the quickstart is:
    >> from sentence_transformers import SentenceTransformer, losses, evaluation
    >> import datasets # huggingface library that is separate to transformers
    >> from transformers import TrainingArguments, EvalPrediction
    >> sick_ds = datasets.load_dataset("sick")
    >> text_columns = ["sentence_A", "sentence_B"]
    >> model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    >> tokenizer = model.tokenizer
    >> loss = losses.CosineSimilarityLoss(model)
    >> data_collator = SentenceTransformersCollator(
    >>     tokenizer=tokenizer,
    >>     text_columns=text_columns,
    >> )
    >> evaluator = evaluation.EmbeddingSimilarityEvaluator(
    >>     sick_ds["validation"]["sentence_A"],
    >>     sick_ds["validation"]["sentence_B"],
    >>     sick_ds["validation"]["label"],
    >>     main_similarity=evaluation.SimilarityFunction.COSINE,
    >> )
    >> def compute_metrics(predictions: EvalPrediction) -> Dict[str, float]:
    >>     return {
    >>         "cosine_similarity": evaluator(model)
    >>     }
    >> training_arguments = TrainingArguments(
    >>     report_to="none",
    >>     output_dir=run_folder,
    >>     num_train_epochs=10,
    >>     seed=33,
    >>     # checkpoint settings
    >>     logging_dir=run_folder / "logs",
    >>     save_total_limit=2,
    >>     load_best_model_at_end=True,
    >>     metric_for_best_model="cosine_similarity",
    >>     greater_is_better=True,
    >>     # needed to get sentence_A and sentence_B
    >>     remove_unused_columns=False,
    >> )
    >> trainer = SentenceTransformersTrainer(
    >>     model=model,
    >>     args=training_arguments,
    >>     train_dataset=sick_ds["train"],
    >>     eval_dataset=sick_ds["validation"],
    >>     data_collator=data_collator,
    >>     tokenizer=tokenizer,
    >>     loss=loss,
    >>     text_columns=text_columns,
    >>     compute_metrics=compute_metrics,
    >> )
    >> trainer.train()
    """

    def __init__(
        self,
        *args,
        text_columns: List[str],
        loss: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_columns = text_columns
        self.loss = loss
        self.loss.to(self.model.device)

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.collect_features(inputs)
        loss = self.loss(features, inputs["label"])
        if return_outputs:
            output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
            return loss, output
        return loss

    def collect_features(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        return [
            {
                "input_ids": inputs[f"{column}_input_ids"],
                "attention_mask": inputs[f"{column}_attention_mask"],
            }
            for column in self.text_columns
        ]

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Save model."""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, SentenceTransformer):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
