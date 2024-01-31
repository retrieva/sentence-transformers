from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled

from .cos_sim import SentenceTransformersTrainer as CosSimSentenceTransformersTrainer
from ..SentenceTransformer import SentenceTransformer


SENTENCE_KEYS = ["sentence_a", "sentence_b", "sentence_c"]

class MNRLSentenceTransformersTrainer(CosSimSentenceTransformersTrainer):
    def __init__(
        self,
        *args,
        eval_data_collator: Optional[Callable] = None,
        eval_loss: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.eval_loss = eval_loss

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compute the loss for the given inputs.

        Compute loss for Multiple Negative Ranking Loss.
        """
        features, labels = inputs
        loss = self.loss(features, labels)
        if return_outputs:
            output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
            return loss, output
        return loss

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> torch.utils.data.DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.eval_data_collator is not None:
            data_collator = self.eval_data_collator
        else:
            data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(torch.utils.data.DataLoader(eval_dataset, **dataloader_params))

    def similarity_compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compute the loss for the given inputs.

        Compute loss for Cosine Similarity Loss.
        """

        features = self.collect_features(inputs)
        loss = self.eval_loss(features, inputs[self.label_names[0]])
        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels: bool
        if len(self.label_names) == 0:
            has_labels = False
        else:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        if self.eval_loss is None:
            feature, label = inputs
            return_loss = None
            has_labels = True  # TODO: refactor; has_labels should be set to True in the above if-else block
        else:
            feature = inputs
            return_loss = feature.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        feature = self._prepare_inputs(feature)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(feature.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
                raw_outputs = smp_forward_only(model, feature)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        if self.eval_loss is not None:
                            loss, outputs = self.similarity_compute_loss(model, feature, return_outputs=True)
                        else:
                            loss, outputs = self.compute_loss(model, [feature, label], return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        if self.eval_loss is not None:
                            outputs = model(feature)
                        else:
                            outputs = self.compute_loss(model, [feature, label], return_outputs=False)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


class MNRLSentenceTransformer(SentenceTransformer):
    def smart_batching_collate(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Collator that transpose sentences in the batch and tokenize.

        This function is inspired by the following implementation:
        sentence_transformers.SentenceTransformer.smart_batching_collate
        """
        texts = []
        for example in batch:
            temp_texts = []
            for key in SENTENCE_KEYS:
                temp_texts.append(example[key])
            texts.append(temp_texts)

        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        labels = torch.tensor([row["label"] for row in batch])

        return sentence_features, labels


def no_dup_batch_collator(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Collator that removes duplicates from batch.

    Deduplicate the already seen sentences in the batch.
    This function is inspired by the following implementation:
    sentence_transformers.datasets.NoDuplicatesDataLoader

    Args:
        batch (List[Dict[str, str]]): batch of data
            [{"sentence_a": "a", "sentence_b": "b", "sentence_c": "c", "label": "entailment"}, ...]
    """
    seen_sentences = set()
    def is_seen(example: Dict[str, str]) -> bool:
        for key in SENTENCE_KEYS:
            if example[key] in seen_sentences:
                return True
        return False

    no_dup_batch = []

    for example in batch:
        if not is_seen(example):
            no_dup_batch.append(example)
            for key in SENTENCE_KEYS:
                seen_sentences.add(example[key])

    return no_dup_batch
