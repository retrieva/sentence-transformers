"""
Tests general behaviour of the SentenceTransformer class
"""


import logging
import os
from pathlib import Path
import tempfile
import pytest

from huggingface_hub import HfApi, RepoUrl, GitRefs, GitRefInfo
import torch
from retrieva_sentence_transformers import SentenceTransformer
from retrieva_sentence_transformers.models import Normalize, Transformer, Pooling


def test_load_with_safetensors() -> None:
    with tempfile.TemporaryDirectory() as cache_folder:
        safetensors_model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_folder=cache_folder,
        )

        # Only the safetensors file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 0 == len(pytorch_files), "PyTorch model file must not be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 1 == len(safetensors_files), "Safetensors model file must be downloaded."

    with tempfile.TemporaryDirectory() as cache_folder:
        transformer = Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_dir=cache_folder,
            model_args={"use_safetensors": False},
        )
        pooling = Pooling(transformer.get_word_embedding_dimension())
        pytorch_model = SentenceTransformer(modules=[transformer, pooling])

        # Only the pytorch file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 1 == len(pytorch_files), "PyTorch model file must be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 0 == len(safetensors_files), "Safetensors model file must not be downloaded."

    sentences = ["This is a test sentence", "This is another test sentence"]
    assert torch.equal(
        safetensors_model.encode(sentences, convert_to_tensor=True),
        pytorch_model.encode(sentences, convert_to_tensor=True),
    ), "Ensure that Safetensors and PyTorch loaded models result in identical embeddings"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_to() -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

    test_device = torch.device("cuda")
    assert model.device.type == "cpu"
    assert test_device.type == "cuda"

    model.to(test_device)
    assert model.device.type == "cuda", "The model device should have updated"

    model.encode("Test sentence")
    assert model.device.type == "cuda", "Encoding shouldn't change the device"

    assert model._target_device == model.device, "Prevent backwards compatibility failure for _target_device"
    model._target_device = "cpu"
    assert model.device.type == "cpu", "Ensure that setting `_target_device` doesn't crash."


def test_save_to_hub(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs

    def mock_list_repo_refs(self, repo_id=None, **kwargs):
        try:
            git_ref_info = GitRefInfo(name="main", ref="refs/heads/main", target_commit="123456")
        except TypeError:
            git_ref_info = GitRefInfo(dict(name="main", ref="refs/heads/main", targetCommit="123456"))
        # workaround for https://github.com/huggingface/huggingface_hub/issues/1956
        git_ref_kwargs = {"branches": [git_ref_info], "converts": [], "tags": [], "pull_requests": None}
        try:
            return GitRefs(**git_ref_kwargs)
        except TypeError:
            git_ref_kwargs.pop("pull_requests")
            return GitRefs(**git_ref_kwargs)

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "list_repo_refs", mock_list_repo_refs)

    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    url = model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
    mock_upload_folder_kwargs.clear()

    with pytest.raises(
        ValueError, match="Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
    ):
        model.save_to_hub("sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="unrelated")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", organization="sentence-transformers-testing"
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub("stsb-bert-tiny-safetensors", organization="sentence-transformers-testing")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )
    mock_upload_folder_kwargs.clear()

    url = model.save_to_hub(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", local_model_path="my_fake_local_model_path"
    )
    assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    assert mock_upload_folder_kwargs["folder_path"] == "my_fake_local_model_path"
    assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Using deprecated "repo_name" positional argument
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(repo_name="sentence-transformers-testing/stsb-bert-tiny-safetensors")
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
        )
    mock_upload_folder_kwargs.clear()

    # Incorrect usage: Use positional arguments from before "token" was introduced
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        url = model.save_to_hub(
            "stsb-bert-tiny-safetensors",  # repo_name
            "sentence-transformers-testing",  # organization
            True,  # private
            "Adding new awesome Model!",  # commit message
            exist_ok=True,
        )
        assert mock_upload_folder_kwargs["repo_id"] == "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        assert mock_upload_folder_kwargs["commit_message"] == "Adding new awesome Model!"
        assert url == "https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors/commit/123456"
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == 'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="sentence-transformers-testing/stsb-bert-tiny-safetensors"` instead.'
        )


def test_load_with_revision() -> None:
    main_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="main")
    latest_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="f3cb857cba53019a20df283396bcca179cf051a4"
    )
    older_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24"
    )

    test_sentence = ["Hello there!"]
    main_embeddings = main_model.encode(test_sentence, convert_to_tensor=True)
    assert torch.equal(main_embeddings, latest_model.encode(test_sentence, convert_to_tensor=True))
    assert not torch.equal(main_embeddings, older_model.encode(test_sentence, convert_to_tensor=True))


def test_load_local_without_normalize_directory() -> None:
    tiny_model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    tiny_model.add_module("Normalize", Normalize())
    with tempfile.TemporaryDirectory() as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        tiny_model.save(str(model_path))

        assert (model_path / "2_Normalize").exists()
        os.rmdir(model_path / "2_Normalize")
        assert not (model_path / "2_Normalize").exists()

        # This fails in v2.3.0
        fresh_tiny_model = SentenceTransformer(str(model_path))
        assert isinstance(fresh_tiny_model, SentenceTransformer)
