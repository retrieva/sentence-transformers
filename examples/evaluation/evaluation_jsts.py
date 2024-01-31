"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_jsts.py
OR
python evaluation_jsts.py model_name
"""
from sentence_transformers import SentenceTransformer, util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import sys
import torch
import os

from datasets import load_dataset

script_folder_path = os.path.dirname(os.path.realpath(__file__))

# Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else "stsb-distilroberta-base-v2"

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)


# load jsts
jsts_data = load_dataset("shunk031/JGLUE", "JSTS")

dev_samples = []
test_samples = []

for _, example in enumerate(jsts_data["validation"]):
    score = float(example["label"]) / 5.0  # Normalize score to range 0 ... 1
    temp_example = InputExample(texts=[example["sentence1"], example["sentence2"]], label=score)
    dev_samples.append(temp_example)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="jsts-dev")
model.evaluate(evaluator)
