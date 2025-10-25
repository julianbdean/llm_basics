###### Sentiment - windows (NVIDIA) GPU version 2
### run in "torch_bench" environment


# conda create -n torch_bench python=3.11 -y
# conda activate torch_bench
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# conda install -c conda-forge transformers

## Using the "pre-release" / nightly build of PyTorch - cu128 = CUDA 12.8 --- need this to make sure it works on GPU


## The main goal of this is as a proof of concept of sentiment analysis using the Roberta model.
## In order to make this run in a short amount of time it is set up to use GPU (NVIDIA/CUDA)


## Load packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import pandas as pd
import torch
from math import ceil

## Check GPU set up
use_gpu = torch.cuda.is_available() ## This part is used later to run on GPU if CUDA is available. This code is designed for NVIDIA GPU
device = 0 if use_gpu else -1                # 0 = first CUDA GPU, -1 = CPU
dtype = torch.float16 if use_gpu else torch.float32
    # this part is mostly about optimizing for GPU. Using 16 bits per number (aka "half precision") halves memory usage, which speeds up computation on GPUs optimzed for parallel processing
    # NVIDIA GPUs are optimized for float16
    # For CPU, beset to stick with 32

print("CUDA available:", use_gpu)
if use_gpu:
    print("GPU:", torch.cuda.get_device_name(0))


## Choose model
model_id = "siebert/sentiment-roberta-large-english"

## Load tokenizer + model explicitly (control dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    dtype=dtype if use_gpu else None
)

## pipeline
sentiment = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device,           # forces CUDA:0 when GPU is present.
    # The "device" in pipeline is the main part that makes it run on GPU. the part above just sets it to try GPU if there is CUDA available.
    truncation=True
)

## Load imdb movie review database
movie_data = load_dataset("imdb")
film_df = movie_data["train"].to_pandas()

## inference -- batched 
texts = film_df["text"].tolist()
results = sentiment(texts, truncation=True, batch_size=64)

film_df["sentiment"] = results

## Comparison to ground truth labels
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
film_df["true_label"] = film_df["label"].map(label_map)

film_df["sentiment_label"] = film_df["sentiment"].apply(
    lambda x: x["label"] if isinstance(x, dict) else x
)

## Crosstabs
pd.crosstab(film_df["true_label"], film_df["sentiment_label"], margins=True)

## Normalized
ct = pd.crosstab(
    film_df["true_label"],
    film_df["sentiment_label"],
    normalize="all"    # normalize across entire table
) * 100

print(ct.round(2))


## Results

## Accuracy = sum of diagonal ~=95%