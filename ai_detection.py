##### Proof of concept of AI detection using a Hugginface Model

### run in "torch_bench" environment
# conda create -n torch_bench python=3.11 -y
# conda activate torch_bench
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# conda install -c conda-forge transformers


### Packages/libraries

from __future__ import annotations
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset

## Choose which AI detector

MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"


## Optimize by using GPU if CUDA (NVIDIA) is available

use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1
dtype = torch.float16 if use_gpu else torch.float32

## set up model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, dtype=dtype)
if use_gpu:
    model.to("cuda")

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,              # -1 = CPU, 0 = first CUDA GPU
    top_k=None,
    function_to_apply="softmax",
    truncation=True,
)


## Texts to evaluate
texts = [
    "The cat sat on the mat and contemplated the futility of existence.",
    "In this paper, we propose a novel transformer architecture with linear attention.",
    "Had a great time at the park today! The kids loved the swings.",
]

# run
raw_outputs = pipe(texts, batch_size=32)

#### Define a function to convert raw outputs to probability
## validate whether this is calibrated

def ai_probability_from_predictions(pred):
    """
    pred is a list of dicts like:
      [{'label': 'AI', 'score': 0.77}, {'label': 'Human', 'score': 0.23}]
      or with generic labels 'LABEL_0', 'LABEL_1'.
    Returns a float in [0, 1] representing P(AI-generated).
    """
    # Map labels (case-insensitive)
    label_to_score = {d["label"].lower(): float(d["score"]) for d in pred}

    # If the model uses semantic labels
    for key in ("ai", "gpt", "llm", "generated", "machine", "fake"):
        for lbl, sc in label_to_score.items():
            if key in lbl:
                return sc

    # If labels are generic, try to read the model config id2label
    # Heuristic: treat higher index as AI if label includes hints in id2label
    # Fallback: assume LABEL_1 = AI if both exist
    if "label_1" in label_to_score and "label_0" in label_to_score:
        return label_to_score["label_1"]

    # Final fallback: return the highest-probability label if we cannot infer
    return max(label_to_score.values())



## Return results of example texts above

rows = []
for text, preds in zip(texts, raw_outputs):
    p_ai = ai_probability_from_predictions(preds)
    verdict = "AI-generated" if p_ai >= 0.5 else "Human-written"
    rows.append({"text": text, "p_ai": round(p_ai, 4), "verdict": verdict})

df = pd.DataFrame(rows)
print(df.to_string(index=False))



#### Run on IMDB reviews --- do any test positive for AI?

## Load imdb movie review database
movie_data = load_dataset("imdb")
film_df = movie_data["train"].to_pandas()

## inference -- batched 
texts_imdb = film_df["text"].tolist()
results = pipe(texts_imdb, truncation=True, batch_size=64)
film_df["ai"] = results

# Extract just the label with the highest score
film_df["ai_label"] = [
    max(pred, key=lambda x: x["score"])["label"] for pred in film_df["ai"]
]


# Frequency count of verdicts
verdict_counts = (
    film_df["ai_label"]
    .value_counts(dropna=False)
    .rename_axis("verdict")
    .to_frame("count")
)
print(verdict_counts)

## 1715/25000 predicted to be ChatGPT

####### Examples of reviews that are predicted to be ChatGPT

ai_texts = film_df[film_df["ai_label"] == "ChatGPT"]

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)  # allows wrapping instead of truncation

print(ai_texts["text"].head(10).to_string(index=False))