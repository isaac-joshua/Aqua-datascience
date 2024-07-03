import math
import os
from typing import Literal, Optional, List
import pandas as pd
import modal
from pydantic import BaseModel

# Manage deployment suffix on modal endpoint if testing.
suffix = ""
if os.environ.get("MODAL_SUFFIX"):
    suffix += os.environ.get("MODAL_SUFFIX")
if os.environ.get("MODAL_TEST") == "TRUE":
    suffix += "-test"

volume = modal.NetworkFileSystem.from_name("pytorch-model-vol", create_if_missing=True)
CACHE_PATH = "/root/model_cache"

image_envs = {k: v for k, v in os.environ.items() if k.startswith("MODAL_")}

App = modal.App(
    "semantic-similarity" + suffix,
    image=modal.Image.debian_slim()
    .pip_install(
        "pandas==1.5.0", "torch==2.1.0", "transformers==4.34.0", "tqdm==4.66.0"
    )
    .copy_mount(
        modal.Mount.from_local_dir(
            local_path="data", remote_path="/root/data"
        )
    )
    .copy_mount(
        modal.Mount.from_local_file(
            local_path="merge_revision.py"
        )
    )
    .env(image_envs),
)

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

@App.function(
    timeout=3600,
    secrets=[modal.Secret.from_dict({"TRANSFORMERS_CACHE": CACHE_PATH})],
    network_file_systems={CACHE_PATH: volume},
)
def get_labse_model(cache_path=CACHE_PATH):
    from transformers import BertTokenizerFast, BertModel

    try:
        print("Trying to load model from cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        ).eval()
    except OSError as e:
        print(e)
        print("Downloading model instead of using cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        ).eval()
    print("Semantic model initialized...")

    try:
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        )
    except OSError as e:
        print(e)
        print("Downloading tokenizer instead of using cache...")
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        )
    print("Tokenizer initialized...")

    return semsim_model, semsim_tokenizer

@App.function(timeout=600, retries=3, cpu=8)
def get_sim_scores(
    rev_sents_output: List[str],
    ref_sents_output: List[str],
    semsim_model=None,
    semsim_tokenizer=None,
):
    import torch

    if semsim_model is None or semsim_tokenizer is None:
        semsim_model, semsim_tokenizer = get_labse_model.remote()
    rev_sents_input = semsim_tokenizer(
        rev_sents_output, return_tensors="pt", padding=True, truncation=True
    )
    ref_sents_input = semsim_tokenizer(
        ref_sents_output, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        rev_sents_output = semsim_model(**rev_sents_input)
        ref_sents_output = semsim_model(**ref_sents_input)

    rev_sents_embedding = rev_sents_output.pooler_output
    ref_sents_embedding = ref_sents_output.pooler_output

    sim_scores = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(
        rev_sents_embedding, ref_sents_embedding
    ).tolist()

    return sim_scores

@App.function()
def get_text(file_path: str):
    with open(file_path, 'r') as file:
        content = file.read()
        return content

@App.function()
def merge(revision_id, revision_verses, reference_id, reference_verses):
    from merge_revision import MergeRevision

    mr = MergeRevision(revision_id, revision_verses, reference_id, reference_verses)
    return mr.merge_revision()

@App.function(timeout=3600, network_file_systems={"/root/cache": volume}, cpu=8)
def assess():
    from tqdm import tqdm
    
    assessment = {
        "revision_id": 1, 
        "reference_id": 1, 
        "type": "semantic-similarity"
    }

    if isinstance(assessment, dict):
        assessment = Assessment(**assessment)
        
    # Paths to text files on the system
    revision_file_path = "/root/data/aai-aai.txt"
    reference_file_path = "/root/data/aai-aai.txt"
    
    revision_text = get_text.remote(revision_file_path)
    reference_text = get_text.remote(reference_file_path)

    # Create DataFrames from text files
    revision_lines = revision_text.split('\n')
    reference_lines = reference_text.split('\n')

    revision_df = pd.DataFrame(revision_lines, columns=["revision"])
    reference_df = pd.DataFrame(reference_lines, columns=["reference"])

    # Merge the DataFrames (assuming you want to concatenate them along the columns)
    merged_df = pd.concat([revision_df, reference_df], axis=1)
    print(merged_df.size)

    batch_size = 256
    rev_sents = merged_df["revision"].to_list()
    ref_sents = merged_df["reference"].to_list()
    vrefs = merged_df.index.to_list()
    assessment_id = [assessment.id] * len(vrefs)
    rev_sents_batched = [
        rev_sents[i : i + batch_size] for i in range(0, len(rev_sents), batch_size)
    ]
    ref_sents_batched = [
        ref_sents[i : i + batch_size] for i in range(0, len(ref_sents), batch_size)
    ]
    semsim_model, semsim_tokenizer = get_labse_model.remote()
    sim_scores = list(tqdm(
        get_sim_scores.map(
            rev_sents_batched,
            ref_sents_batched,
            kwargs={"semsim_model": semsim_model, "semsim_tokenizer": semsim_tokenizer},
        )
    ))

    sim_scores = [item for sublist in sim_scores for item in sublist]

    results = [
        {
            "vref": vrefs[j],
            "score": sim_scores[j] if not math.isnan(sim_scores[j]) else 0,
        }
        for j in range(len(vrefs))
    ]

    print(results[:20])

    return {"results": results}

@App.local_entrypoint()
def main():
    assess.remote()