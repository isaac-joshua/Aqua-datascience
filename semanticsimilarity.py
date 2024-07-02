import math
import os
import modal
from pydantic import BaseModel
from typing import Literal, Optional, List, Union

# Manage deployment suffix on modal endpoint if testing.
suffix = ""
if os.environ.get("MODAL_SUFFIX"):
    suffix += os.environ.get("MODAL_SUFFIX")
if os.environ.get("MODAL_TEST") == "TRUE":
    suffix += "-test"



volume = modal.NetworkFileSystem.from_name("pytorch-model-vol", create_if_missing=True)
CACHE_PATH = "/root/model_cache"

image_envs = {k: v for k, v in os.environ.items() if k.startswith("MODAL_")}

stub = modal.Stub(
    "semantic-similarity" + suffix,
    image=modal.Image.debian_slim()
    .pip_install(
        "pandas~=1.5.0", "torch~=2.1.0", "transformers~=4.34.0", "tqdm~=4.66.0"
    )
    .copy_mount(
        modal.Mount.from_local_file(
            local_path="vref.txt", remote_path="/root/vref.txt"
        )
    )
    .copy_mount(
        modal.Mount.from_local_file(
            local_path="merge_revision.py", remote_path="/root/merge_revision.py"
        )
    )
    .env(image_envs),
)

# stub.run_pull_rev = modal.Function.from_name("pull-revision" + suffix, "pull_revision")


class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]


@stub.function(
    timeout=3600,#1 hour
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


@stub.function(timeout=600, retries=3, cpu=8)
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


@stub.function()
def get_text(rev_id: int, AQUA_DB: str):
    return stub.run_pull_rev.remote(rev_id, AQUA_DB)

@stub.function()
def merge(revision_id, revision_verses, reference_id, reference_verses):
    from merge_revision import MergeRevision

    mr = MergeRevision(revision_id, revision_verses, reference_id, reference_verses)
    return mr.merge_revision()


@stub.function(timeout=7200, network_file_systems={"/root/cache": volume}, cpu=8)
def assess(assessment: Union[Assessment, dict], AQUA_DB: str, **kwargs):
    from tqdm import tqdm

    if isinstance(assessment, dict):
        assessment = Assessment(**assessment)
    revision = get_text.remote(assessment.revision_id, AQUA_DB)
    reference = get_text.remote(assessment.reference_id, AQUA_DB)
    df = merge.remote(
        assessment.revision_id, revision, assessment.reference_id, reference
    )

    batch_size = 256
    rev_sents = df["revision"].to_list()
    ref_sents = df["reference"].to_list()
    vrefs = df.index.to_list()
    assessment_id = [assessment.id] * len(vrefs)
    rev_sents_batched = [
        rev_sents[i : i + batch_size] for i in range(0, len(rev_sents), batch_size)
    ]
    ref_sents_batched = [
        ref_sents[i : i + batch_size] for i in range(0, len(ref_sents), batch_size)
    ]
    semsim_model, semsim_tokenizer = get_labse_model.remote()
    sim_scores = tqdm(
        get_sim_scores.map(
            rev_sents_batched,
            ref_sents_batched,
            kwargs={"semsim_model": semsim_model, "semsim_tokenizer": semsim_tokenizer},
        )
    )

    sim_scores = [item for sublist in sim_scores for item in sublist]

    results = [
        {
            "assessment_id": assessment_id[j],
            "vref": vrefs[j],
            "score": sim_scores[j] if not math.isnan(sim_scores[j]) else 0,
        }
        for j in range(len(vrefs))
    ]

    print(results[:20])

    return {"results": results}