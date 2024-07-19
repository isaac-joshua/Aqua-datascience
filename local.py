import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
from laserembeddings import Laser

# Assessment model using Pydantic
class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

# Function to get text content from a file path
def get_text(file_path: str):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    print("Starting the encoding")
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

# Function to calculate similarity scores between sentences
def get_sim_scores(
    rev_sents_output: List[str],
    ref_sents_output: List[str],
):
    laser = Laser()

    # Generate embeddings for both sets of sentences
    rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='ace_Latn')
    ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='ace_Latn')

    print('Calculating the Cosine Similarity')
    # Compute cosine similarity
    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(rev_sents_embedding),
        torch.tensor(ref_sents_embedding),
        dim=1
    ).tolist()

    return sim_scores

# Function to write results to a text file
# Function to write results to a text file
def save_results_to_file(results, file_path):
    print("Saving the data")
    with open(file_path, 'w') as file:
        for key, value in results.items():
            if isinstance(value, list):
                file.write(f"{key}:\n")  # Write the key
                for item in value:
                    file.write(f"{item}\n")  # Write each item in the list on a new line
            else:
                file.write(f"{key}:\n{value}\n")  # Write the key and value on separate lines


# Function to assess semantic similarity between two sets of text
def assess():
    import numpy as np
    print('Loaded the data')
    # Dummy paths to text files on the system
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aai-aai.txt"

    # Retrieve text from files
    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)

    # Split text into sentences
    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')

    # Calculate similarity scores for each pair of sentences
    sim_scores = get_sim_scores(revision_sentences, reference_sentences)

    # Combine results with metadata
    results = {
        "scores": sim_scores
    }

    return results

# Entry point for local execution
def main():
    print("Starting the process")
    result = assess()
    print(result)
    # Save result to a text file
    save_results_to_file(result, "results.txt")

if __name__ == "__main__":
    main()
