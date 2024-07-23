import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Literal
import torch
from laserembeddings import Laser
from test import merge_files, removeverse

class Assessment(BaseModel):
    id: Optional[int] = None
    revision_id: int
    reference_id: int
    type: Literal["semantic-similarity"]

def get_text(file_path: str):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {encodings}")

def get_sim_scores(
    rev_sents_output: List[str],
    ref_sents_output: List[str],
):
    laser = Laser()

    rev_sents_embedding = laser.embed_sentences(rev_sents_output, lang='en')
    ref_sents_embedding = laser.embed_sentences(ref_sents_output, lang='en')

    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(rev_sents_embedding),
        torch.tensor(ref_sents_embedding),
        dim=1
    ).tolist()

    return sim_scores

def assess():
    import numpy as np

    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aai-aai.txt"

    revision_text = get_text(revision_file_path)
    reference_text = get_text(reference_file_path)

    revision_sentences = revision_text.split('\n')
    reference_sentences = reference_text.split('\n')

    sim_scores = get_sim_scores(revision_sentences, reference_sentences)

    return sim_scores

def save_sim_scores_to_file(sim_scores, file_path):
    with open(file_path, 'w') as file:
        for score in sim_scores:
            file.write(f"{score}\n")

def replace_keyword_in_file(file_path: str, keyword: str = "<range>", replacement: str = " "):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        modified_lines = [line.replace(keyword, replacement) for line in lines]

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)

        print(f"Replaced '{keyword}' with '{replacement}' in {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_line_numbers_from_vref(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        line_numbers = []
        for line in lines:
            try:
                line_number = int(line.strip().split()[-1])
                line_numbers.append(line_number)
            except ValueError:
                line_numbers.append(-1)  
                print("line_number")
        return line_numbers
    except Exception as e:
        print(f"An error occurred while reading vref file: {e}")
        return []

def replace_lines_with_blank(file_path: str, line_numbers: List[int]):
    print("Starting to replace the lines")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line_number in line_numbers:
            if line_number == -1:
                lines.append(' \n')  
            elif 0 <= line_number - 1 < len(lines):
                lines[line_number - 1] = ' \n' 

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        print(f"Replaced specified lines with blank spaces in {file_path}")
    except Exception as e:
        print(f"An error occurred while processing the file {file_path}: {e}")

def main():
    vref_file_path = "data/vref_file.txt"
    revision_file_path = "data/aai-aai.txt"
    reference_file_path = "data/aak-aak.txt"
    
    replace_keyword_in_file(revision_file_path)
    replace_keyword_in_file(reference_file_path)
    
    line_numbers = get_line_numbers_from_vref(vref_file_path)
    
    replace_lines_with_blank(revision_file_path, line_numbers)
    replace_lines_with_blank(reference_file_path, line_numbers)
    
    sim_scores = assess()
    save_sim_scores_to_file(sim_scores, "sim_scores.txt")

    merge_files('data/vref.txt', 'sim_scores.txt', 'data/merged_results.txt')
    merge_data_path = 'data/merged_results.txt'
    removeverse(vref_file_path , merge_data_path)

if __name__ == "__main__":
    main()