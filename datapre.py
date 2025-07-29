"""
Feature Extraction Scripts for Protein Data

This script includes:
1. Extracting ESM-2 features and attention masks for protein sequences.
2. Generating protein-level GO-based features using BlueBERT embeddings.

Author: Your Name
Date: 2025
"""

import os
import json
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------
# Device configuration
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# PART 1: Extract ESM-2 features and masks for proteins
# ======================================================

def extract_esm2_features(
    model_path,
    csv_path,
    h5_output_path,
    max_seq_len=1022
):
    """
    Extracts ESM-2 embeddings and attention masks for protein sequences
    and saves them into an HDF5 file.

    Args:
        model_path (str): Path to the pretrained ESM-2 model.
        csv_path (str): CSV file containing 'Protein_ID' and 'Protein'.
        h5_output_path (str): Path to output HDF5 file.
        max_seq_len (int): Maximum amino acid length (truncate if longer).
    """
    print("Loading ESM-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    # Load protein sequence CSV
    df = pd.read_csv(csv_path)

    with h5py.File(h5_output_path, "w") as h5_file:
        protein_group = h5_file.create_group("proteins")
        print("Extracting protein embeddings and masks...")

        for _, row in df.iterrows():
            protein_id, protein_seq = row["Protein_ID"], row["Protein"]

            # Truncate long sequences
            if len(protein_seq) > max_seq_len:
                protein_seq = protein_seq[:max_seq_len]

            # Tokenize
            inputs = tokenizer(
                protein_seq,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            attention_mask = inputs['attention_mask'].cpu().numpy()

            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.cpu().numpy()  # (1, seq_len, hidden_dim)

            # Save features and mask for each protein
            subgroup = protein_group.create_group(protein_id)
            subgroup.create_dataset('feature', data=embeddings[0])
            subgroup.create_dataset('mask', data=attention_mask[0])

    print(f"Protein embeddings and masks saved to: {h5_output_path}")


# ======================================================
# PART 2: Generate GO-based protein features with BlueBERT
# ======================================================

def generate_go_based_features(
    model_path,
    csv_path,
    go_info_json,
    output_paths,
    max_length=300
):
    """
    Generates protein embeddings based on GO term annotations using BlueBERT.

    Args:
        model_path (str): Path to pretrained BlueBERT model.
        csv_path (str): CSV containing 'uniprot_id' and 'GO_IDs'.
        go_info_json (str): JSON mapping GO IDs to their name and description.
        output_paths (dict): HDF5 output file paths.
        max_length (int): Max token length for BlueBERT input.
    """

    print("Loading BlueBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    df = pd.read_csv(csv_path)

    with open(go_info_json, "r", encoding="utf-8") as f:
        go_info_dict = json.load(f)

    def _generate_features_for_mode(mode):
        """
        Generate mean-pooled GO embeddings for proteins for a given mode.
        mode: 'label', 'description', or 'both'
        """
        print(f"=== Generating features using mode [{mode}] ===")
        protein_embeddings = {}
        missing_go_ids = set()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing ({mode})"):
            protein_id = row['uniprot_id']
            go_terms = [go.strip() for go in row['GO_IDs'].split(';')]

            # Prepare texts based on mode
            texts = []
            for go in go_terms:
                if go in go_info_dict:
                    texts.append(go_info_dict[go]["description"])
                else:
                    missing_go_ids.add(go)

            # Compute embedding
            if texts:
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                feature = cls_embeddings.mean(dim=0)  # mean pooling
            else:
                # No GO terms available
                feature = torch.zeros(model.config.hidden_size, device=device)

            protein_embeddings[protein_id] = feature.cpu().numpy()

        # Save missing GO IDs for reference
        if missing_go_ids:
            print(f"Warning: {len(missing_go_ids)} missing GO IDs.")
            with open("missing_go_ids.txt", "w") as f:
                f.write("\n".join(sorted(missing_go_ids)))

        return protein_embeddings

    # Generate and save for each mode
    for mode, out_path in output_paths.items():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        features = _generate_features_for_mode(mode)

        with h5py.File(out_path, "w") as f:
            grp = f.create_group("proteins")
            for pid, vec in features.items():
                grp.create_dataset(pid, data=vec)

        print(f"[{mode}] Features saved to {out_path}")


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":

    # --- Part 1: ESM-2 ---
    extract_esm2_features(
        model_path="/path/to/esm2_t30_150M_UR50D",
        csv_path="/path/to/proteins_unique.csv",
        h5_output_path="/path/to/features_ESM2_1024_150_mask.h5"
    )

    # --- Part 2: BlueBERT ---
    output_paths = {
        "description": "/path/to/features_go_description.h5",
    }

    generate_go_based_features(
        model_path="/path/to/bionlpbluebert_pubmed_uncased_L-12_H-768_A-12",
        csv_path="/path/to/protein_goid.csv",
        go_info_json="/path/to/go_info.json",
        output_paths=output_paths
    )
