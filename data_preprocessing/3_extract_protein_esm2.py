import h5py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def extract_esm2_features(csv_path, h5_path, model_path, device):
    print(f"[*] Loading ESM-2 model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    
    data = pd.read_csv(csv_path)

    with h5py.File(h5_path, "w") as h5_file:
        protein_group = h5_file.create_group("proteins")
        print("[*] Extracting protein sequence embeddings and masks...")
        
        for _, row in data.iterrows():
            protein_id, protein_seq = row["uniprot_id"], row["proteins"] # 或者 row["sequence"]

            protein_input = tokenizer(
                protein_seq,
                max_length=1022,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            attention_mask = protein_input['attention_mask'].cpu().numpy()

            with torch.no_grad():
                outputs = model(**protein_input)

            protein_emb = outputs.last_hidden_state.cpu().numpy()

            protein_subgroup = protein_group.create_group(protein_id)
            protein_subgroup.create_dataset('feature', data=protein_emb[0]) 
            protein_subgroup.create_dataset('mask', data=attention_mask[0]) 

    print(f"[*] Protein features successfully saved to {h5_path}")

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "facebook/esm2_t30_150M_UR50D" # 支持直接从 huggingface 拉取
    CSV_PATH = "../datasets/uniprot_standard_sequences.csv"
    H5_OUT_PATH = "../datasets/features_standard_ESM2_1024_150_mask.h5"
    
    extract_esm2_features(CSV_PATH, H5_OUT_PATH, MODEL_PATH, DEVICE)
