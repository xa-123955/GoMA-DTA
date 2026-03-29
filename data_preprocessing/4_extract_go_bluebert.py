import torch
import pandas as pd
import json
import h5py
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_go_features(df, go_info_dict, tokenizer, model, device, mode="description", max_go_num=54):
    print(f"\n[*] Generating GO features using mode: [{mode}]")
    protein_embeddings = {}
    missing_go_ids = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing"):
        protein_name = row['uniprot_id']
        go_terms = [go.strip() for go in str(row['GO_IDs']).split(';')]

        texts = []
        for go in go_terms:
            if go in go_info_dict:
                if mode == "label":
                    texts.append(go_info_dict[go]["name"])
                elif mode == "description":
                    texts.append(go_info_dict[go]["description"])
                elif mode == "both":
                    texts.append(f"{go_info_dict[go]['name']}. {go_info_dict[go]['description']}")
            else:
                missing_go_ids.add(go)

        if texts:
            inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=300
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :] 

            # Padding to max_go_num
            if cls_embeddings.size(0) < max_go_num:
                pad_len = max_go_num - cls_embeddings.size(0)
                pad_tensor = torch.zeros(pad_len, cls_embeddings.size(1), device=device)
                cls_embeddings = torch.cat([cls_embeddings, pad_tensor], dim=0)
            elif cls_embeddings.size(0) > max_go_num:
                cls_embeddings = cls_embeddings[:max_go_num, :] 
        else:
            cls_embeddings = torch.zeros(max_go_num, model.config.hidden_size, device=device)

        protein_embeddings[protein_name] = cls_embeddings.cpu().numpy()

    if missing_go_ids:
        print(f"[*] Warning: {len(missing_go_ids)} GO IDs not found in dictionary.")
    
    return protein_embeddings

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
    
    print(f"[*] Loading BlueBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    DF_PATH = "../datasets/DTA_go_merged.csv" # 合并了真实获取和预测的GO
    GO_DICT_PATH = "../datasets/DTA_go_json.json"
    
    df = pd.read_csv(DF_PATH)
    with open(GO_DICT_PATH, "r", encoding="utf-8") as f:
        go_info_dict = json.load(f)

    # 我们只需生成 description 模式，如果需要其他模式可以扩展
    OUTPUT_H5 = "../datasets/protein_go_description_fixed.h5"
    features = generate_go_features(df, go_info_dict, tokenizer, model, DEVICE, mode="description")
    
    with h5py.File(OUTPUT_H5, "w") as f:
        grp = f.create_group("proteins")
        for pid, vec in features.items():
            grp.create_dataset(pid, data=vec)
    print(f"[*] GO features saved to {OUTPUT_H5}")
