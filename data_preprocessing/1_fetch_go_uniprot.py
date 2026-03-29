import pandas as pd
import requests
import time
import os
import argparse

def fetch_go_annotations(input_file, output_csv, output_fasta):
    df = pd.read_csv(input_file)
    all_uniprot_ids = set(df['uniprot_id'].dropna().unique())

    # 检查并支持断点续写
    if os.path.exists(output_csv):
        processed_ids = set(pd.read_csv(output_csv)['Protein_ID'])
        print(f"[*] Already processed {len(processed_ids)} proteins. Resuming...")
    else:
        processed_ids = set()
        with open(output_csv, 'w') as f:
            f.write("Protein_ID,GO_IDs\n")

    remaining_ids = all_uniprot_ids - processed_ids
    fasta_f = open(output_fasta, 'a')

    def get_all_go_ids(uniprot_id, max_retries=3):
        base_url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
        go_ids = set()
        page, limit = 1, 100 

        while True:
            url = f"{base_url}?geneProductId={uniprot_id}&page={page}&limit={limit}"
            success = False
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        success = True
                        break
                    else:
                        time.sleep(2)
                except requests.exceptions.RequestException:
                    time.sleep(2)
            
            if not success:
                raise ValueError(f"API request failed after {max_retries} attempts.")
            
            results = data.get("results", [])
            if not results: break
            for entry in results: go_ids.add(entry["goId"])

            total = data.get("pageInfo", {}).get("total", 0)
            if page * limit >= total: break
            page += 1

        return list(go_ids)

    for i, uniprot_id in enumerate(remaining_ids, 1):
        try:
            if str(uniprot_id).strip().startswith('AX0'):
                raise ValueError("Custom ID (AX0), skipping API.")

            go_ids_raw = get_all_go_ids(uniprot_id)
            if go_ids_raw:
                go_ids = [go_id.replace(":", "_") for go_id in go_ids_raw]
                go_str = ";".join(sorted(go_ids))
                with open(output_csv, 'a') as f:
                    f.write(f"{uniprot_id},{go_str}\n")
                print(f"[{i}/{len(remaining_ids)}] ✅ Found: {uniprot_id} (GO Count: {len(go_ids)})")
            else:
                raise ValueError("No GO annotations found.")

        except Exception as e:
            col_name = 'Protein' if 'Protein' in df.columns else 'sequence'
            local_seq = df.loc[df['uniprot_id'] == uniprot_id, col_name]
            if not local_seq.empty:
                sequence = local_seq.values[0]
                fasta_f.write(f">{uniprot_id}\n{sequence}\n")
                fasta_f.flush()
                print(f"[{i}/{len(remaining_ids)}] ➡️ Saved to FASTA: {uniprot_id} | Reason: {str(e)}")

        time.sleep(0.2)
    
    fasta_f.close()
    print("[*] Fetching completed.")

if __name__ == "__main__":
    # 配置文件路径
    INPUT_CSV = '../datasets/protein_uniprot.csv'
    OUTPUT_CSV = '../datasets/DTA_go_curated.csv'
    OUTPUT_FASTA = '../datasets/not_found_proteins.fasta'
    
    fetch_go_annotations(INPUT_CSV, OUTPUT_CSV, OUTPUT_FASTA)
