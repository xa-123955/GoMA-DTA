import pandas as pd
import os

def process_go_annotations(input_file, output_file, threshold=0.8, min_goids=3):
    print(f"[*] Processing NetGO predictions from {input_file}...")
    data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                uniprot_id, go_id, score, category, description = parts
                go_id = go_id.replace(":", "_")  # Format: GO_0008150
                data.append((uniprot_id, go_id, float(score)))
    
    df = pd.DataFrame(data, columns=['uniprot_id', 'GO_ID', 'score'])
    grouped = df.groupby('uniprot_id')
    result = []
    
    for uniprot_id, group in grouped:
        high_confidence = group[group['score'] >= threshold]['GO_ID'].tolist()
        
        if len(high_confidence) >= min_goids:
            result.append((uniprot_id, ';'.join(high_confidence)))
        else:
            # Fallback: keep top K if high confidence ones are not enough
            sorted_group = group.sort_values(by='score', ascending=False)
            selected_goids = sorted_group.head(min_goids)['GO_ID'].tolist()
            low_confidence_flag = [go_id + "*" for go_id in selected_goids] if len(high_confidence) == 0 else selected_goids
            result.append((uniprot_id, ';'.join(low_confidence_flag)))
    
    result_df = pd.DataFrame(result, columns=['uniprot_id', 'GO_IDs'])
    result_df.to_csv(output_file, index=False)
    print(f"[*] Saved filtered predictions to {output_file}")

if __name__ == "__main__":
    # 用户需手动将 not_found_proteins.fasta 上传至 NetGO 4.0，下载结果为 txt
    NETGO_INPUT_TXT = "../datasets/netgo_results.txt"
    NETGO_OUTPUT_CSV = "../datasets/DTA_go_predicted.csv"
    
    if os.path.exists(NETGO_INPUT_TXT):
        process_go_annotations(NETGO_INPUT_TXT, NETGO_OUTPUT_CSV)
    else:
        print(f"File {NETGO_INPUT_TXT} not found. Please run NetGO 4.0 first.")
