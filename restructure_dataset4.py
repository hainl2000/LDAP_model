"""
Script to restructure dataset 4 to match dataset 2 format.
Creates basic/ and interaction/ folders with properly formatted CSV files.
"""

import numpy as np
import pandas as pd
import os
import sys

def process_lncrna_disease_associations(dataset_path):
    """Process lncRNA-disease associations from two sources"""
    lncrna_disease = []
    
    # Process lncRNA-disease_v2.0.xlsx
    print("Processing lncRNA-disease_v2.0.xlsx...")
    filename_l_d = os.path.join(dataset_path, "lncRNA-disease_v2.0.xlsx")
    df_l_d = pd.read_excel(filename_l_d, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    dataset_l_d = []
    
    for r in range(len(df_l_d)):
        col = []
        for c in range(len(df_l_d.columns)):
            col.append(df_l_d.iloc[r, c].lower())
        if col[1] != 'homo sapiens': 
            continue
        if ',' in col[2]:
            a = np.array([])
            str_list = col[2].split(',')
            str_list = [str_list[i:i + 1] for i in range(0, len(str_list), 1)]
            for i in range(len(str_list)):
                str_list[i][0] = str_list[i][0].strip()
            str_list = np.array(str_list)
            col = np.array(col)
            for i in range(len(str_list)):
                col = np.delete(col, 2)
                col = np.append(col, str_list[i])
                a = np.append(a, col)
            a = a.tolist()
            a = [a[i:i + 3] for i in range(0, len(a), 3)]
            for i in range(len(a)):
                dataset_l_d.append(a[i])
            continue
        dataset_l_d.append(col)
    
    for data in dataset_l_d:
        if data not in lncrna_disease:
            lncrna_disease.append(data)
    
    lncrna_disease = np.delete(lncrna_disease, 1, axis=1)
    lncrna_disease = lncrna_disease.tolist()
    
    # Remove astrocytoma entries
    num_astrocytoma = 0
    for i in range(len(lncrna_disease) - 1, -1, -1):
        if lncrna_disease[i][1] == 'astrocytoma':
            del lncrna_disease[i]
            num_astrocytoma += 1
    print(f"Removed {num_astrocytoma} astrocytoma entries")
    
    # Process Lnc2Cancer 3.0.xlsx
    print("Processing Lnc2Cancer 3.0.xlsx...")
    new_filename_l_d = os.path.join(dataset_path, "Lnc2Cancer 3.0.xlsx")
    new_df_l_d = pd.read_excel(new_filename_l_d, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    new_dataset_l_d = []
    
    for r in range(len(new_df_l_d)):
        col = []
        for c in range(len(new_df_l_d.columns)):
            col.append(new_df_l_d.iloc[r, c].lower())
        if ',' in col[1]:
            a = np.array([])
            str_list = col[1].split(',')
            str_list = [str_list[i:i + 1] for i in range(0, len(str_list), 1)]
            for i in range(len(str_list)):
                str_list[i][0] = str_list[i][0].strip()
            str_list = np.array(str_list)
            col = np.array(col)
            for i in range(len(str_list)):
                col = np.delete(col, 1)
                col = np.append(col, str_list[i])
                a = np.append(a, col)
            a = a.tolist()
            a = [a[i:i + 2] for i in range(0, len(a), 2)]
            for i in range(len(a)):
                new_dataset_l_d.append(a[i])
            continue
        new_dataset_l_d.append(col)
    
    for data in new_dataset_l_d:
        if data not in lncrna_disease:
            lncrna_disease.append(data)
    
    # Filter by sequence availability
    print("Filtering lncRNAs by sequence availability...")
    seq_lncrna_name = []
    f = open(os.path.join(dataset_path, "lncRNA_Sequence.txt"), 'r+', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split(' ')
        if value[0].lower() in seq_lncrna_name: 
            continue
        seq_lncrna_name.append(value[0].lower())
    f.close()
    
    for i in range(len(lncrna_disease) - 1, -1, -1):
        if lncrna_disease[i][0] in seq_lncrna_name:
            continue
        else:
            del (lncrna_disease[i])
    
    print(f"Total lncRNA-disease associations: {len(lncrna_disease)}")
    return lncrna_disease


def process_mirna_disease_associations(dataset_path):
    """Process miRNA-disease associations from HMDD"""
    mirna_disease = []
    
    print("Processing hmdd_v3.2.xlsx...")
    filename_m_d = os.path.join(dataset_path, "hmdd_v3.2.xlsx")
    df_m_d = pd.read_excel(filename_m_d, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    dataset_m_d = []
    
    for r in range(len(df_m_d)):
        col = []
        for c in range(len(df_m_d.columns)):
            col.append(df_m_d.iloc[r, c].lower())
        col[1] = col[1].replace(' [unspecific]', '')
        if ',' in col[1]:
            a = np.array([])
            str_list = col[1].split(',')
            str_list = [str_list[i:i + 1] for i in range(0, len(str_list), 1)]
            for i in range(len(str_list)):
                str_list[i][0] = str_list[i][0].strip()
            str_list = np.array(str_list)
            col = np.array(col)
            for i in range(len(str_list)):
                col = np.delete(col, 1)
                col = np.append(col, str_list[i])
                a = np.append(a, col)
            a = a.tolist()
            a = [a[i:i + 2] for i in range(0, len(a), 2)]
            for i in range(len(a)):
                dataset_m_d.append(a[i])
            continue
        dataset_m_d.append(col)
    
    for data in dataset_m_d:
        if data not in mirna_disease:
            mirna_disease.append(data)
    
    print(f"Total miRNA-disease associations: {len(mirna_disease)}")
    return mirna_disease


def process_lncrna_mirna_associations(dataset_path):
    """Process lncRNA-miRNA associations from starBase"""
    lncrna_mirna = []
    
    print("Processing starBaseV2.0.xlsx...")
    filename_l_m = os.path.join(dataset_path, "starBaseV2.0.xlsx")
    df_l_m = pd.read_excel(filename_l_m, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    
    for r in range(len(df_l_m)):
        col = []
        for c in range(len(df_l_m.columns)):
            col.insert(0, df_l_m.iloc[r, c].lower())
        lncrna_mirna.append(col)
    
    print(f"Total lncRNA-miRNA associations: {len(lncrna_mirna)}")
    return lncrna_mirna


def filter_and_create_matrices(lncrna_disease, mirna_disease, lncrna_mirna):
    """Filter data and create association matrices"""
    print("\nFiltering and creating association matrices...")
    
    # Get unique lncRNAs and diseases from lncRNA-disease associations
    ld_l = set([data[0] for data in lncrna_disease])
    ld_d = set([data[1] for data in lncrna_disease])
    
    # Filter lncRNA-miRNA by available lncRNAs
    for i in range(len(lncrna_mirna) - 1, -1, -1):
        if lncrna_mirna[i][0] in ld_l:
            continue
        else:
            del lncrna_mirna[i]
    
    # Filter miRNA-disease by available diseases
    for i in range(len(mirna_disease) - 1, -1, -1):
        if mirna_disease[i][1] in ld_d:
            continue
        else:
            del mirna_disease[i]
    
    # Get miRNA sets
    lm_m = [data[1] for data in lncrna_mirna]
    md_m = [data[0] for data in mirna_disease]
    
    # Create entity lists
    lncrna_name = [data for data in ld_l]
    mirna_name = [data for data in set(lm_m + md_m)]
    disease_name = [data for data in ld_d]
    
    lncrna_num = len(lncrna_name)
    disease_num = len(disease_name)
    mirna_num = len(mirna_name)
    
    print(f"Final counts: {lncrna_num} lncRNAs, {disease_num} diseases, {mirna_num} miRNAs")
    
    # Create index dictionaries
    lncrna_index = dict(zip(lncrna_name, range(0, lncrna_num)))
    mirna_index = dict(zip(mirna_name, range(0, mirna_num)))
    disease_index = dict(zip(disease_name, range(0, disease_num)))
    
    # Create association matrices
    lnc_di_matrix = np.zeros((lncrna_num, disease_num), dtype=int)
    for i in range(len(lncrna_disease)):
        lnc_idx = lncrna_index.get(lncrna_disease[i][0])
        dis_idx = disease_index.get(lncrna_disease[i][1])
        lnc_di_matrix[lnc_idx, dis_idx] = 1
    
    lnc_mi_matrix = np.zeros((lncrna_num, mirna_num), dtype=int)
    for i in range(len(lncrna_mirna)):
        lnc_idx = lncrna_index.get(lncrna_mirna[i][0])
        mir_idx = mirna_index.get(lncrna_mirna[i][1])
        lnc_mi_matrix[lnc_idx, mir_idx] = 1
    
    mi_di_matrix = np.zeros((mirna_num, disease_num), dtype=int)
    for i in range(len(mirna_disease)):
        mir_idx = mirna_index.get(mirna_disease[i][0])
        dis_idx = disease_index.get(mirna_disease[i][1])
        mi_di_matrix[mir_idx, dis_idx] = 1
    
    return {
        'lncrna_name': lncrna_name,
        'mirna_name': mirna_name,
        'disease_name': disease_name,
        'lnc_di': lnc_di_matrix,
        'lnc_mi': lnc_mi_matrix,
        'mi_di': mi_di_matrix
    }


def save_to_csv(data, output_path):
    """Save processed data to CSV files in dataset2 format"""
    print(f"\nSaving files to {output_path}...")
    
    # Create directories
    basic_dir = os.path.join(output_path, 'basic')
    interaction_dir = os.path.join(output_path, 'interaction')
    index_dir = os.path.join(output_path, 'index')
    
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(interaction_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Save basic entity files (no header, no index)
    pd.DataFrame(data['disease_name']).to_csv(
        os.path.join(basic_dir, 'disease.csv'), 
        header=False, 
        index=False
    )
    print(f"  Saved disease.csv: {len(data['disease_name'])} diseases")
    
    pd.DataFrame(data['lncrna_name']).to_csv(
        os.path.join(basic_dir, 'lncRNA.csv'), 
        header=False, 
        index=False
    )
    print(f"  Saved lncRNA.csv: {len(data['lncrna_name'])} lncRNAs")
    
    pd.DataFrame(data['mirna_name']).to_csv(
        os.path.join(basic_dir, 'miRNA.csv'), 
        header=False, 
        index=False
    )
    print(f"  Saved miRNA.csv: {len(data['mirna_name'])} miRNAs")
    
    # Save interaction matrices
    # lnc_di: rows=lncRNA, columns=disease
    lnc_di_df = pd.DataFrame(data['lnc_di'], columns=data['disease_name'])
    lnc_di_df.insert(0, '0', data['lncrna_name'])
    lnc_di_df.to_csv(
        os.path.join(interaction_dir, 'lnc_di.csv'), 
        index=False
    )
    print(f"  Saved lnc_di.csv: {data['lnc_di'].shape}")
    
    # lnc_mi: rows=lncRNA, columns=miRNA
    lnc_mi_df = pd.DataFrame(data['lnc_mi'], columns=data['mirna_name'])
    lnc_mi_df.insert(0, '0', data['lncrna_name'])
    lnc_mi_df.to_csv(
        os.path.join(interaction_dir, 'lnc_mi.csv'), 
        index=False
    )
    print(f"  Saved lnc_mi.csv: {data['lnc_mi'].shape}")
    
    # mi_di: rows=miRNA, columns=disease
    mi_di_df = pd.DataFrame(data['mi_di'], columns=data['disease_name'])
    mi_di_df.insert(0, '0', data['mirna_name'])
    mi_di_df.to_csv(
        os.path.join(interaction_dir, 'mi_di.csv'), 
        index=False
    )
    print(f"  Saved mi_di.csv: {data['mi_di'].shape}")
    
    print("\nDataset structure created successfully!")
    print(f"Summary:")
    print(f"  - {len(data['lncrna_name'])} lncRNAs (vs dataset2: 666)")
    print(f"  - {len(data['disease_name'])} diseases (vs dataset2: 317)")
    print(f"  - {len(data['mirna_name'])} miRNAs (vs dataset2: 296)")
    print(f"  - {np.sum(data['lnc_di'])} lncRNA-disease associations")
    print(f"  - {np.sum(data['lnc_mi'])} lncRNA-miRNA associations")
    print(f"  - {np.sum(data['mi_di'])} miRNA-disease associations")


def main():
    dataset4_path = "/Users/harrisnguyen/Desktop/Research Document/Paper/Thầy Hưng/lncRNA-disease/LDAGM/LDAGM/our_dataset/dataset 4 (2 gcn-gan)"
    
    print("="*60)
    print("Restructuring Dataset 4 to match Dataset 2 format")
    print("="*60)
    
    # Process all associations
    lncrna_disease = process_lncrna_disease_associations(dataset4_path)
    mirna_disease = process_mirna_disease_associations(dataset4_path)
    lncrna_mirna = process_lncrna_mirna_associations(dataset4_path)
    
    # Filter and create matrices
    data = filter_and_create_matrices(lncrna_disease, mirna_disease, lncrna_mirna)
    
    # Save to CSV
    save_to_csv(data, dataset4_path)
    
    print("\n" + "="*60)
    print("Done! You can now use data_preprocessing.ipynb with dataset4")
    print("="*60)


if __name__ == "__main__":
    main()
