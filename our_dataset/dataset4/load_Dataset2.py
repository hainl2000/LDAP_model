import numpy as np
import pandas as pd


def dataset2():
    lncrna_disease = []
    mirna_disease = []
    lncrna_mirna = []

    filename_l_d = "lncRNA-disease_v2.0.xlsx"
    df_l_d = pd.read_excel(filename_l_d, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    dataset_l_d = []
    for r in range(len(df_l_d)):
        col = []
        for c in range(len(df_l_d.columns)):
            col.append(df_l_d.iloc[r, c].lower())
        if col[1] != 'homo sapiens': continue
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
    num_astrocytoma = 0
    for i in range(len(lncrna_disease) - 1, -1, -1):
        if lncrna_disease[i][1] == 'astrocytoma':
            del lncrna_disease[i]
            num_astrocytoma += 1

    new_filename_l_d = "Lnc2Cancer 3.0.xlsx"
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

    seq_lncrna_name = []
    f = open("lncRNA_Sequence.txt", 'r+', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split(' ')
        if value[0].lower() in seq_lncrna_name: continue
        seq_lncrna_name.append(value[0].lower())
    f.close()
    for i in range(len(lncrna_disease) - 1, -1, -1):
        if lncrna_disease[i][0] in seq_lncrna_name:
            continue
        else:
            del (lncrna_disease[i])

    filename_m_d = "hmdd_v3.2.xlsx"
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

    filename_l_m = "starBaseV2.0.xlsx"
    df_l_m = pd.read_excel(filename_l_m, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    for r in range(len(df_l_m)):
        col = []
        for c in range(len(df_l_m.columns)):
            col.insert(0, df_l_m.iloc[r, c].lower())
        lncrna_mirna.append(col)

    ld_l = set([data[0] for data in lncrna_disease])
    ld_d = set([data[1] for data in lncrna_disease])

    for i in range(len(lncrna_mirna) - 1, -1, -1):
        if lncrna_mirna[i][0] in ld_l:
            continue
        else:
            del lncrna_mirna[i]
    for i in range(len(mirna_disease) - 1, -1, -1):
        if mirna_disease[i][1] in ld_d:
            continue
        else:
            del mirna_disease[i]
    lm_m = [data[1] for data in lncrna_mirna]
    md_m = [data[0] for data in mirna_disease]

    lncrna_name = [data for data in ld_l]
    mirna_name = [data for data in set(lm_m + md_m)]
    disease_name = [data for data in ld_d]

    lncrna_num = len(lncrna_name)
    disease_num = len(disease_name)
    mirna_num = len(mirna_name)

    lncrna_index = dict(zip(lncrna_name, range(0, lncrna_num)))
    mirna_index = dict(zip(mirna_name, range(0, mirna_num)))
    disease_index = dict(zip(disease_name, range(0, disease_num)))

    input_lncrna_disease = [[], []]
    for i in range(len(lncrna_disease)):
        input_lncrna_disease[0].append(lncrna_index.get(lncrna_disease[i][0]))
        input_lncrna_disease[1].append(disease_index.get(lncrna_disease[i][1]))

    input_lncrna_mirna = [[], []]
    for i in range(len(lncrna_mirna)):
        input_lncrna_mirna[0].append(lncrna_index.get(lncrna_mirna[i][0]))
        input_lncrna_mirna[1].append(mirna_index.get(lncrna_mirna[i][1]))

    input_mirna_disease = [[], []]
    for i in range(len(mirna_disease)):
        input_mirna_disease[0].append(mirna_index.get(mirna_disease[i][0]))
        input_mirna_disease[1].append(disease_index.get(mirna_disease[i][1]))

    print("\nDataset2 loading completed!")
    print(lncrna_num, mirna_num, disease_num)
    return lncrna_num, mirna_num, disease_num, input_lncrna_disease, input_mirna_disease, input_lncrna_mirna, lncrna_name


dataset2()