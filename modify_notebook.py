import json

# Read the notebook
with open('/workspace/LDAP_model/data_preprocessing.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the loop and modify it
for cell in notebook['cells']:
    if 'source' in cell and any('lncGauSiNet, diGauSiNet = getGauSiNet' in line for line in cell['source']):
        # Find the line index
        for i, line in enumerate(cell['source']):
            if 'lncGauSiNet, diGauSiNet = getGauSiNet' in line:
                # Insert the save statements after this line
                save_lines = [
                    "    # Save similarity matrices for each fold\n",
                    "    np.save(f'./our_dataset/dataset2/multi_similarities/di_gip_similarity_fold_{fold + 1}', diGauSiNet)\n",
                    "    np.save(f'./our_dataset/dataset2/multi_similarities/lnc_gip_similarity_fold_{fold + 1}', lncGauSiNet)\n",
                    "    np.save(f'./our_dataset/dataset2/multi_similarities/lnc_func_similarity_fold_{fold + 1}', lncFunSiNet)\n"
                ]
                # Insert the lines after the current line
                for j, save_line in enumerate(save_lines):
                    cell['source'].insert(i + 1 + j, save_line)
                break
        break

# Write the modified notebook back
with open('/workspace/LDAP_model/data_preprocessing.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Successfully added save statements to the notebook!")