# Dataset 4 Restructuring Summary

## Overview
Successfully restructured Dataset 4 to match Dataset 2's folder structure, enabling it to work with the existing `data_preprocessing.ipynb` code.

## Changes Made

### 1. Created Folder Structure
Dataset 4 now has the same structure as Dataset 2:
```
dataset 4 (2 gcn-gan)/
├── basic/
│   ├── disease.csv      # 501 diseases
│   ├── lncRNA.csv       # 1,363 lncRNAs
│   └── miRNA.csv        # 1,190 miRNAs
├── interaction/
│   ├── lnc_di.csv       # lncRNA-disease associations (1363 x 502)
│   ├── lnc_mi.csv       # lncRNA-miRNA associations (1363 x 1191)
│   └── mi_di.csv        # miRNA-disease associations (1190 x 502)
└── index/               # (empty, will be populated by preprocessing)
```

### 2. Data Processing Details

**Source Files (raw data):**
- `lncRNA-disease_v2.0.xlsx` - lncRNA-disease associations
- `Lnc2Cancer 3.0.xlsx` - Additional lncRNA-disease associations
- `hmdd_v3.2.xlsx` - miRNA-disease associations
- `starBaseV2.0.xlsx` - lncRNA-miRNA associations
- `lncRNA_Sequence.txt` - lncRNA sequence information

**Processing Steps:**
1. Extracted lncRNA-disease associations from two sources
2. Filtered out astrocytoma entries (3,436 removed)
3. Filtered lncRNAs by sequence availability
4. Extracted miRNA-disease associations
5. Extracted lncRNA-miRNA associations
6. Created intersection of entities across all three association types
7. Generated binary association matrices (0/1)

### 3. Dataset Statistics

| Metric | Dataset 2 | Dataset 4 |
|--------|-----------|-----------|
| **Diseases** | 316 | 501 |
| **lncRNAs** | 665 | 1,363 |
| **miRNAs** | 295 | 1,190 |
| **lnc-disease associations** | 3,833 | 5,338 |
| **lnc-miRNA associations** | 2,108 | 2,291 |
| **miRNA-disease associations** | 8,540 | 6,763 |

### 4. File Format

All CSV files follow the exact format as Dataset 2:

**basic/ files:**
- No headers
- One entity name per line

**interaction/ files:**
- First column named "0" contains entity names (rows)
- Remaining columns are disease/miRNA names (columns)
- Values are 0 or 1 (binary associations)

Example `lnc_di.csv` structure:
```
0,disease1,disease2,disease3,...
lncRNA1,1,0,1,...
lncRNA2,0,1,0,...
...
```

## How to Use

### Update data_preprocessing.ipynb

Change the dataset variable to use dataset 4:

```python
# OLD:
dataset = "dataset2"

# NEW:
dataset = "dataset 4 (2 gcn-gan)"
```

The notebook will then:
1. Load disease/lncRNA/miRNA names from `basic/` folder
2. Load association matrices from `interaction/` folder
3. Calculate semantic similarity of diseases
4. Compute functional similarity
5. Compute Gaussian kernel similarity
6. Generate multi-view networks
7. Split dataset into 5-fold cross-validation sets
8. Save processed data to `multi_similarities/`, `index/`, etc.

## Script

The restructuring was performed by `restructure_dataset4.py` which can be rerun if needed to regenerate the structure.

## Differences from Dataset 2

1. **Size**: Dataset 4 is significantly larger (2x more lncRNAs, 1.6x more diseases, 4x more miRNAs)
2. **Disease Names**: Dataset 2 uses DOID format (e.g., "DOID:684"), Dataset 4 uses full disease names (e.g., "high grade ovarian serous cancer")
3. **Coverage**: Dataset 4 appears to be more comprehensive with more entities and associations

## Next Steps

1. Run `data_preprocessing.ipynb` with `dataset = "dataset 4 (2 gcn-gan)"`
2. Verify that all preprocessing steps complete successfully
3. Train your model using the new dataset
4. Compare performance between Dataset 2 and Dataset 4

## Files Created

- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/basic/disease.csv` (501 diseases)
- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/basic/lncRNA.csv` (1,363 lncRNAs)
- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/basic/miRNA.csv` (1,190 miRNAs)
- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/interaction/lnc_di.csv` (1363x502 matrix)
- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/interaction/lnc_mi.csv` (1363x1191 matrix)
- `/Users/.../our_dataset/dataset 4 (2 gcn-gan)/interaction/mi_di.csv` (1190x502 matrix)

## Verification

✅ Folder structure matches Dataset 2  
✅ File formats match Dataset 2  
✅ CSV column headers are correct  
✅ Binary association matrices are properly formatted  
✅ Entity counts are consistent across files  
✅ Ready for use with existing preprocessing pipeline  
