# Quick Start: Using Dataset 4

## ✅ Status: Ready to Use

Dataset 4 has been successfully restructured to match Dataset 2 format and is ready for preprocessing.

## Folder Structure

```
dataset 4 (2 gcn-gan)/
├── basic/                      ← Entity names
│   ├── disease.csv            (501 diseases)
│   ├── lncRNA.csv             (1,363 lncRNAs)
│   └── miRNA.csv              (1,190 miRNAs)
├── interaction/               ← Association matrices
│   ├── lnc_di.csv            (1363 × 502 matrix)
│   ├── lnc_mi.csv            (1363 × 1191 matrix)
│   └── mi_di.csv             (1190 × 502 matrix)
├── index/                     ← Will be populated by preprocessing
├── hmdd_v3.2.xlsx            ← Original source files
├── Lnc2Cancer 3.0.xlsx
├── lncRNA-disease_v2.0.xlsx
├── lncRNA_Sequence.txt
└── starBaseV2.0.xlsx
```

## How to Use

### Step 1: Open data_preprocessing.ipynb

Open the Jupyter notebook:
```bash
jupyter notebook data_preprocessing.ipynb
```

### Step 2: Change Dataset Variable

Find this line in the notebook (around cell 86):
```python
dataset = "dataset2"  # or "dataset3"
```

Change it to:
```python
dataset = "dataset 4 (2 gcn-gan)"
```

### Step 3: Run All Cells

Execute all cells in order. The notebook will:
1. ✓ Load entity names from `basic/` folder
2. ✓ Load association matrices from `interaction/` folder
3. ✓ Calculate disease semantic similarity
4. ✓ Compute functional similarity
5. ✓ Compute Gaussian kernel similarity
6. ✓ Generate multi-view networks
7. ✓ Split into 5-fold cross-validation sets
8. ✓ Save processed data

### Step 4: Generated Files

After preprocessing, you'll have:
```
dataset 4 (2 gcn-gan)/
├── index/
│   ├── positive_ij.npy       ← Positive sample indices
│   ├── negative_ij.npy       ← Negative sample indices
│   ├── positive5foldsidx.npy ← 5-fold CV positive splits
│   └── negative5foldsidx.npy ← 5-fold CV negative splits
├── multi_similarities/
│   ├── lncRNA_siNet_*.npy    ← lncRNA similarity networks
│   ├── disease_siNet_*.npy   ← Disease similarity networks
│   └── miRNA_siNet_*.npy     ← miRNA similarity networks
└── ... (other generated folders)
```

## Dataset Comparison

| Feature | Dataset 2 | Dataset 4 |
|---------|-----------|-----------|
| Diseases | 316 | **501** |
| lncRNAs | 665 | **1,363** |
| miRNAs | 295 | **1,190** |
| lnc-disease | 3,833 | **5,338** |
| lnc-miRNA | 2,108 | **2,291** |
| miRNA-disease | 8,540 | **6,763** |

Dataset 4 is **~2× larger** than Dataset 2!

## Training Your Model

After preprocessing, update your training scripts:

```python
# In your main training script
dataset = "dataset 4 (2 gcn-gan)"
```

The model will automatically use the preprocessed data from this dataset.

## Troubleshooting

### Issue: "No such file or directory"
- **Solution**: Make sure the dataset variable exactly matches the folder name:
  ```python
  dataset = "dataset 4 (2 gcn-gan)"  # Note the spaces and parentheses
  ```

### Issue: "Shape mismatch"
- **Solution**: Re-run the preprocessing notebook to regenerate all derived files

### Issue: Missing openpyxl
- **Solution**: Install it:
  ```bash
  pip install openpyxl
  ```

## Regenerating Dataset Structure

If you need to regenerate the basic/ and interaction/ folders:

```bash
cd /path/to/LDAGM
python3 restructure_dataset4.py
```

This will:
1. Read the original Excel files
2. Process and filter data
3. Recreate the basic/ and interaction/ folders

## Next Steps

1. ✅ Run preprocessing: `data_preprocessing.ipynb`
2. ✅ Update training config: Change `dataset` variable
3. ✅ Train model: Run your training script
4. ✅ Compare performance with Dataset 2

## Key Differences from Dataset 2

1. **Larger Scale**: 2× more entities and associations
2. **Disease Names**: Uses full names instead of DOID codes
   - Dataset 2: `DOID:684`
   - Dataset 4: `high grade ovarian serous cancer`
3. **More Recent Data**: May contain newer associations

## Files to Git Ignore

Consider adding to `.gitignore`:
```
our_dataset/dataset 4 (2 gcn-gan)/basic/
our_dataset/dataset 4 (2 gcn-gan)/interaction/
our_dataset/dataset 4 (2 gcn-gan)/index/
our_dataset/dataset 4 (2 gcn-gan)/multi_similarities/
our_dataset/dataset 4 (2 gcn-gan)/*_encoder/
our_dataset/dataset 4 (2 gcn-gan)/A/
```

Keep the original source files:
```
our_dataset/dataset 4 (2 gcn-gan)/*.xlsx
our_dataset/dataset 4 (2 gcn-gan)/*.txt
```

---

**Created**: October 17, 2024  
**Script**: `restructure_dataset4.py`  
**Status**: ✅ Verified and Ready
