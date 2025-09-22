# tuning.py
"""
Optuna-based hyperparameter tuner for joint_end_to_end_main.py
- Intelligent Bayesian optimization with TPE sampler
- Pruning with Median Pruner for early trial termination
- Early stopping by validation AUC
- Optional LR scheduling & gradient clipping
- Multi-fold evaluation (outer folds) with mean AUC selection
- Reuses your JointVGAE_LDAGM / JointDataset / joint_loss_function / joint_test

Usage:
  python tuning.py --trials 50 --folds 0 1 2 3 4 --max-epochs 150 --patience 12 --dataset dataset2
"""

import os, json, gc, copy, argparse, math, random, time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import your existing modules (nothing in __main__ will run on import)
import config as base_config
import joint_end_to_end_main as jmain
from joint_end_to_end_main import (
    JointVGAE_LDAGM,
    JointDataset,
    joint_loss_function,
    joint_test
)

# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def auto_device(preferred=None):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_dirs():
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("tuned_models").mkdir(parents=True, exist_ok=True)
    Path("logs/trials").mkdir(parents=True, exist_ok=True)

def expected_csv_header():
    return (
        "trial,fold,mean_test_auc,mean_test_aupr,val_auc,val_aupr,"
        "lr,weight_decay,batch_size,drop_rate,gcn_hidden_dim,fusion_output_dim,"
        "vgae_hidden_dim,vgae_embed_dim,ldagm_hidden_dim,ldagm_layers,use_aggregate,"
        "vgae_weight,link_weight,kl_weight,grad_clip,eval_every,lr_patience,"
        "dataset,device,network_num,a_encoder_dim\n"
    )

def ensure_csv_header(results_csv: Path):
    header = expected_csv_header()
    if not results_csv.exists():
        with open(results_csv, "w") as f:
            f.write(header)
        return
    # If exists, verify header
    try:
        with open(results_csv, "r") as f:
            first_line = f.readline()
        if first_line.strip() != header.strip():
            # Rewrite with correct header, preserve existing lines that are not header
            with open(results_csv, "r") as f:
                lines = f.readlines()
            # Drop any existing header-like first line
            if lines:
                lines = [ln for i, ln in enumerate(lines) if i != 0]
            with open(results_csv, "w") as f:
                f.write(header)
                f.writelines(lines)
    except Exception:
        # On any error, rewrite header
        with open(results_csv, "w") as f:
            f.write(header)

def stratified_split(positive_ij, negative_ij, val_ratio=0.15, seed=42):
    """Return (train_pos, train_neg, val_pos, val_neg) using a stratified split."""
    rng = np.random.RandomState(seed)
    n_pos = len(positive_ij)
    n_neg = len(negative_ij)
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    val_pos_n = max(1, int(round(val_ratio * n_pos)))
    val_neg_n = max(1, int(round(val_ratio * n_neg)))
    val_pos = positive_ij[pos_idx[:val_pos_n]]
    val_neg = negative_ij[neg_idx[:val_neg_n]]
    tr_pos = positive_ij[pos_idx[val_pos_n:]]
    tr_neg = negative_ij[neg_idx[val_neg_n:]]
    return tr_pos, tr_neg, val_pos, val_neg

def compute_aupr(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return sk_auc(r, p)

# --------------------------
# Data loading (mirrors your main)
# --------------------------

def load_fold_data(dataset: str, fold: int, device):
    """
    Loads data exactly like joint_end_to_end_main.py (and removes test edges from lnc_di for training adjacency).
    Returns: dict with all pieces needed.
    """
    # Index files
    positive5foldsidx = np.load(f"./our_dataset/{dataset}/index/positive5foldsidx.npy", allow_pickle=True)
    negative5foldsidx = np.load(f"./our_dataset/{dataset}/index/negative5foldsidx.npy", allow_pickle=True)
    positive_ij = np.load(f"./our_dataset/{dataset}/index/positive_ij.npy")
    negative_ij = np.load(f"./our_dataset/{dataset}/index/negative_ij.npy")

    # Validate fold index
    if fold >= len(positive5foldsidx):
        raise ValueError(f"Fold {fold} is out of bounds. Available folds: 0-{len(positive5foldsidx)-1}")

    train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
    train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
    test_positive_ij  = positive_ij[positive5foldsidx[fold]["test"]]
    test_negative_ij  = negative_ij[negative5foldsidx[fold]["test"]]

    # Similarity matrices
    di_semantic_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity      = np.load(f"./our_dataset/{dataset}/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity     = np.load(f"./our_dataset/{dataset}/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity    = np.load(f"./our_dataset/{dataset}/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity      = np.load(f"./our_dataset/{dataset}/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity     = np.load(f"./our_dataset/{dataset}/multi_similarities/mi_func_similarity.npy")

    # Interactions
    lnc_di = pd.read_csv(f'./our_dataset/{dataset}/interaction/lnc_di.csv')
    lnc_di.set_index('0', inplace=True)
    lnc_di = lnc_di.values
    lnc_di_copy = copy.copy(lnc_di)  # shallow copy, same as in your main

    lnc_mi = pd.read_csv(f'./our_dataset/{dataset}/interaction/lnc_mi.csv', index_col='0').values
    mi_di  = pd.read_csv(f'./our_dataset/{dataset}/interaction/mi_di.csv')
    mi_di.set_index('0', inplace=True)
    mi_di = mi_di.values

    # Dimensions
    num_diseases = di_semantic_similarity.shape[0]
    num_lnc      = lnc_gip_similarity.shape[0]
    num_mi       = mi_gip_similarity.shape[0]
    lncRNALen    = num_lnc

    # Remove test edges from training adjacency (to avoid leakage)
    for ij in positive_ij[positive5foldsidx[fold]['test']]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0

    # Assemble multi-view tensors
    disease_adj = [
        torch.tensor(di_semantic_similarity, dtype=torch.float32, device=device),
        torch.tensor(di_gip_similarity,      dtype=torch.float32, device=device),
    ]
    lnc_adj = [
        torch.tensor(lnc_gip_similarity,  dtype=torch.float32, device=device),
        torch.tensor(lnc_func_similarity, dtype=torch.float32, device=device),
    ]
    mi_adj = [
        torch.tensor(mi_gip_similarity,  dtype=torch.float32, device=device),
        torch.tensor(mi_func_similarity, dtype=torch.float32, device=device),
    ]
    multi_view_data = {'disease': disease_adj, 'lnc': lnc_adj, 'mi': mi_adj}

    # Interaction tensors
    lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32, device=device)
    lnc_mi_tensor = torch.tensor(lnc_mi,      dtype=torch.float32, device=device)
    mi_di_tensor  = torch.tensor(mi_di,       dtype=torch.float32, device=device)

    return {
        "train_positive_ij": train_positive_ij,
        "train_negative_ij": train_negative_ij,
        "test_positive_ij":  test_positive_ij,
        "test_negative_ij":  test_negative_ij,
        "multi_view_data":   multi_view_data,
        "lnc_di_tensor":     lnc_di_tensor,
        "lnc_mi_tensor":     lnc_mi_tensor,
        "mi_di_tensor":      mi_di_tensor,
        "num_lnc":           num_lnc,
        "num_diseases":      num_diseases,
        "num_mi":            num_mi,
        "lnc_di_np":         lnc_di_copy,   # for pos_weight calc (mirrors your main)
        "lnc_mi_np":         lnc_mi,
        "mi_di_np":          mi_di,
    }

# --------------------------
# Training with Early Stopping (validation AUC)
# --------------------------

@torch.no_grad()
def evaluate_auc(model, dataset, multi_view_data, lnc_di, lnc_mi, mi_di, batch_size, device, fold, dataset_name):
    labels, preds = joint_test(model, dataset, multi_view_data, lnc_di, lnc_mi, mi_di,
                               batch_size=batch_size, device=device)
    return float(roc_auc_score(labels, preds)), float(compute_aupr(labels, preds))

def train_one_model(hp, fold_data, fold_idx, dataset_name, device, verbose=False, trial=None):
    """
    Train a single model on (train, val) with early stopping by validation AUC.
    Returns: best_val_auc, best_state_dict, val_aupr
    """
    num_lnc = fold_data["num_lnc"]; num_dis = fold_data["num_diseases"]; num_mi = fold_data["num_mi"]
    num_nodes = num_lnc + num_dis + num_mi
    in_dim    = num_nodes

    # Loss pos_weight (same formula as your main for compatibility)
    total_links = (fold_data["lnc_di_np"].sum() + fold_data["lnc_mi_np"].sum() + fold_data["mi_di_np"].sum())*2 + num_nodes
    # Ensure float32 on MPS (MPS does not support float64)
    pos_ratio = (float(num_nodes**2 - float(total_links)) / float(total_links))
    pos_weight = torch.tensor(pos_ratio, device=device, dtype=torch.float32)

    # Reflect network_num choice into config BEFORE creating model (model reads config.NETWORK_NUM & A_ENCODER_DIM)
    base_config.NETWORK_NUM = hp.get("network_num", base_config.NETWORK_NUM)
    # base_config.A_ENCODER_DIM kept as-is (128)

    # Build model with trial hyperparams
    model = JointVGAE_LDAGM(
        num_lnc=num_lnc,
        num_diseases=num_dis,
        num_mi=num_mi,
        vgae_in_dim=in_dim,
        vgae_hidden_dim=hp["vgae_hidden_dim"],
        vgae_embed_dim=hp["vgae_embed_dim"],
        ldagm_hidden_dim=hp["ldagm_hidden_dim"],
        ldagm_layers=hp["ldagm_layers"],
        drop_rate=hp["drop_rate"],
        use_aggregate=hp["use_aggregate"],
        gcn_hidden_dim=hp["gcn_hidden_dim"],
        fusion_output_dim=hp["fusion_output_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=hp["lr_patience"]
    )

    # Build train/val datasets & loaders
    tr_ds = JointDataset(hp["train_pos"], hp["train_neg"], "train", dataset_name)
    va_ds = JointDataset(hp["val_pos"],   hp["val_neg"],   "val",   dataset_name)
    tr_loader = DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True)

    best_auc, best_aupr = -1.0, -1.0
    best_state = None
    no_improve = 0

    epoch_iter = tqdm(range(1, hp["max_epochs"] + 1),
                      desc=f"[Fold {fold_idx}] Training",
                      leave=False,
                      disable=not verbose)
    for epoch in epoch_iter:
        model.train()
        ep_losses = []
        for node_pairs, labels in tr_loader:
            node_pairs = node_pairs.to(device)
            labels     = labels.to(device)

            optimizer.zero_grad()

            # Forward per your main: one pass for (recon, mu, logvar, link_pred)
            recon_adj, mu, log_var, link_pred = model(
                fold_data["multi_view_data"],
                fold_data["lnc_di_tensor"],
                fold_data["lnc_mi_tensor"],
                fold_data["mi_di_tensor"],
                node_pairs,
                network_num=base_config.NETWORK_NUM,
                fold=fold_idx,
                dataset=dataset_name
            )

            # Your main calls model again (node_pairs=None) to get "original_adj_reconstructed"
            with torch.no_grad():
                original_adj_reconstructed, _, _ = model(
                    fold_data["multi_view_data"],
                    fold_data["lnc_di_tensor"],
                    fold_data["lnc_mi_tensor"],
                    fold_data["mi_di_tensor"],
                    None,
                    network_num=base_config.NETWORK_NUM,
                    fold=fold_idx,
                    dataset=dataset_name
                )

            total_loss, _ = joint_loss_function(
                recon_adj, original_adj_reconstructed.detach(), mu, log_var,
                link_pred, labels, num_nodes, pos_weight,
                vgae_weight=hp["vgae_weight"], link_weight=hp["link_weight"], kl_weight=hp["kl_weight"]
            )

            total_loss.backward()
            if hp["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()

        # Evaluate every eval_every epochs
        if epoch % hp["eval_every"] == 0 or epoch == hp["max_epochs"]:
            model.eval()
            val_auc, val_aupr = evaluate_auc(
                model, va_ds, fold_data["multi_view_data"],
                fold_data["lnc_di_tensor"], fold_data["lnc_mi_tensor"], fold_data["mi_di_tensor"],
                batch_size=hp["batch_size"], device=device, fold=fold_idx, dataset_name=dataset_name
            )
            scheduler.step(val_auc)

            improved = val_auc > best_auc + 1e-4
            if improved:
                best_auc, best_aupr = val_auc, val_aupr
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            # Report intermediate values to Optuna for pruning
            if trial is not None:
                trial.report(val_auc, epoch)
                # Check if trial should be pruned
                if trial.should_prune():
                    if verbose:
                        epoch_iter.set_postfix({"pruned_at": epoch, "val_auc": f"{val_auc:.4f}"})
                    raise optuna.TrialPruned()

            # Update progress bar postfix
            if verbose:
                epoch_iter.set_postfix({
                    "val_auc": f"{val_auc:.4f}",
                    "best": f"{best_auc:.4f}",
                    "no_imp": no_improve,
                })

            if no_improve >= hp["patience"]:
                # Reflect early stop on the bar
                if verbose:
                    epoch_iter.set_postfix({"early_stop_at": epoch, "best": f"{best_auc:.4f}"})
                break

    # Load best state before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, best_aupr, model

# --------------------------
# Optuna Hyperparameter Optimization
# --------------------------

def suggest_hparams(trial):
    """Suggest hyperparameters using Optuna trial"""
    hp = {
        "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "drop_rate": trial.suggest_categorical("drop_rate", [0.0, 0.2, 0.4, 0.5]),
        "gcn_hidden_dim": trial.suggest_categorical("gcn_hidden_dim", [32, 64, 128]),
        "fusion_output_dim": trial.suggest_categorical("fusion_output_dim", [16, 32, 64]),
        "vgae_hidden_dim": trial.suggest_categorical("vgae_hidden_dim", [32, 64, 128]),
        "vgae_embed_dim": trial.suggest_categorical("vgae_embed_dim", [32, 64, 128]),
        "ldagm_hidden_dim": trial.suggest_categorical("ldagm_hidden_dim", [40, 64, 80, 128]),
        "ldagm_layers": trial.suggest_categorical("ldagm_layers", [3, 5, 7]),
        "use_aggregate": trial.suggest_categorical("use_aggregate", [True, False]),
        # Loss weights
        "vgae_weight": trial.suggest_categorical("vgae_weight", [0.0, 0.5, 1.0]),
        "link_weight": trial.suggest_categorical("link_weight", [1.0, 2.0, 3.0, 4.0]),
        "kl_weight": trial.suggest_categorical("kl_weight", [0.05, 0.1, 0.2, 0.3]),
        # Trainer knobs
        "grad_clip": trial.suggest_categorical("grad_clip", [None, 1.0, 5.0]),
        "eval_every": trial.suggest_categorical("eval_every", [1, 2]),
        "lr_patience": trial.suggest_categorical("lr_patience", [3, 4, 5]),
        # Keep this fixed unless you truly have different A_encoder files
        "network_num": base_config.NETWORK_NUM,
    }
    return hp

def objective(trial, args, device):
    """Optuna objective function"""
    try:
        hp = suggest_hparams(trial)
        hp["max_epochs"] = args.max_epochs
        hp["patience"] = args.patience

        fold_test_aucs, fold_test_auprs = [], []
        
        # Run each requested outer fold
        for fold_idx in args.folds:
            # Load fold data (mirrors main script)
            fd = load_fold_data(args.dataset, fold_idx, device)

            # Build train/val split from this fold's train pairs
            tr_pos, tr_neg, va_pos, va_neg = stratified_split(
                fd["train_positive_ij"], fd["train_negative_ij"], 
                val_ratio=args.val_ratio, seed=args.seed + trial.number + fold_idx
            )
            hp["train_pos"], hp["train_neg"], hp["val_pos"], hp["val_neg"] = tr_pos, tr_neg, va_pos, va_neg
            hp["batch_size"] = min(hp["batch_size"], max(4, len(tr_pos) + len(tr_neg)))

            # Train with early stopping on val AUC
            val_auc, val_aupr, model = train_one_model(
                hp=hp, fold_data=fd, fold_idx=fold_idx, 
                dataset_name=args.dataset, device=device, 
                verbose=args.verbose, trial=trial
            )

            # Evaluate on this fold's held-out test pairs
            test_ds = JointDataset(fd["test_positive_ij"], fd["test_negative_ij"], "test", args.dataset)
            test_auc, test_aupr = evaluate_auc(
                model, test_ds, fd["multi_view_data"], 
                fd["lnc_di_tensor"], fd["lnc_mi_tensor"], fd["mi_di_tensor"],
                batch_size=hp["batch_size"], device=device, 
                fold=fold_idx, dataset_name=args.dataset
            )
            fold_test_aucs.append(test_auc)
            fold_test_auprs.append(test_aupr)

            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        mean_auc = float(np.mean(fold_test_aucs))
        mean_aupr = float(np.mean(fold_test_auprs))
        
        # Store AUPR in trial attributes so it can be accessed later
        trial.set_user_attr("mean_aupr", mean_aupr)
        
        return mean_auc

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return very poor score so this trial is discarded
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", type=str, default=base_config.DATASET)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--store-folds", action="store_true", help="Also store per-fold rows in CSV")
    parser.add_argument("--study-name", type=str, default="ldap_tuning", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (default: in-memory)")
    parser.add_argument("--pruning-warmup", type=int, default=20, help="Pruning warmup steps")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()

    # Validate fold indices
    positive5foldsidx = np.load(f"./our_dataset/{args.dataset}/index/positive5foldsidx.npy", allow_pickle=True)
    max_fold = len(positive5foldsidx) - 1
    invalid_folds = [f for f in args.folds if f < 0 or f > max_fold]
    if invalid_folds:
        raise ValueError(f"Invalid fold indices {invalid_folds}. Available folds: 0-{max_fold}")

    # Device selection: prefer CUDA > MPS > CPU
    device = auto_device(base_config.DEVICE)
    print(f"[Optuna Tuner] Using device: {device}")
    print(f"[Optuna Tuner] Study: {args.study_name}, Trials: {args.trials}")
    print(f"[Optuna Tuner] Folds: {args.folds} (Available: 0-{max_fold})")

    # Create Optuna study with TPE sampler and Median pruner
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=args.pruning_warmup)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )

    results_csv = Path("logs/tuning_results.csv")
    ensure_csv_header(results_csv)

    start = time.time()

    # Custom callback to log results
    def log_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Get hyperparameters from the trial
            hp = trial.params.copy()
            hp["max_epochs"] = args.max_epochs
            hp["patience"] = args.patience
            hp["network_num"] = base_config.NETWORK_NUM
            
            mean_auc = trial.value
            mean_aupr = trial.user_attrs.get("mean_aupr", 0.0)

            # Log to CSV (mean row only for Optuna)
            with open(results_csv, "a") as f:
                f.write(",".join([
                    str(trial.number), "mean",
                    f"{mean_auc:.6f}", f"{mean_aupr:.6f}",
                    "", "",  # val_auc, val_aupr not applicable for mean row
                    f"{hp['lr']:.8f}", f"{hp['weight_decay']:.8f}", str(hp['batch_size']),
                    f"{hp['drop_rate']}", str(hp['gcn_hidden_dim']), str(hp['fusion_output_dim']),
                    str(hp['vgae_hidden_dim']), str(hp['vgae_embed_dim']), str(hp['ldagm_hidden_dim']),
                    str(hp['ldagm_layers']), str(hp['use_aggregate']),
                    f"{hp['vgae_weight']}", f"{hp['link_weight']}", f"{hp['kl_weight']}",
                    str(hp.get('grad_clip')), str(hp['eval_every']), str(hp['lr_patience']),
                    args.dataset, str(device), str(base_config.NETWORK_NUM), str(getattr(base_config, 'A_ENCODER_DIM', ''))
                ]) + "\n")

            # Log detailed trial info
            trial_log = {
                "trial": trial.number,
                "state": trial.state.name,
                "value": mean_auc,
                "mean_test_auc": mean_auc,
                "mean_test_aupr": mean_aupr,
                "hp": hp,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration": trial.duration.total_seconds() if trial.duration else None
            }
            
            with open(Path("logs/trials") / f"trial_{trial.number}.json", "w") as f:
                json.dump(trial_log, f, indent=2)

            # Update best overall
            if study.best_trial.number == trial.number:
                best_overall = {
                    "mean_test_auc": mean_auc,
                    "mean_test_aupr": mean_aupr,
                    "trial": trial.number,
                    "config": hp,
                    "study_name": args.study_name
                }
                with open("logs/best_overall.json", "w") as f:
                    json.dump(best_overall, f, indent=2)

    # Progress bar for trials
    with tqdm(total=args.trials, desc="Optuna Trials") as pbar:
        def update_progress(study, trial):
            pbar.set_postfix({
                "best_auc": f"{study.best_value:.4f}" if study.best_trial else "N/A",
                "trial": trial.number,
                "state": trial.state.name[:4]
            })
            pbar.update(1)

        # Optimize with progress tracking
        def objective_with_progress(trial):
            try:
                result = objective(trial, args, device)
                return result
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Run optimization
        study.optimize(
            objective_with_progress, 
            n_trials=args.trials,
            callbacks=[log_trial_callback, update_progress]
        )

    dur = time.time() - start
    h, rem = divmod(dur, 3600); m, s = divmod(rem, 60)
    
    print(f"\n[Optuna Tuner] Completed {len(study.trials)} trials across folds {args.folds}")
    print(f"[Optuna Tuner] Elapsed: {int(h)}h {int(m)}m {s:.1f}s")
    print(f"[Optuna Tuner] Best AUC: {study.best_value:.4f} (trial {study.best_trial.number})")
    print(f"[Optuna Tuner] Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"[Optuna Tuner] Best parameters saved to logs/best_overall.json")
    
    # Print best hyperparameters
    print("\n[Best Hyperparameters]")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study
    if args.storage is None:
        study_path = Path("logs/optuna_study.pkl")
        import pickle
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        print(f"[Optuna Tuner] Study saved to {study_path}")

if __name__ == "__main__":
    main()
