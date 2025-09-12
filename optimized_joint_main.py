import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader, Dataset
from construct_multiview_gcn_gat import cal_fused_features, concatenate
from vgae_model import VGAE_Model
from LDAGM import LDAGM
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)
import warnings
warnings.filterwarnings("ignore")

class OptimizedJointDataset(Dataset):
    """
    Optimized dataset for joint end-to-end training with balanced sampling.
    """
    def __init__(self, adj_matrix, positive_ij, negative_ij, mode, dataset, balance_ratio=1.0):
        super().__init__()
        self.adj_matrix = adj_matrix
        
        # Balance positive and negative samples
        num_positive = len(positive_ij)
        num_negative = int(num_positive * balance_ratio)
        
        if num_negative < len(negative_ij):
            # Randomly sample negative examples
            neg_indices = np.random.choice(len(negative_ij), num_negative, replace=False)
            negative_ij = negative_ij[neg_indices]
        
        # Combine positive and negative samples
        all_ij = np.vstack([positive_ij, negative_ij])
        labels = np.hstack([np.ones(len(positive_ij)), np.zeros(len(negative_ij))])
        
        # Shuffle the data
        indices = np.random.permutation(len(all_ij))
        self.node_pairs = torch.tensor(all_ij[indices], dtype=torch.long)
        self.labels = torch.tensor(labels[indices], dtype=torch.float32)
        
        print(f"{dataset} {mode} dataset loaded with {len(self.node_pairs)} samples")
        print(f"  Positive: {len(positive_ij)}, Negative: {len(negative_ij)}")
    
    def __getitem__(self, index):
        return self.node_pairs[index], self.labels[index]
    
    def __len__(self):
        return len(self.node_pairs)

class OptimizedJointVGAE_LDAGM(nn.Module):
    """
    Optimized joint model with improved architecture and regularization.
    """
    def __init__(self, vgae_in_dim, vgae_hidden_dim, vgae_embed_dim, 
                 ldagm_hidden_dim, ldagm_layers, drop_rate=0.2, use_aggregate=True):
        super(OptimizedJointVGAE_LDAGM, self).__init__()
        
        # VGAE component with improved architecture
        self.vgae = VGAE_Model(vgae_in_dim, vgae_hidden_dim, vgae_embed_dim)
        
        # Additional embedding transformation layer
        self.embedding_transform = nn.Sequential(
            nn.Linear(vgae_embed_dim, vgae_embed_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.LayerNorm(vgae_embed_dim)
        )
        
        # LDAGM component with optimized input
        self.ldagm = LDAGM(
            input_dimension=2 * vgae_embed_dim,
            hidden_dimension=ldagm_hidden_dim,
            feature_num=1,
            hiddenLayer_num=ldagm_layers,
            drop_rate=drop_rate,
            use_aggregate=use_aggregate
        )
        
        # Note: LDAGM already includes prediction layers, no additional head needed
        
    def forward(self, adj_matrix, features, node_pairs=None, training=True):
        """
        Optimized forward pass with improved feature processing.
        """
        # VGAE forward pass
        reconstructed_adj, mu, log_var = self.vgae(adj_matrix, features)
        
        if node_pairs is None:
            return reconstructed_adj, mu, log_var
        
        # Transform embeddings
        if training:
            transformed_mu = self.embedding_transform(mu)
        else:
            transformed_mu = mu
        
        # Extract node embeddings for link prediction
        node1_embeddings = transformed_mu[node_pairs[:, 0]]
        node2_embeddings = transformed_mu[node_pairs[:, 1]]
        
        # Multiple feature combination strategies
        # 1. Concatenation
        concat_features = torch.cat([node1_embeddings, node2_embeddings], dim=1)
        
        # 2. Element-wise product (Hadamard)
        hadamard_features = node1_embeddings * node2_embeddings
        
        # 3. Absolute difference
        diff_features = torch.abs(node1_embeddings - node2_embeddings)
        
        # Combine all features
        combined_features = torch.cat([concat_features, hadamard_features, diff_features], dim=1)
        
        # Reshape for LDAGM: [batch_size, 1, feature_dim]
        ldagm_input = concat_features.unsqueeze(1)  # Use concatenation for LDAGM
        
        # LDAGM forward pass (already includes prediction)
        final_predictions = self.ldagm(ldagm_input)
        
        return reconstructed_adj, mu, log_var, final_predictions

def optimized_joint_loss_function(reconstructed_adj, original_adj, mu, log_var, 
                                 link_predictions, link_labels, num_nodes, pos_weight,
                                 vgae_weight=0.5, link_weight=2.0, kl_weight=0.01, 
                                 regularization_weight=0.001):
    """
    Optimized loss function with better weighting and regularization.
    """
    # VGAE reconstruction loss with focal loss-like weighting
    vgae_reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstructed_adj.view(-1), 
        original_adj.view(-1), 
        pos_weight=pos_weight,
        reduction='mean'
    )
    
    # KL divergence loss with annealing
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / num_nodes
    
    # Link prediction loss with class balancing
    link_prediction_loss = F.binary_cross_entropy_with_logits(
        link_predictions.squeeze(), link_labels,
        reduction='mean'
    )
    
    # L2 regularization on embeddings
    l2_reg = regularization_weight * torch.norm(mu, p=2)
    
    # Combined loss with adaptive weighting
    total_loss = (
        vgae_weight * vgae_reconstruction_loss + 
        kl_weight * kl_loss + 
        link_weight * link_prediction_loss +
        l2_reg
    )
    
    return total_loss, {
        'vgae_reconstruction': vgae_reconstruction_loss.item(),
        'kl_divergence': kl_loss.item(),
        'link_prediction': link_prediction_loss.item(),
        'l2_regularization': l2_reg.item(),
        'total': total_loss.item()
    }

def optimized_joint_train(model, train_dataset, adj_matrix, features, num_nodes, pos_weight,
                         batch_size=64, epochs=200, lr=1e-3, weight_decay=1e-5, device='cpu',
                         vgae_weight=0.5, link_weight=2.0, kl_weight=0.01, patience=20):
    """
    Optimized training with learning rate scheduling and early stopping.
    """
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    features = features.to(device)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # Adaptive KL weight annealing
        current_kl_weight = min(kl_weight * (epoch / 50), kl_weight)
        
        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_adj, mu, log_var, link_predictions = model(
                adj_matrix, features, batch_node_pairs, training=True
            )
            
            # Compute optimized loss
            total_loss, loss_components = optimized_joint_loss_function(
                reconstructed_adj, adj_matrix, mu, log_var,
                link_predictions, batch_labels, num_nodes, pos_weight,
                vgae_weight, link_weight, current_kl_weight
            )
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss_components)
        
        # Average losses for the epoch
        avg_losses = {
            key: np.mean([loss[key] for loss in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        loss_history.append(avg_losses)
        
        # Learning rate scheduling
        scheduler.step(avg_losses['total'])
        
        # Early stopping
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            patience_counter = 0
            # Save best model
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}")
            print(f"  KL Divergence: {avg_losses['kl_divergence']:.4f}")
            print(f"  Link Prediction: {avg_losses['link_prediction']:.4f}")
            print(f"  L2 Regularization: {avg_losses['l2_regularization']:.4f}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model, loss_history

def optimized_joint_test(model, test_dataset, adj_matrix, features, batch_size=64, device='cpu'):
    """
    Optimized test function with ensemble-like prediction.
    """
    model.eval()
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    features = features.to(device)
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            
            # Forward pass
            _, _, _, link_predictions = model(
                adj_matrix, features, batch_node_pairs, training=False
            )
            
            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(link_predictions.squeeze())
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    return np.array(all_labels), np.array(all_predictions)

if __name__ == '__main__':
    # Configuration
    dataset = "dataset2"
    fold = 0
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data (same as original)
    positive5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/positive5foldsidx.npy",
        allow_pickle=True,
    )
    negative5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/negative5foldsidx.npy",
        allow_pickle=True,
    )
    positive_ij = np.load("./our_dataset/" + dataset + "/index/positive_ij.npy")
    negative_ij = np.load("./our_dataset/" + dataset + "/index/negative_ij.npy")
    
    train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
    train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
    test_positive_ij = positive_ij[positive5foldsidx[fold]["test"]]
    test_negative_ij = negative_ij[negative5foldsidx[fold]["test"]]
    
    # Load similarity matrices
    di_semantic_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_func_similarity.npy")
    
    # Load interaction matrices
    lnc_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_di.csv')
    lnc_di.set_index('0', inplace=True)
    lnc_di = lnc_di.values
    lnc_di_copy = copy.copy(lnc_di)
    
    lnc_mi = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_mi.csv', index_col='0').values
    mi_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/mi_di.csv')
    mi_di.set_index('0', inplace=True)
    mi_di = mi_di.values
    
    # Get dimensions
    num_diseases = di_semantic_similarity.shape[0]
    num_lnc = lnc_gip_similarity.shape[0]
    num_mi = mi_gip_similarity.shape[0]
    num_views = 2
    lncRNALen = num_lnc
    
    # Remove test edges from training adjacency matrix
    for ij in positive_ij[positive5foldsidx[fold]['test']]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0
    
    # Create adjacency matrices for each entity type
    disease_adjacency_matrices = [
        torch.tensor(di_semantic_similarity, dtype=torch.float32).to(device), 
        torch.tensor(di_gip_similarity, dtype=torch.float32).to(device)
    ]
    lnaRNA_adjacency_matrices = [
        torch.tensor(lnc_gip_similarity, dtype=torch.float32).to(device), 
        torch.tensor(lnc_func_similarity, dtype=torch.float32).to(device)
    ]
    miRNA_adjacency_matrices = [
        torch.tensor(mi_gip_similarity, dtype=torch.float32).to(device), 
        torch.tensor(mi_func_similarity, dtype=torch.float32).to(device)
    ]
    
    # Compute fused features
    disease_fused_features = cal_fused_features(num_diseases, num_views, disease_adjacency_matrices)
    lnc_fused_features = cal_fused_features(num_lnc, num_views, lnaRNA_adjacency_matrices)
    mi_fused_features = cal_fused_features(num_mi, num_views, miRNA_adjacency_matrices)
    
    # Create heterogeneous adjacency matrix
    A_numpy = concatenate(num_lnc, num_diseases, num_mi, lnc_di_copy, lnc_mi, mi_di, 
                         lnc_fused_features, disease_fused_features, mi_fused_features)
    A = torch.Tensor(A_numpy).to(device)
    
    # Model parameters (optimized)
    num_nodes = A.shape[0]
    in_dimension = A.shape[1]
    vgae_hidden_dim = 64  # Increased
    vgae_embed_dim = 32   # Increased
    ldagm_hidden_dim = 64 # Increased
    ldagm_layers = 3      # Reduced for better generalization
    
    # Prepare inputs
    adj_input = A + torch.eye(num_nodes).to(device)
    features_input = torch.eye(num_nodes).to(device)
    pos_weight = torch.tensor(float(num_nodes**2 - adj_input.sum()) / adj_input.sum()).to(device)
    
    # Create optimized datasets with balanced sampling
    train_dataset = OptimizedJointDataset(
        adj_input, train_positive_ij, train_negative_ij, "train", dataset, balance_ratio=1.0
    )
    test_dataset = OptimizedJointDataset(
        adj_input, test_positive_ij, test_negative_ij, "test", dataset, balance_ratio=1.0
    )
    
    # Initialize optimized joint model
    optimized_joint_model = OptimizedJointVGAE_LDAGM(
        vgae_in_dim=in_dimension,
        vgae_hidden_dim=vgae_hidden_dim,
        vgae_embed_dim=vgae_embed_dim,
        ldagm_hidden_dim=ldagm_hidden_dim,
        ldagm_layers=ldagm_layers,
        drop_rate=0.2,
        use_aggregate=True
    )
    
    print("Starting optimized joint end-to-end training...")
    
    # Optimized joint training
    trained_model, loss_history = optimized_joint_train(
        model=optimized_joint_model,
        train_dataset=train_dataset,
        adj_matrix=adj_input,
        features=features_input,
        num_nodes=num_nodes,
        pos_weight=pos_weight,
        batch_size=64,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-5,
        device=device,
        vgae_weight=0.5,
        link_weight=2.0,
        kl_weight=0.01,
        patience=30
    )
    
    print("\nTesting optimized joint model...")
    
    # Testing
    test_labels, test_predictions = optimized_joint_test(
        model=trained_model,
        test_dataset=test_dataset,
        adj_matrix=adj_input,
        features=features_input,
        batch_size=64,
        device=device
    )
    
    # Evaluate results
    AUC = roc_auc_score(test_labels, test_predictions)
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
    AUPR = auc(recall, precision)
    
    # Binary predictions for other metrics
    binary_preds = (test_predictions > 0.5).astype(int)
    MCC = matthews_corrcoef(test_labels, binary_preds)
    ACC = accuracy_score(test_labels, binary_preds)
    P = precision_score(test_labels, binary_preds)
    R = recall_score(test_labels, binary_preds)
    F1 = f1_score(test_labels, binary_preds)
    
    print("\n=== Optimized Joint End-to-End Training Results ===")
    print(f"AUC: {AUC:.4f}")
    print(f"AUPR: {AUPR:.4f}")
    print(f"MCC: {MCC:.4f}")
    print(f"ACC: {ACC:.4f}")
    print(f"Precision: {P:.4f}")
    print(f"Recall: {R:.4f}")
    print(f"F1-Score: {F1:.4f}")
    
    print("\nOptimized joint end-to-end training completed successfully!")