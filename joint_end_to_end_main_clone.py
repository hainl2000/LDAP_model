import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader, Dataset
from construct_multiview_gcn_gat import concatenate, MultiViewFeatureExtractor
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

class JointDataset(Dataset):
    """
    Dataset for joint end-to-end training that provides both graph structure 
    and node pair features for link prediction.
    """
    def __init__(self, adj_matrix, positive_ij, negative_ij, mode, dataset):
        super().__init__()
        self.adj_matrix = adj_matrix
        
        # Combine positive and negative samples
        all_ij = np.vstack([positive_ij, negative_ij])
        labels = np.hstack([np.ones(len(positive_ij)), np.zeros(len(negative_ij))])
        
        self.node_pairs = torch.tensor(all_ij, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        print(f"{dataset} {mode} dataset loaded with {len(self.node_pairs)} samples")
    
    def __getitem__(self, index):
        return self.node_pairs[index], self.labels[index]
    
    def __len__(self):
        return len(self.node_pairs)

class JointVGAE_LDAGM(nn.Module):
    """
    Joint model combining GCN-Attention, VGAE for graph representation learning 
    and LDAGM for link prediction in an end-to-end manner.
    """
    def __init__(self, num_lnc, num_diseases, num_mi, vgae_in_dim, vgae_hidden_dim, vgae_embed_dim, 
                 ldagm_hidden_dim, ldagm_layers, drop_rate=0.1, use_aggregate=True, 
                 gcn_hidden_dim=128, fusion_output_dim=128):
        super(JointVGAE_LDAGM, self).__init__()
        
        # Store dimensions for multi-view processing
        self.num_lnc = num_lnc
        self.num_diseases = num_diseases
        self.num_mi = num_mi
        
        # GCN-Attention components for multi-view feature extraction
        self.disease_feature_extractor = MultiViewFeatureExtractor(
            num_nodes=num_diseases, num_views=2, 
            gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
        )
        self.lnc_feature_extractor = MultiViewFeatureExtractor(
            num_nodes=num_lnc, num_views=2, 
            gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
        )
        self.mi_feature_extractor = MultiViewFeatureExtractor(
            num_nodes=num_mi, num_views=2, 
            gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
        )
        
        # VGAE component for graph representation learning
        self.vgae = VGAE_Model(vgae_in_dim, vgae_hidden_dim, vgae_embed_dim)
        
        # LDAGM component for link prediction
        # Input dimension: 2 * vgae_embed_dim (concatenated node embeddings)
        self.ldagm = LDAGM(
            input_dimension=2 * vgae_embed_dim,
            hidden_dimension=ldagm_hidden_dim,
            feature_num=1,  # Single feature vector per node pair
            hiddenLayer_num=ldagm_layers,
            drop_rate=drop_rate,
            use_aggregate=use_aggregate
        )
        
    def forward(self, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, node_pairs=None):
        """
        Forward pass for joint end-to-end training.
        
        Args:
            multi_view_data: Dictionary containing multi-view adjacency matrices
                - 'disease_matrices': List of disease similarity matrices
                - 'lnc_matrices': List of lncRNA similarity matrices  
                - 'mi_matrices': List of miRNA similarity matrices
            lnc_di_interaction: lncRNA-disease interaction matrix
            lnc_mi_interaction: lncRNA-miRNA interaction matrix
            mi_di_interaction: miRNA-disease interaction matrix
            node_pairs: Node pairs for link prediction (optional)
            
        Returns:
            If node_pairs is None: (reconstructed_adj, mu, log_var)
            If node_pairs is provided: (reconstructed_adj, mu, log_var, link_predictions)
        """
        # Step 1: Extract multi-view features using GCN-Attention
        disease_fused_features, _, _ = self.disease_feature_extractor(multi_view_data['disease'])
        lnc_fused_features, _, _ = self.lnc_feature_extractor(multi_view_data['lnc'])
        mi_fused_features, _, _ = self.mi_feature_extractor(multi_view_data['mi'])
        
        # Step 2: Reconstruct similarity matrices from fused features
        from construct_multiview_gcn_gat import reconstruct_similarity_matrix
        disease_similarity = reconstruct_similarity_matrix(disease_fused_features)
        lnc_similarity = reconstruct_similarity_matrix(lnc_fused_features)
        mi_similarity = reconstruct_similarity_matrix(mi_fused_features)
        
        # Step 3: Concatenate into final adjacency matrix
        adj_matrix_np = concatenate(
            self.num_lnc, self.num_diseases, self.num_mi,
            lnc_di_interaction.detach().cpu().numpy(),
            lnc_mi_interaction.detach().cpu().numpy(), 
            mi_di_interaction.detach().cpu().numpy(),
            lnc_similarity.detach().cpu().numpy(),
            disease_similarity.detach().cpu().numpy(),
            mi_similarity.detach().cpu().numpy()
        )
        adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32, device=disease_fused_features.device)
        
        # Step 4: Add self-loops and create features
        num_nodes = adj_matrix.shape[0]
        adj_input = adj_matrix + torch.eye(num_nodes, device=adj_matrix.device)
        features_input = torch.eye(num_nodes, device=adj_matrix.device)
        
        # Step 5: VGAE forward pass
        reconstructed_adj, mu, log_var = self.vgae(adj_input, features_input)
        
        if node_pairs is None:
            return reconstructed_adj, mu, log_var
        
        # Step 6: Extract node embeddings for link prediction
        node1_embeddings = mu[node_pairs[:, 0]]  # Shape: [batch_size, embed_dim]
        node2_embeddings = mu[node_pairs[:, 1]]  # Shape: [batch_size, embed_dim]
        
        # Concatenate node embeddings
        pair_features = torch.cat([node1_embeddings, node2_embeddings], dim=1)
        # Reshape for LDAGM: [batch_size, 1, 2*embed_dim]
        pair_features = pair_features.unsqueeze(1)
        
        # Step 7: LDAGM forward pass for link prediction
        link_predictions = self.ldagm(pair_features)
        
        return reconstructed_adj, mu, log_var, link_predictions

def joint_loss_function(reconstructed_adj, original_adj, mu, log_var, 
                       link_predictions, link_labels, num_nodes, pos_weight,
                       vgae_weight=1.0, link_weight=1.0, kl_weight=0.1):
    """
    Unified loss function for joint end-to-end training.
    
    Args:
        reconstructed_adj: Reconstructed adjacency matrix from VGAE
        original_adj: Original adjacency matrix
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        link_predictions: Link prediction scores from LDAGM
        link_labels: Ground truth link labels
        num_nodes: Number of nodes in the graph
        pos_weight: Weight for positive samples in reconstruction loss
        vgae_weight: Weight for VGAE reconstruction loss
        link_weight: Weight for link prediction loss
        kl_weight: Weight for KL divergence term
        
    Returns:
        Combined loss value and individual loss components
    """
    # VGAE reconstruction loss
    vgae_reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstructed_adj.view(-1), 
        original_adj.view(-1), 
        pos_weight=pos_weight
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / num_nodes
    
    # Link prediction loss
    link_prediction_loss = F.binary_cross_entropy_with_logits(
        link_predictions, link_labels
    )
    
    # Combined loss
    total_loss = (
        vgae_weight * vgae_reconstruction_loss + 
        kl_weight * kl_loss + 
        link_weight * link_prediction_loss
    )
    
    return total_loss, {
        'vgae_reconstruction': vgae_reconstruction_loss.item(),
        'kl_divergence': kl_loss.item(),
        'link_prediction': link_prediction_loss.item(),
        'total': total_loss.item()
    }

def joint_train(model, train_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, 
               mi_di_interaction, num_nodes, pos_weight, batch_size=32, epochs=100, lr=1e-3, 
               weight_decay=1e-4, device='cpu', vgae_weight=1.0, link_weight=1.0, kl_weight=0.1):
    """
    Joint end-to-end training function with GCN-Attention integration.
    
    Args:
        model: JointVGAE_LDAGM model
        train_dataset: Training dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        num_nodes: Number of nodes
        pos_weight: Positive weight for reconstruction loss
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Training device
        vgae_weight: Weight for VGAE loss
        link_weight: Weight for link prediction loss
        kl_weight: Weight for KL divergence
        
    Returns:
        Trained model and loss history
    """
    model = model.to(device)
    lnc_di_interaction = lnc_di_interaction.to(device)
    lnc_mi_interaction = lnc_mi_interaction.to(device)
    mi_di_interaction = mi_di_interaction.to(device)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with end-to-end GCN-Attention + VGAE + LDAGM
            reconstructed_adj, mu, log_var, link_predictions = model(
                multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs
            )
            
            # Get the original adjacency matrix from the model's forward pass
            # We need to call the model again without node_pairs to get the original adj matrix
            with torch.no_grad():
                original_adj_reconstructed, _, _ = model(
                    multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, None
                )
            
            # Compute joint loss using the dynamically generated adjacency matrix
            total_loss, loss_components = joint_loss_function(
                reconstructed_adj, original_adj_reconstructed.detach(), mu, log_var,
                link_predictions, batch_labels, num_nodes, pos_weight,
                vgae_weight, link_weight, kl_weight
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_components)
        
        # Average losses for the epoch
        avg_losses = {
            key: np.mean([loss[key] for loss in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        loss_history.append(avg_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}")
            print(f"  KL Divergence: {avg_losses['kl_divergence']:.4f}")
            print(f"  Link Prediction: {avg_losses['link_prediction']:.4f}")
    
    return model, loss_history

def joint_test(model, test_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_size=32, device='cpu'):
    """
    Test function for joint model with GCN-Attention integration.
    
    Args:
        model: Trained JointVGAE_LDAGM model
        test_dataset: Test dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        batch_size: Batch size for testing
        device: Testing device
        
    Returns:
        True labels and predictions
    """
    model.eval()
    model = model.to(device)
    lnc_di_interaction = lnc_di_interaction.to(device)
    lnc_mi_interaction = lnc_mi_interaction.to(device)
    mi_di_interaction = mi_di_interaction.to(device)
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_node_pairs, batch_labels in dataloader:
            batch_node_pairs = batch_node_pairs.to(device)
            
            # Forward pass with end-to-end GCN-Attention integration
            _, _, _, link_predictions = model(multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs)
            
            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(link_predictions)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    return np.array(all_labels), np.array(all_predictions)

if __name__ == '__main__':
    # Configuration
    dataset = "dataset2"
    fold = 0
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
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
    
    # Note: GCN-Attention feature extraction is now handled within the model
    # We still need the concatenated matrix for VGAE input dimensions
    # Using identity matrices as placeholder similarity matrices since actual features come from GCN-Attention
    lnc_placeholder = np.eye(num_lnc)
    disease_placeholder = np.eye(num_diseases) 
    mi_placeholder = np.eye(num_mi)
    A_numpy = concatenate(num_lnc, num_diseases, num_mi, lnc_di_copy, lnc_mi, mi_di, 
                         lnc_placeholder, disease_placeholder, mi_placeholder)
    A = torch.Tensor(A_numpy).to(device)

    # Model parameters
    num_nodes = A.shape[0]
    in_dimension = A.shape[1]
    vgae_hidden_dim = 32
    vgae_embed_dim = 16
    ldagm_hidden_dim = 40
    ldagm_layers = 5
    
    # Prepare inputs
    adj_input = A + torch.eye(num_nodes).to(device)
    features_input = torch.eye(num_nodes).to(device)
    pos_weight = torch.tensor(float(num_nodes**2 - adj_input.sum()) / adj_input.sum()).to(device)
    
    # Create datasets
    train_dataset = JointDataset(adj_input, train_positive_ij, train_negative_ij, "train", dataset)
    test_dataset = JointDataset(adj_input, test_positive_ij, test_negative_ij, "test", dataset)
    
    # Prepare multi-view data structure for GCN-Attention
    multi_view_data = {
        'disease': disease_adjacency_matrices,
        'lnc': lnaRNA_adjacency_matrices,
        'mi': miRNA_adjacency_matrices
    }
    
    # Prepare interaction matrices as tensors
    lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32).to(device)
    lnc_mi_tensor = torch.tensor(lnc_mi, dtype=torch.float32).to(device)
    mi_di_tensor = torch.tensor(mi_di, dtype=torch.float32).to(device)
    
    # Initialize joint model with GCN-Attention integration
    joint_model = JointVGAE_LDAGM(
        num_lnc=num_lnc,
        num_diseases=num_diseases,
        num_mi=num_mi,
        vgae_in_dim=in_dimension,
        vgae_hidden_dim=vgae_hidden_dim,
        vgae_embed_dim=vgae_embed_dim,
        ldagm_hidden_dim=ldagm_hidden_dim,
        ldagm_layers=ldagm_layers,
        drop_rate=0.1,
        use_aggregate=True,
        gcn_hidden_dim=128,
        fusion_output_dim=128
    )
    
    print("Starting joint end-to-end training...")
    
    # Joint training with end-to-end GCN-Attention integration
    trained_model, loss_history = joint_train(
        model=joint_model,
        train_dataset=train_dataset,
        multi_view_data=multi_view_data,
        lnc_di_interaction=lnc_di_tensor,
        lnc_mi_interaction=lnc_mi_tensor,
        mi_di_interaction=mi_di_tensor,
        num_nodes=num_nodes,
        pos_weight=pos_weight,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        vgae_weight=1.0,
        link_weight=2.0,  # Higher weight for link prediction task
        kl_weight=0.1
    )
    
    print("\nTesting joint model...")
    
    # Testing with end-to-end GCN-Attention integration
    test_labels, test_predictions = joint_test(
        model=trained_model,
        test_dataset=test_dataset,
        multi_view_data=multi_view_data,
        lnc_di_interaction=lnc_di_tensor,
        lnc_mi_interaction=lnc_mi_tensor,
        mi_di_interaction=mi_di_tensor,
        batch_size=32,
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
    
    print("\n=== Joint End-to-End Training Results ===")
    print(f"AUC: {AUC:.4f}")
    print(f"AUPR: {AUPR:.4f}")
    print(f"MCC: {MCC:.4f}")
    print(f"ACC: {ACC:.4f}")
    print(f"Precision: {P:.4f}")
    print(f"Recall: {R:.4f}")
    print(f"F1-Score: {F1:.4f}")
    
    print("\nJoint end-to-end training completed successfully!")