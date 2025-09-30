import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
from construct_multiview_gcn_gat import concatenate, MultiViewFeatureExtractor, reconstruct_similarity_matrix
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
import os
import csv
import config
warnings.filterwarnings("ignore")


class JointDataset(Dataset):
    """
    Dataset for joint end-to-end training that provides both graph structure 
    and node pair features for link prediction.
    """
    def __init__(self, positive_ij, negative_ij, mode, dataset):
        super().__init__()
        
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
                 ldagm_hidden_dim, ldagm_layers, drop_rate=config.DROP_RATE, use_aggregate=config.USE_AGGREGATE, 
                 gcn_hidden_dim=config.GCN_HIDDEN_DIM, fusion_output_dim=config.FUSION_OUTPUT_DIM):
        super(JointVGAE_LDAGM, self).__init__()
        
        # Store dimensions for multi-view processing
        self.num_lnc = num_lnc
        self.num_diseases = num_diseases
        self.num_mi = num_mi
        
        # GCN-Attention components for multi-view feature extraction
        # Initialize feature extractors for each entity type in parallel
        def create_disease_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_diseases, num_views=2, 
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )
        
        def create_lnc_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_lnc, num_views=2, 
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )
        
        def create_mi_extractor():
            return MultiViewFeatureExtractor(
                num_nodes=num_mi, num_views=2, 
                gcn_hidden_dim=gcn_hidden_dim, fusion_output_dim=fusion_output_dim
            )
        
        # Parallel initialization of feature extractors
        with ThreadPoolExecutor(max_workers=3) as executor:
            self.disease_feature_extractor = executor.submit(create_disease_extractor).result()
            self.lnc_feature_extractor = executor.submit(create_lnc_extractor).result()
            self.mi_feature_extractor = executor.submit(create_mi_extractor).result()
        
        # VGAE component for graph representation learning
        self.vgae = VGAE_Model(vgae_in_dim, vgae_hidden_dim, vgae_embed_dim)
        
        # LDAGM component for link prediction
        # Input dimension: flattened features from pair_features tensor
        # pair_features shape: [batch_size, 2, features_per_node]
        # After flattening: [batch_size, 2 * features_per_node]
        # features_per_node = network_num * (vgae_embed_dim + a_encoder_dim)
        a_encoder_dim = config.A_ENCODER_DIM  # Based on A_encoder file shape
        network_num = config.NETWORK_NUM 
        features_per_node = network_num * (vgae_embed_dim + a_encoder_dim)  
        ldagm_input_dim = 2 * features_per_node
        self.ldagm = LDAGM(
            input_dimension=ldagm_input_dim,
            hidden_dimension=ldagm_hidden_dim,
            feature_num=1,  # Single feature vector per node pair
            hiddenLayer_num=ldagm_layers,
            drop_rate=drop_rate,
            use_aggregate=use_aggregate
        )
        
    def forward(self, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, node_pairs=None, network_num=4, fold=0, dataset="dataset1"):
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
            network_num: Number of networks for A_encoder loading
            fold: Fold number for A_encoder file paths
            dataset: Dataset name for A_encoder file paths
            
        Returns:
            If node_pairs is None: (reconstructed_adj, mu, log_var)
            If node_pairs is provided: (reconstructed_adj, mu, log_var, link_predictions)
        """
        # Step 1: Extract multi-view features using GCN-Attention
    
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Gửi 3 nhiệm vụ để thực hiện đồng thời và lấy kết quá
            disease_fused_features, _, _ = executor.submit(self.disease_feature_extractor, multi_view_data['disease']).result()
            lnc_fused_features, _, _ = executor.submit(self.lnc_feature_extractor, multi_view_data['lnc']).result()
            mi_fused_features, _, _ = executor.submit(self.mi_feature_extractor, multi_view_data['mi']).result()
        
        # Step 2: Reconstruct similarity matrices from fused features in parallel
        
        def reconstruct_disease_similarity():
            return reconstruct_similarity_matrix(disease_fused_features)
        
        def reconstruct_lnc_similarity():
            return reconstruct_similarity_matrix(lnc_fused_features)
        
        def reconstruct_mi_similarity():
            return reconstruct_similarity_matrix(mi_fused_features)
        
        # Parallel reconstruction of similarity matrices
        with ThreadPoolExecutor(max_workers=3) as executor:
            disease_similarity = executor.submit(reconstruct_disease_similarity).result()
            lnc_similarity = executor.submit(reconstruct_lnc_similarity).result()
            mi_similarity = executor.submit(reconstruct_mi_similarity).result()
        
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
        adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32, device=config.DEVICE)
        
        # Step 4: Add self-loops and create features
        num_nodes = adj_matrix.shape[0]
        adj_input = adj_matrix
        features_input = torch.eye(num_nodes, device=config.DEVICE)
        
        # Step 5: VGAE forward pass
        reconstructed_adj, mu, log_var = self.vgae(adj_input, features_input)
        
        if node_pairs is None:
            return reconstructed_adj, mu, log_var
        
        # Step 6: Extract features for link prediction following MyDataset pattern
        # Load A_encoder files for each network in the loop
        A_encoders = []
        for i in range(network_num):
            A_encoder = np.load(
                "./our_dataset/"
                + dataset
                + "/Temp_A_encoder/A_"
                + str(fold + 1)
                + "_"
                + str(i + 1)
                + ".npy"
            )
            A_encoders.append(torch.tensor(A_encoder, dtype=torch.float32, device=config.DEVICE))
        
        # Get computational embeddings from VGAE
        node1_computational_embeddings = mu[node_pairs[:, 0]]
        node2_computational_embeddings = mu[node_pairs[:, 1]]
        
        # Collect features from all networks for each node
        node1_all_features = []
        node2_all_features = []
        
        for i in range(network_num):
            # Concatenate computational embeddings with A_encoder features for each network
            node1_features_net_i = torch.cat([node1_computational_embeddings, A_encoders[i][node_pairs[:, 0]]], dim=1)
            node2_features_net_i = torch.cat([node2_computational_embeddings, A_encoders[i][node_pairs[:, 1]]], dim=1)
            
            node1_all_features.append(node1_features_net_i)
            node2_all_features.append(node2_features_net_i)
        
        # Concatenate features from all networks for each node
        node1_final_features = torch.cat(node1_all_features, dim=1)
        node2_final_features = torch.cat(node2_all_features, dim=1)
        
        # Stack node1 and node2 final features
        pair_features = torch.stack([node1_final_features, node2_final_features], dim=1)
        
        # Step 7: LDAGM forward pass for link prediction
        # Reshape from [batch_size, 2, features] to [batch_size, 2*features] for LDAGM
        pair_features_flattened = pair_features.view(pair_features.size(0), -1)
        link_predictions = self.ldagm(pair_features_flattened)
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

def log_hyperparameters():
    if not os.path.exists(os.path.dirname(config.LOG_FILE)):
        os.makedirs(os.path.dirname(config.LOG_FILE))
    with open(config.LOG_FILE, 'a') as f:
        f.write("Hyperparameters:\n")
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value} -- ")
        f.write("\n")

def joint_train(num_lnc, num_diseases, num_mi, train_dataset, multi_view_data, 
               lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
               fold=0, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, lr=config.LEARNING_RATE, 
               weight_decay=config.WEIGHT_DECAY, device=config.DEVICE, 
               vgae_weight=config.VGAE_WEIGHT, link_weight=config.LINK_WEIGHT, kl_weight=config.KL_WEIGHT,
               vgae_hidden_dim=config.VGAE_HIDDEN_DIM, vgae_embed_dim=config.VGAE_EMBED_DIM, 
               ldagm_hidden_dim=config.LDAGM_HIDDEN_DIM, ldagm_layers=config.LDAGM_LAYERS):
    """
    Joint end-to-end training function with GCN-Attention integration.
    Creates model internally and handles all training setup.
    
    Args:
        num_lnc: Number of lncRNA nodes
        num_diseases: Number of disease nodes
        num_mi: Number of miRNA nodes
        train_dataset: Training dataset
        multi_view_data: Dictionary containing multi-view adjacency matrices
        lnc_di_interaction: lncRNA-disease interaction matrix
        lnc_mi_interaction: lncRNA-miRNA interaction matrix
        mi_di_interaction: miRNA-disease interaction matrix
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Training device
        vgae_weight: Weight for VGAE loss
        link_weight: Weight for link prediction loss
        kl_weight: Weight for KL divergence
        vgae_hidden_dim: VGAE hidden dimension
        vgae_embed_dim: VGAE embedding dimension
        ldagm_hidden_dim: LDAGM hidden dimension
        ldagm_layers: Number of LDAGM layers
        
    Returns:
        Trained model and loss history
    """
    
    # Convert interactions to numpy for concatenation
    lnc_di_np = lnc_di_interaction.detach().cpu().numpy()
    lnc_mi_np = lnc_mi_interaction.detach().cpu().numpy()
    mi_di_np = mi_di_interaction.detach().cpu().numpy()

    # Model parameters
    num_nodes = num_lnc + num_diseases + num_mi
    in_dimension = num_lnc + num_diseases + num_mi
    total_links = (lnc_di_np.sum() + lnc_mi_np.sum() + mi_di_np.sum())*2 + num_lnc + num_diseases + num_mi
    pos_weight = torch.tensor(float(num_nodes**2 - total_links) / total_links, device=device)

    # Initialize joint model
    model = JointVGAE_LDAGM(
        num_lnc=num_lnc,
        num_diseases=num_diseases,
        num_mi=num_mi,
        vgae_in_dim=in_dimension,
        vgae_hidden_dim=vgae_hidden_dim,
        vgae_embed_dim=vgae_embed_dim,
        ldagm_hidden_dim=ldagm_hidden_dim,
        ldagm_layers=ldagm_layers,
        drop_rate=config.DROP_RATE,
        use_aggregate=config.USE_AGGREGATE,
        gcn_hidden_dim=config.GCN_HIDDEN_DIM,
        fusion_output_dim=config.FUSION_OUTPUT_DIM
    )
    
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
                multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs, 
                network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET
            )
            
            # Get the original adjacency matrix from the model's forward pass
            # We need to call the model again without node_pairs to get the original adj matrix
            with torch.no_grad():
                original_adj_reconstructed, _, _ = model(
                    multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, None,
                    network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET
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
        
        with open(config.LOG_FILE, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}: Total Loss: {avg_losses['total']:.4f}, VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}, KL Divergence: {avg_losses['kl_divergence']:.4f}, Link Prediction: {avg_losses['link_prediction']:.4f}\n")
            f.write("-----------------------")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  VGAE Reconstruction: {avg_losses['vgae_reconstruction']:.4f}")
            print(f"  KL Divergence: {avg_losses['kl_divergence']:.4f}")
            print(f"  Link Prediction: {avg_losses['link_prediction']:.4f}")
    
    return model, loss_history

def joint_test(model, test_dataset, multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_size=32, fold=0, device='cpu'):
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
            _, _, _, link_predictions = model(multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction, batch_node_pairs,
                                            network_num=config.NETWORK_NUM, fold=fold, dataset=config.DATASET)
            
            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(link_predictions)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    return np.array(all_labels), np.array(all_predictions)

def log_to_csv(config, metrics):
    if not os.path.exists(os.path.dirname(config.CSV_LOG_FILE)):
        os.makedirs(os.path.dirname(config.CSV_LOG_FILE))

    hyperparams = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    fieldnames = list(hyperparams.keys()) + list(metrics.keys())

    file_exists = os.path.isfile(config.CSV_LOG_FILE)

    with open(config.CSV_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**metrics, **hyperparams})

if __name__ == '__main__':
    log_hyperparameters()
    start_time = time.time()
    
    # Configuration
    dataset = config.DATASET
    # Device selection logic based on config and availability
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if config.DEVICE in ["cuda", "mps"]:
            print(f"Warning: {config.DEVICE} not available, falling back to CPU")
    print(f"Using device: {device}")
    
    # Initialize results storage for all folds
    all_fold_results = []
    
    print(f"Starting 5-fold cross-validation for {dataset}...")
    
    for fold in range(config.TOTAL_FOLDS):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold + 1}/5")
        print(f"{'='*50}")
    
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
        
        # Create datasets
        train_dataset = JointDataset(train_positive_ij, train_negative_ij, "train", dataset)
        test_dataset = JointDataset(test_positive_ij, test_negative_ij, "test", dataset)
    
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
        
        print(f"Starting joint end-to-end training for fold {fold + 1}...")
        
        # Joint training with end-to-end GCN-Attention integration
        trained_model, loss_history = joint_train(
            num_lnc=num_lnc,
            num_diseases=num_diseases,
            num_mi=num_mi,
            train_dataset=train_dataset,
            multi_view_data=multi_view_data,
            lnc_di_interaction=lnc_di_tensor,
            lnc_mi_interaction=lnc_mi_tensor,
            mi_di_interaction=mi_di_tensor,
            fold=fold,
            device=device
        )
    
        print(f"\nTesting joint model for fold {fold + 1}...")
        
        # Testing with end-to-end GCN-Attention integration
        test_labels, test_predictions = joint_test(
            model=trained_model,
            test_dataset=test_dataset,
            multi_view_data=multi_view_data,
            lnc_di_interaction=lnc_di_tensor,
            lnc_mi_interaction=lnc_mi_tensor,
            mi_di_interaction=mi_di_tensor,
            fold=fold,
            batch_size=config.BATCH_SIZE,
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

        print(f"\n=== Fold {fold + 1} Results ===")
        print(f"AUC: {AUC:.4f}")
        print(f"AUPR: {AUPR:.4f}")
        print(f"MCC: {MCC:.4f}")
        print(f"ACC: {ACC:.4f}")
        print(f"Precision: {P:.4f}")
        print(f"Recall: {R:.4f}")
        print(f"F1-Score: {F1:.4f}")
        
        # Store results for this fold
        fold_results = {
            "fold": fold + 1,
            "AUC": AUC,
            "AUPR": AUPR,
            "MCC": MCC,
            "ACC": ACC,
            "Precision": P,
            "Recall": R,
            "F1-Score": F1
        }
        all_fold_results.append(fold_results)
    
    # Calculate statistics across all folds
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(all_fold_results)
    
    # Calculate mean and std for each metric
    metrics = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score']
    
    print("\nDetailed Results by Fold:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    print("\nStatistical Summary:")
    print("-" * 50)
    for metric in metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # End timing
    end_time = time.time()
    total_seconds = end_time - start_time
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    print(f"\nTotal runtime: {int(h)} hours, {int(m)} minutes, and {s:.2f} seconds")
    
    # Log to file
    with open(config.LOG_FILE, 'a') as f:
        f.write("\n=== 5-Fold Cross-Validation Results ===\n")
        for metric in metrics:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write(f"Total runtime: {int(h)} hours, {int(m)} minutes, and {s:.2f} seconds\n")
        f.write("================================================\n")
    
    # Log average metrics to CSV
    avg_metrics = {
        "AUC": results_df['AUC'].mean(),
        "AUPR": results_df['AUPR'].mean(),
        "MCC": results_df['MCC'].mean(),
        "ACC": results_df['ACC'].mean(),
        "Precision": results_df['Precision'].mean(),
        "Recall": results_df['Recall'].mean(),
        "F1-Score": results_df['F1-Score'].mean(),
        "Time": f"{int(h)} hours, {int(m)} minutes, and {s:.2f} seconds",
    }
    log_to_csv(config, avg_metrics)
    
    print("\n5-fold cross-validation completed successfully!")