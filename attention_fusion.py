import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from main import MyDataset

class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for integrating multiple disease similarity views.
    
    Implements the attention mechanism described in equations (8) and (9):
    - Eq. (8): ωd = σ(fcd(AvgPool(Hd)))
    - Eq. (9): Hf_d = CNNd(σ(ωd⋅Hd))
    
    This module calculates attention weights for diseases across different similarity
    perspectives and integrates diverse similarity features.
    """
    
    def __init__(self, input_dim, num_views=2, attention_dim=64, output_dim=128, 
                 conv_out_channels=64, kernel_size=3, activation='relu'):
        """
        Initialize the AttentionFusion module.
        
        Args:
            input_dim (int): Input feature dimension from GCN embeddings
            num_views (int): Number of similarity views (default: 2)
            attention_dim (int): Dimension of attention mechanism (default: 64)
            output_dim (int): Output dimension after fusion (default: 128)
            conv_out_channels (int): Number of output channels for 2D convolution (default: 64)
            kernel_size (int): Kernel size for 2D convolution (default: 3)
            activation (str): Activation function type ('relu', 'tanh', 'sigmoid')
        """
        super(AttentionFusion, self).__init__()
        
        self.input_dim = input_dim
        self.num_views = num_views
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.conv_out_channels = conv_out_channels
        
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer for attention weight calculation (fcd)
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            self._get_activation(activation),
            nn.Linear(attention_dim, num_views),
            nn.Softmax(dim=-1)  # Normalize attention weights across views
        )
        
        # 2D Convolution layer for feature integration (CNNd)
        # Input: (batch_size, num_views, num_nodes, input_dim)
        # We treat num_views and num_nodes as height and width dimensions
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=True
        )
        
        # Additional layers for final feature mapping
        self.feature_mapping = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, input_dim)),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(conv_out_channels * input_dim, output_dim),
            self._get_activation(activation)
        )
        
        # Activation function for element-wise operations
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation):
        """
        Get activation function based on string name.
        
        Args:
            activation (str): Activation function name
            
        Returns:
            nn.Module: Activation function
        """
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # Default to ReLU
    
    def _init_weights(self):
        """
        Initialize the weights of the layers using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def calculate_attention_weights(self, disease_embeddings):
        """
        Calculate attention weights for diseases across different similarity perspectives.
        Implements Eq. (8): ωd = σ(fcd(AvgPool(Hd)))
        
        Args:
            disease_embeddings (torch.Tensor): Disease node embeddings from different views
                                              Shape: (batch_size, num_views, num_nodes, input_dim)
                                              
        Returns:
            torch.Tensor: Attention weights for each view
                         Shape: (batch_size, num_nodes, num_views)
        """
        batch_size, num_views, num_nodes, input_dim = disease_embeddings.shape
        
        # Reshape for global average pooling: (batch_size * num_views * num_nodes, input_dim)
        embeddings_flat = disease_embeddings.view(-1, input_dim)
        
        # Apply global average pooling
        # Reshape to (batch_size * num_views * num_nodes, input_dim, 1) for 1D pooling
        pooled_features = self.global_avg_pool(embeddings_flat.unsqueeze(-1)).squeeze(-1)
        
        # Apply fully connected layer to get attention weights
        # Shape: (batch_size * num_views * num_nodes, num_views)
        attention_weights = self.attention_fc(pooled_features)
        
        # Reshape back to (batch_size, num_views, num_nodes, num_views)
        attention_weights = attention_weights.view(batch_size, num_views, num_nodes, num_views)
        
        # Average across the first view dimension to get final attention weights
        # Shape: (batch_size, num_nodes, num_views)
        attention_weights = torch.mean(attention_weights, dim=1)
        
        return attention_weights
    
    def apply_attention_and_fuse(self, disease_embeddings, attention_weights):
        """
        Apply attention weights and fuse features using 2D convolution.
        Implements Eq. (9): Hf_d = CNNd(σ(ωd⋅Hd))
        
        Args:
            disease_embeddings (torch.Tensor): Disease node embeddings from different views
                                              Shape: (batch_size, num_views, num_nodes, input_dim)
            attention_weights (torch.Tensor): Attention weights for each view
                                             Shape: (batch_size, num_nodes, num_views)
                                             
        Returns:
            torch.Tensor: Fused disease features
                         Shape: (batch_size, num_nodes, output_dim)
        """
        batch_size, num_views, num_nodes, input_dim = disease_embeddings.shape
        
        # Expand attention weights to match embedding dimensions
        # Shape: (batch_size, num_views, num_nodes, 1)
        attention_expanded = attention_weights.transpose(1, 2).unsqueeze(-1)
        
        # Apply attention weights element-wise: ωd⋅Hd
        # Shape: (batch_size, num_views, num_nodes, input_dim)
        weighted_embeddings = attention_expanded * disease_embeddings
        
        # Apply activation function: σ(ωd⋅Hd)
        activated_embeddings = self.activation(weighted_embeddings)
        
        # Reshape for 2D convolution: (batch_size, 1, num_views * num_nodes, input_dim)
        conv_input = activated_embeddings.view(batch_size, 1, num_views * num_nodes, input_dim)
        
        # Apply 2D convolution: CNNd(σ(ωd⋅Hd))
        conv_output = self.conv2d(conv_input)
        
        # Apply feature mapping to get final output
        fused_features = self.feature_mapping(conv_output)
        
        # Reshape to (batch_size, num_nodes, output_dim)
        fused_features = fused_features.view(batch_size, num_nodes, -1)
        
        return fused_features
    
    def forward(self, disease_embeddings_list):
        """
        Forward pass of the AttentionFusion module.
        
        Args:
            disease_embeddings_list (list): List of disease embeddings from different similarity views
                                           Each element shape: (batch_size, num_nodes, input_dim)
                                           
        Returns:
            torch.Tensor: Fused disease features
                         Shape: (batch_size, num_nodes, output_dim)
        """
        # Stack embeddings from different views
        # Shape: (batch_size, num_views, num_nodes, input_dim)
        disease_embeddings = torch.stack(disease_embeddings_list, dim=1)
        
        # Calculate attention weights (Eq. 8)
        attention_weights = self.calculate_attention_weights(disease_embeddings)
        
        # Apply attention and fuse features (Eq. 9)
        fused_features = self.apply_attention_and_fuse(disease_embeddings, attention_weights)
        
        return fused_features, attention_weights
    
    def get_attention_weights_only(self, disease_embeddings_list):
        """
        Get only the attention weights without fusion (for analysis purposes).
        
        Args:
            disease_embeddings_list (list): List of disease embeddings from different similarity views
                                           
        Returns:
            torch.Tensor: Attention weights for each view
        """
        disease_embeddings = torch.stack(disease_embeddings_list, dim=1)
        attention_weights = self.calculate_attention_weights(disease_embeddings)
        return attention_weights


class MultiViewGCNWithAttention(nn.Module):
    """
    Complete module that combines GCN feature extraction with attention-based fusion
    for multi-view disease similarity learning.
    """
    
    def __init__(self, input_dim, gcn_hidden_dim=128, gcn_num_layers=2, gcn_dropout=0.1,
                 num_views=2, attention_dim=64, fusion_output_dim=128, 
                 conv_out_channels=64, kernel_size=3, activation='relu'):
        """
        Initialize the complete multi-view GCN with attention fusion.
        
        Args:
            input_dim (int): Input feature dimension
            gcn_hidden_dim (int): Hidden dimension for GCN layers
            gcn_num_layers (int): Number of GCN layers
            gcn_dropout (float): Dropout rate for GCN
            num_views (int): Number of similarity views
            attention_dim (int): Dimension of attention mechanism
            fusion_output_dim (int): Output dimension after fusion
            conv_out_channels (int): Number of output channels for 2D convolution
            kernel_size (int): Kernel size for 2D convolution
            activation (str): Activation function type
        """
        super(MultiViewGCNWithAttention, self).__init__()
        
        # Import GCN from the previously created module
        from gcn_feature_learning import GCN
        
        # Create separate GCN models for each view
        self.gcn_models = nn.ModuleList([
            GCN(input_dim, gcn_hidden_dim, gcn_num_layers, gcn_dropout)
            for _ in range(num_views)
        ])
        
        # Attention fusion module
        self.attention_fusion = AttentionFusion(
            input_dim=gcn_hidden_dim,
            num_views=num_views,
            attention_dim=attention_dim,
            output_dim=fusion_output_dim,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size,
            activation=activation
        )
        
        self.num_views = num_views
    
    def forward(self, node_features_list, adjacency_matrices_list):
        """
        Forward pass through multi-view GCN with attention fusion.
        
        Args:
            node_features_list (list): List of node features for each view
            adjacency_matrices_list (list): List of adjacency matrices for each view
            
        Returns:
            tuple: (fused_features, attention_weights, individual_embeddings)
        """
        # Extract features from each view using separate GCN models
        view_embeddings = []
        for i in range(self.num_views):
            embeddings = self.gcn_models[i](node_features_list[i], adjacency_matrices_list[i])
            view_embeddings.append(embeddings)
        
        # Fuse features using attention mechanism
        fused_features, attention_weights = self.attention_fusion(view_embeddings)
        
        return fused_features, attention_weights, view_embeddings


def run_attention_fusion(input_dim, num_views):
    """
    Tests the MultiViewGCNWithAttention model with random data.

    Initializes the model, generates random node features and adjacency matrices,
    runs a forward pass, and prints the shapes of the resulting tensors.

    Args:
        di_num_nodes (int): The number of disease nodes.
        input_dim (int): The input feature dimension.
        num_views (int): The number of similarity views.
    """    
    # Initialize the complete model
    model = MultiViewGCNWithAttention(
        input_dim=input_dim,
        gcn_hidden_dim=128,
        num_views=num_views,
        fusion_output_dim=128
    )
    
    # Forward pass
    node_features_list = [node_features_view1, node_features_view2]
    adjacency_matrices_list = [adjacency_matrix_view1, adjacency_matrix_view2]
    
    fused_features, attention_weights, individual_embeddings = model(
        node_features_list, adjacency_matrices_list
    )
    
    print("--- Attention Fusion Test ---")
    print(f"Input shapes: {[x.shape for x in node_features_list]}")
    print(f"Individual embedding shapes: {[x.shape for x in individual_embeddings]}")
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights (first 5 nodes, both views):")
    print(attention_weights[:5, :])
    print("--------------------------")


# Example usage
if __name__ == "__main__":
    dataset = "dataset2"
    fold = 0

    # Pytorch for NVIDIA
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Pytorch for Apple Silicon
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(device)

    # Read the indexes of positive and negative samples for the training and test sets
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

    print(f"train_positive_ij shape: {train_positive_ij.shape}")
    print(f"train_negative_ij shape: {train_negative_ij.shape}")
    print(f"test_positive_ij shape: {test_positive_ij.shape}")
    print(f"test_negative_ij shape: {test_negative_ij.shape}")


    di_semantic_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_func_similarity.npy")

    # GCN, GAT Parameters
    batch_size = 32
    di_num_nodes = di_semantic_similarity.shape[0]
    lnc_num_nodes = lnc_gip_similarity.shape[0]
    mi_num_nodes = mi_gip_similarity.shape[0]
    input_dim = 64
    num_views = 2
        
    # dataloader = DataLoader(
    #     dataset, batch_size, shuffle=True, drop_last=False, pin_memory=True
    # )
    # train_dataset = MyDataset(
    #     network_num, fold, train_positive_ij, train_negative_ij, "训练", dataset, file_name
    # )
    # test_dataset = MyDataset(
    #     network_num, fold, test_positive_ij, test_negative_ij, "测试", dataset, file_name
    # )
    # feature_num = train_dataset.data.shape[1]
    # input_dimension = train_dataset.data.shape[2]
    # print(f"input_dimension: {input_dimension}; feature_num: {feature_num}")