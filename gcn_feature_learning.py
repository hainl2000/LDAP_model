import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for feature learning.
    
    Implements the GCN formula:
    H(l+1) = σ(D^(-1/2) * A * D^(-1/2) * W(l) * H(l))
    
    where:
    - A is the adjacency matrix
    - H(l+1) represents the embedding of the node at the (l+1)-th layer
    - W(l) is a learnable weight matrix
    - D is the degree matrix of the adjacency matrix A
    - σ is an activation function (ReLU)
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.4):
        """
        Initialize the GCN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension (default: 128)
            num_layers (int): Number of GCN layers (default: 2)
            dropout_rate (float): Dropout rate (default: 0.4)
        """
        super(GCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Create GCN layers
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization.
        """
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _compute_normalized_adjacency(self, adjacency_matrix):
        """
        Compute the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
        
        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix A
            
        Returns:
            torch.Tensor: Normalized adjacency matrix
        """
        # Add self-connections (A + I)
        identity = torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)
        adj_with_self_loops = adjacency_matrix + identity
        
        # Compute degree matrix D
        degree = torch.sum(adj_with_self_loops, dim=1)
        
        # Compute D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Create diagonal matrix D^(-1/2)
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # Compute normalized adjacency: D^(-1/2) * A * D^(-1/2)
        normalized_adj = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj_with_self_loops), degree_matrix_inv_sqrt)
        
        return normalized_adj
    
    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass of the GCN.
        
        Args:
            node_features (torch.Tensor): Input node features H(0) of shape (num_nodes, input_dim)
            adjacency_matrix (torch.Tensor): Adjacency matrix A of shape (num_nodes, num_nodes)
            
        Returns:
            torch.Tensor: Output node embeddings of shape (num_nodes, hidden_dim)
        """
        # Compute normalized adjacency matrix
        normalized_adj = self._compute_normalized_adjacency(adjacency_matrix)
        
        # Initialize with input features
        h = node_features
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            # Apply linear transformation: W * H
            h = layer(h)
            
            # Apply graph convolution: D^(-1/2) * A * D^(-1/2) * (W * H)
            h = torch.mm(normalized_adj, h)
            
            # Apply activation function (ReLU) except for the last layer
            if i < len(self.layers) - 1:
                h = F.relu(h)
                # Apply dropout
                h = self.dropout(h)
        
        return h
    
    def get_embeddings(self, node_features, adjacency_matrix):
        """
        Get node embeddings without gradient computation (for inference).
        
        Args:
            node_features (torch.Tensor): Input node features
            adjacency_matrix (torch.Tensor): Adjacency matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(node_features, adjacency_matrix)
        return embeddings


class GCNFeatureLearner:
    """
    Wrapper class for training and using GCN for feature learning.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.4, device='cpu'):
        """
        Initialize the GCN feature learner.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension (default: 128)
            num_layers (int): Number of GCN layers (default: 2)
            dropout_rate (float): Dropout rate (default: 0.4)
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = GCN(input_dim, hidden_dim, num_layers, dropout_rate).to(device)
        
    def train_model(self, node_features, adjacency_matrix, labels=None, epochs=100, lr=0.01):
        """
        Train the GCN model (if supervised learning is needed).
        
        Args:
            node_features (torch.Tensor): Input node features
            adjacency_matrix (torch.Tensor): Adjacency matrix
            labels (torch.Tensor, optional): Node labels for supervised learning
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Move data to device
        node_features = node_features.to(self.device)
        adjacency_matrix = adjacency_matrix.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
            criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(node_features, adjacency_matrix)
            
            if labels is not None:
                # Supervised learning
                loss = criterion(embeddings, labels)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            else:
                # Unsupervised learning (reconstruction loss or other objectives)
                # This can be extended based on specific requirements
                pass
    
    def extract_features(self, node_features, adjacency_matrix):
        """
        Extract features using the trained GCN model.
        
        Args:
            node_features (torch.Tensor): Input node features
            adjacency_matrix (torch.Tensor): Adjacency matrix
            
        Returns:
            torch.Tensor: Extracted node embeddings
        """
        # Move data to device
        node_features = node_features.to(self.device)
        adjacency_matrix = adjacency_matrix.to(self.device)
        
        return self.model.get_embeddings(node_features, adjacency_matrix)


# Example usage
if __name__ == "__main__":
    # Example parameters
    num_nodes = 100
    input_dim = 64
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.1
    
    # Create sample data
    node_features = torch.randn(num_nodes, input_dim)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    
    # Make adjacency matrix symmetric
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
    adjacency_matrix = (adjacency_matrix > 0.5).float()
    
    # Initialize GCN
    gcn_learner = GCNFeatureLearner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        device='cpu'
    )
    
    # Extract features
    embeddings = gcn_learner.extract_features(node_features, adjacency_matrix)
    print(f"Input shape: {node_features.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"GCN model parameters: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout_rate={dropout_rate}")