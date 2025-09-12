#!/usr/bin/env python3
"""
GATE (Graph Attention Auto-Encoders) Parameter Testing Script
Converted from Jupyter notebook with fixes for RuntimeError.

This script tests different parameter combinations for the GATE model:
- embedding_dimension: [16, 32, 64, 128, 256, 512, 1024]
- network_num: [1, 2, 3, 4]
- n_epochs: [30, 50, 70, 100]

Results are saved in structured folders: dataset2/A_encoder_{embedding_dimension}_{network_num}_{n_epochs}
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import itertools


class GATE(nn.Module):
    """Graph Attention Auto-Encoder model implementation."""
    
    def __init__(self, in_dimension, hidden_dims, embedding_size):
        """Initialize GATE model with specified dimensions.
        
        Args:
            in_dimension: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            embedding_size: Final embedding dimension
        """
        super(GATE, self).__init__()
        all_dims = [in_dimension] + hidden_dims + [embedding_size]
        self.n_layers = len(all_dims) - 1

        # Define weight matrices W for encoder/decoder layers
        self.W = nn.ModuleList([
            nn.Linear(all_dims[i], all_dims[i+1]) for i in range(self.n_layers)
        ])

        # Define attention vectors v for attention mechanism
        self.v = nn.ParameterList()
        for i in range(self.n_layers):
            # Each attention layer has 2 vectors v
            v_layer = nn.ParameterDict({
                '0': nn.Parameter(torch.randn(all_dims[i+1], 1)),
                '1': nn.Parameter(torch.randn(all_dims[i+1], 1))
            })
            self.v.append(v_layer)

    def _encoder(self, A, H, layer_idx):
        """Encoder layer with graph attention mechanism.
        
        Args:
            A: Adjacency matrix (sparse tensor)
            H: Node features
            layer_idx: Current layer index
            
        Returns:
            Encoded features and attention matrix
        """
        H = self.W[layer_idx](H)
        # Calculate attention matrix C
        C = self.graph_attention_layer(A, H, self.v[layer_idx])
        # Apply attention
        return torch.sparse.mm(C, H), C

    def _decoder(self, H, C, layer_idx):
        """Decoder layer with attention matrix from encoder.
        
        Args:
            H: Hidden features
            C: Attention matrix from encoder
            layer_idx: Current layer index
            
        Returns:
            Decoded features
        """
        # Multiply with transposed weight matrix
        # F.linear(input, weight.T) is equivalent to input @ weight
        H = F.linear(H, self.W[layer_idx].weight.t())
        # Apply attention C from encoder
        return torch.sparse.mm(C, H)

    def graph_attention_layer(self, A, M, v):
        """Graph attention mechanism implementation.
        
        Args:
            A: Adjacency matrix (sparse tensor)
            M: Feature matrix after linear transformation
            v: Attention parameter vectors
            
        Returns:
            Normalized attention matrix
        """
        # A must be a sparse tensor
        # M is feature matrix after Linear layer
        # Ensure A is coalesced before accessing indices
        A = A.coalesce()
        
        with torch.no_grad():
            f1 = M @ v['0']
            f2 = M @ v['1']
            logits_indices = A.indices()
            logits_values = A.values() * (f1.squeeze(1)[logits_indices[0]] + f2.squeeze(1)[logits_indices[1]])
            logits_uncoalesced = torch.sparse_coo_tensor(logits_indices, logits_values, A.shape)
            logits = logits_uncoalesced.coalesce()

        # Use sigmoid as in original code, instead of LeakyReLU as in GAT paper
        unnormalized_attentions = torch.sparse_coo_tensor(logits.indices(), torch.sigmoid(logits.values()), logits.shape)
        
        # Normalize using sparse_softmax
        attentions = torch.sparse.softmax(unnormalized_attentions, dim=1)
        return attentions

    def forward(self, A, X):
        """Forward pass through GATE model.
        
        Args:
            A: Adjacency matrix (sparse tensor)
            X: Input features (dense tensor)
            
        Returns:
            Reconstructed features and final embeddings
        """
        # A must be a sparse tensor, X is dense tensor
        
        # --- Encoder ---
        H = X
        attention_matrices = []  # Store attention matrices
        for i in range(self.n_layers):
            H, C = self._encoder(A, H, i)
            attention_matrices.append(C)
        
        # H is final node representations
        final_H = H

        # --- Decoder ---
        # Run in reverse and reuse stored attention matrices
        for i in range(self.n_layers - 1, -1, -1):
            C = attention_matrices[i]
            H = self._decoder(H, C, i)
        
        X_reconstructed = H

        return X_reconstructed, final_H


def gate_loss_function(X_reconstructed, X_original, H_final, R_indices, S_indices, lambda_):
    """Loss function for GATE model, converted from TensorFlow code.
    
    Args:
        X_reconstructed: Reconstructed features
        X_original: Original features
        H_final: Final embeddings
        R_indices: Target node indices
        S_indices: Source node indices
        lambda_: Structure loss weight
        
    Returns:
        Total loss value
    """
    # 1. Feature reconstruction loss (Equivalent to tf.sqrt(tf.reduce_sum...))
    # This is Frobenius norm, torch.linalg.norm is efficient way to calculate.
    features_loss = torch.linalg.norm(X_original - X_reconstructed)

    # 2. Structure reconstruction loss (Equivalent to -tf.log(tf.sigmoid(...)))
    # Get embeddings of nodes corresponding to edges
    S_emb = H_final[S_indices]
    R_emb = H_final[R_indices]
    
    # Calculate dot product to get score for each edge
    positive_scores = torch.sum(S_emb * R_emb, dim=-1)
    
    # Use F.binary_cross_entropy_with_logits for numerical stability
    # instead of manually calculating -log(sigmoid(x)).
    # It calculates loss for real edges (positive samples), target is label 1.
    structure_loss = F.binary_cross_entropy_with_logits(
        positive_scores, 
        torch.ones_like(positive_scores),
        reduction='sum'
    )
    
    # Total loss
    total_loss = features_loss + lambda_ * structure_loss
    return total_loss


def load_dataset(dataset_name):
    """Load dataset files and return interaction matrices and dimensions.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'dataset2')
        
    Returns:
        Tuple of (lnc_mi, mi_di, lnc_di, L_num, D_num, M_num)
    """
    lnc_mi = pd.read_csv(f'./our_dataset/{dataset_name}/interaction/lnc_mi.csv', index_col='0').values
    mi_di = pd.read_csv(f'./our_dataset/{dataset_name}/interaction/mi_di.csv', index_col='0').values
    lnc_di = pd.read_csv(f'./our_dataset/{dataset_name}/interaction/lnc_di.csv', index_col='0').values
    L_num = lnc_mi.shape[0]
    D_num = mi_di.shape[1]
    M_num = mi_di.shape[0]
    print(f"Dataset dimensions - L: {L_num}, D: {D_num}, M: {M_num}")
    return lnc_mi, mi_di, lnc_di, L_num, D_num, M_num


def train_gate_model(dataset, embedding_dim, network_num, n_epochs, output_dir):
    """Train GATE model with specified parameters.
    
    Args:
        dataset: Dataset name
        embedding_dim: Embedding dimension
        network_num: Number of networks
        n_epochs: Number of training epochs
        output_dir: Directory to save results
        
    Returns:
        Training success status and final loss
    """
    # Load adjacency matrix
    A = np.load(f'./our_dataset/{dataset}/A/A_1_1.npy')
    folds = 5
    in_dimension = A.shape[1]
    
    # Fixed parameters
    learn_rate = 1e-4
    weight_decay = 1e-5
    hidden_dimension = []
    lambda_param = 1
    
    rds = []
    res = []
    training_losses = []
    
    try:
        for fold in range(folds):
            print(f"Fold {fold + 1}")
            d = []
            e = []
            fold_losses = []
            
            for i in range(network_num):
                epoch = 0
                A_numpy = np.load(f'./our_dataset/{dataset}/A/A_{fold + 1}_{i + 1}.npy')
                
                # Create feature matrix (dense tensor) as input features
                A_features = torch.Tensor(A_numpy)

                edge_indices = torch.from_numpy(np.vstack(A_numpy.nonzero()))
                edge_values = torch.ones(edge_indices.shape[1], dtype=torch.float32)
                A_sparse = torch.sparse_coo_tensor(edge_indices, edge_values, A_numpy.shape).coalesce()
                source_nodes_indices = edge_indices[0]  # Source nodes (S)
                target_nodes_indices = edge_indices[1]  # Target nodes (R)

                # Initialize GATE model
                model = GATE(in_dimension, hidden_dimension, embedding_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
                epoch_losses = []
                while epoch < n_epochs:
                    model.train()
                    optimizer.zero_grad()
                
                    reconstructed_A, final_embeddings = model(A_sparse, A_features)
                    
                    # Calculate GATE loss
                    loss = gate_loss_function(reconstructed_A, A_features, final_embeddings, 
                                            target_nodes_indices, source_nodes_indices, lambda_param)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.detach().cpu().item())
                    epoch += 1

                model.eval()
                with torch.no_grad():
                    rd, re = model(A_sparse, A_features)
                    d.append(rd)
                    e.append(re)
                    
                fold_losses.append(epoch_losses)
                print(f"  Fold {fold + 1}, Network {i + 1} final loss: {loss.detach().cpu().item():.6f}")
            
            res.append(e)
            rds.append(d)
            training_losses.append(fold_losses)
    except Exception as e:
        print(e)
    # Save results
    final_embeddings_np = torch.stack([torch.stack(fold_embs).mean(dim=0) for fold_embs in res]).mean(dim=0).detach().cpu().numpy()
    np.save(os.path.join(output_dir, 'embeddings.npy'), final_embeddings_np)
    np.save(os.path.join(output_dir, 'training_losses.npy'), np.array(training_losses))
    
    # Save experiment info
    with open(os.path.join(output_dir, 'experiment_info.txt'), 'w') as f:
        f.write(f"Embedding Dimension: {embedding_dim}\n")
        f.write(f"Network Number: {network_num}\n")
        f.write(f"Number of Epochs: {n_epochs}\n")
        f.write(f"Learning Rate: {learn_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Lambda Parameter: {lambda_param}\n")
        f.write(f"Input Dimension: {in_dimension}\n")
        f.write(f"Dataset: {dataset}\n")
    
    # Save summary statistics
    final_loss = np.mean([fold_losses[-1][-1] for fold_losses in training_losses])
    with open(os.path.join(output_dir, 'summary_stats.txt'), 'w') as f:
        f.write(f"Final Average Loss: {final_loss:.6f}\n")
        f.write(f"Embedding Shape: {final_embeddings_np.shape}\n")
        f.write(f"Total Training Time: {n_epochs * folds * network_num} iterations\n")
    
    return True, final_loss


def analyze_results(embedding_dimensions, network_nums, n_epochs_list):
    """Analyze results from all parameter combinations.
    
    Args:
        embedding_dimensions: List of embedding dimensions tested
        network_nums: List of network numbers tested
        n_epochs_list: List of epoch numbers tested
        
    Returns:
        DataFrame with results summary
    """
    results_summary = []

    for embedding_dim in embedding_dimensions:
        for network_num in network_nums:
            for n_epochs in n_epochs_list:
                output_dir = f"./our_dataset/dataset2/A_encoder_{embedding_dim}_{network_num}_{n_epochs}"
                
                if os.path.exists(os.path.join(output_dir, 'summary_stats.txt')):
                    with open(os.path.join(output_dir, 'summary_stats.txt'), 'r') as f:
                        content = f.read()
                        # Extract final loss
                        for line in content.split('\n'):
                            if 'Final Average Loss:' in line:
                                final_loss = float(line.split(':')[1].strip())
                                results_summary.append({
                                    'embedding_dim': embedding_dim,
                                    'network_num': network_num,
                                    'n_epochs': n_epochs,
                                    'final_loss': final_loss
                                })
                                break

    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results_summary)
    return results_df


def main():
    """Main function to run parameter testing."""
    print("GATE Parameter Testing Script")
    print("=============================")
    
    # Load dataset
    dataset = "dataset2"
    lnc_mi, mi_di, lnc_di, L_num, D_num, M_num = load_dataset(dataset)
    
    # Parameter combinations for testing (updated based on
    #  your notebook)
    embedding_dimensions = [1024]
    network_nums = [1,2,3, 4]
    n_epochs_list = [30, 50, 70, 100]
    
    print("\nParameter Testing Configuration:")
    print(f"Embedding dimensions: {embedding_dimensions}")
    print(f"Network numbers: {network_nums}")
    print(f"Epoch numbers: {n_epochs_list}")
    
    # Create all parameter combinations
    parameter_combinations = list(itertools.product(embedding_dimensions, network_nums, n_epochs_list))
    print(f"\nTotal parameter combinations: {len(parameter_combinations)}")
    
    print("\nStarting parameter testing on 1024...")
    
    for combo_idx, (embedding_dim, network_num, n_epochs) in enumerate(parameter_combinations):
        print(f"\nTesting combination {combo_idx + 1}/{len(parameter_combinations)}: "
              f"embedding_dim={embedding_dim}, network_num={network_num}, n_epochs={n_epochs}")
        
        # Create output directory
        output_dir = f"./our_dataset/dataset2/A_encoder_{embedding_dim}_{network_num}_{n_epochs}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print('Here')
            success, final_loss = train_gate_model(dataset, embedding_dim, network_num, n_epochs, output_dir)
            if success:
                print(f"  Results saved to: {output_dir}")
                print(f"  Final loss: {final_loss:.6f}")
            
        except Exception as e:
            # Save error log
            with open(os.path.join(output_dir, 'error_log.txt'), 'w') as f:
                f.write(f"Error occurred during training:\n{str(e)}\n")
                f.write(f"Parameters: embedding_dim={embedding_dim}, network_num={network_num}, n_epochs={n_epochs}\n")
            print(f"  Error in combination {combo_idx + 1}: {str(e)}")
            continue
    
    print("\nParameter testing completed!")
    
    # Analyze results
    print("\nAnalyzing results...")
    results_df = analyze_results(embedding_dimensions, network_nums, n_epochs_list)
    
    if len(results_df) > 0:
        print("\nResults Summary:")
        print(results_df.head(10))
        
        # Find best parameters
        best_result = results_df.loc[results_df['final_loss'].idxmin()]
        print(f"\nBest parameters:")
        print(f"Embedding Dimension: {best_result['embedding_dim']}")
        print(f"Network Number: {best_result['network_num']}")
        print(f"Number of Epochs: {best_result['n_epochs']}")
        print(f"Final Loss: {best_result['final_loss']:.6f}")
        
        # Save results summary
        results_df.to_csv('./our_dataset/dataset2/parameter_testing_results.csv', index=False)
        print("\nResults summary saved to ./sample/dataset2/parameter_testing_results.csv")
    else:
        print("No successful results found. Check error logs in individual directories.")


if __name__ == "__main__":
    main()