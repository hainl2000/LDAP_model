import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from construct_multiview_gcn_gat import cal_fused_features, concatenate
from vgae_model import VGAE_Model
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

class MyDataset(Dataset):
    def get_data(self, Ai, A_encoders, ij):
        data = []
        feature = np.array(
            
        )
        for item in ij:
            feature = np.array(
                [
                    np.hstack((Ai[0][item[0]], A_encoders[0][item[0]])),
                    np.hstack((Ai[0][item[1]], A_encoders[0][item[1]])),
                ]
            )
            for dim in range(1, Ai.shape[0]):
                feature = np.concatenate(
                    (
                        feature,
                        np.array(
                            [
                                np.hstack((Ai[dim][item[0]], A_encoders[dim][item[0]])),
                                np.hstack((Ai[dim][item[1]], A_encoders[dim][item[1]])),
                            ]
                        ),
                    )
                )
        data.append(feature)
        return np.array(data)

    def __init__(self, A, A_encoder, fold, positive_ij, negative_ij, mode, dataset):
        super().__init__()
        Ai = []
        A_encoders = []
        Ai.append(A)
        A_encoders.append(A_encoder)
        # A contains the adjacency matrices, e contains the encoded features
        Ai = np.array(Ai)
        A_encoders = np.array(A_encoders)
        positive_data = torch.Tensor(self.get_data(Ai, A_encoders, positive_ij))
        negative_data = torch.Tensor(self.get_data(Ai, A_encoders, negative_ij))

        data = torch.cat((positive_data, negative_data)).transpose(2, 1)
        self.data = data
        self.target = torch.Tensor(
            [1] * positive_data.shape[0] + [0] * negative_data.shape[0]
        )
        print(f"{dataset} {mode} the data is loaded and the shape is {data.shape}")

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def vgae_loss_function(reconstructed_adj, original_adj, mu, log_var, num_nodes, pos_weight, kl_weight=0.1):
    """
    VGAE loss function combining reconstruction loss and KL divergence loss.
    
    Args:
        reconstructed_adj: Reconstructed adjacency matrix
        original_adj: Original adjacency matrix
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        num_nodes: Number of nodes in the graph
        pos_weight: Weight for positive samples in reconstruction loss
        kl_weight: Weight for KL divergence term
    
    Returns:
        Combined loss value
    """
    # Reconstruction loss (Binary Cross Entropy)
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstructed_adj.view(-1), 
        original_adj.view(-1), 
        pos_weight=pos_weight
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / num_nodes
    
    # Combined loss
    total_loss = reconstruction_loss + kl_weight * kl_loss
    
    return total_loss

def train(
    dataset,
    hidden_dimension,
    hiddenLayer_num,
    drop_rate,
    use_aggregate,
    batch_size,
    epochs,
    device,
    learn_rate,
    weight_decay,
):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, drop_last=False, pin_memory=True
    )

    feature_num = dataset.data.shape[1]
    input_dimension = dataset.data.shape[2]
    print(f"input_dimension: {input_dimension}; feature_num: {feature_num}")
    model = LDAGM(
        input_dimension,
        hidden_dimension,
        feature_num,
        hiddenLayer_num,
        drop_rate=drop_rate,
        use_aggregate=use_aggregate,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    epoch = 0
    loss_record = []
    while epoch < epochs:
        model.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre = model(x)
            loss = loss_fn(pre, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().cpu().item())
        epoch += 1
        print(f"In round {epoch}, the loss is: {loss.detach().cpu().item()}")
    print("End of training")
    return loss_record, model

def test(model, test_dataset, batch_size, device):
    """
    Test function for model evaluation.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        batch_size: Batch size for testing
        device: Device to run on
    
    Returns:
        test_targets: True labels
        predictions: Model predictions
    """
    model.eval()
    test_targets = []
    predictions = []
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            # Assuming batch contains features and labels
            if isinstance(batch, (list, tuple)):
                features, labels = batch[0].to(device), batch[1].to(device)
            else:
                features = batch.to(device)
                labels = batch.y.to(device) if hasattr(batch, 'y') else None
            
            outputs = model(features)
            
            if labels is not None:
                test_targets.extend(labels.cpu().numpy())
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    
    return test_targets, predictions

if __name__ == '__main__':
    dataset = "dataset2"
    node_type = "disease"
    fold = 0
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

    di_semantic_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_func_similarity.npy")

    lnc_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_di.csv')
    lnc_di.set_index('0', inplace=True)
    lnc_di = lnc_di.values
    lnc_di_copy = copy.copy(lnc_di)


    lnc_mi = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_mi.csv', index_col='0').values

    mi_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/mi_di.csv')
    mi_di.set_index('0', inplace=True)
    mi_di = mi_di.values
    
    # # Giả lập cho bài toán disease
    num_diseases = di_semantic_similarity.shape[0]
    num_lnc = lnc_gip_similarity.shape[0]
    num_mi = mi_gip_similarity.shape[0]
    num_views = 2
    
    # Define missing variables
    lncRNALen = num_lnc
    network_num = 1
    embedding_dimension = 16
    n_epochs = 100
    file_name = f"A_encoder_{embedding_dimension}_{network_num}_{n_epochs}"
    test_times = 10
    
    # Fix the lnc_di_copy processing
    for ij in positive_ij[positive5foldsidx[fold]['test']]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0

    disease_adjacency_matrices = [torch.tensor(di_semantic_similarity, dtype=torch.float32).to(device), 
                          torch.tensor(di_gip_similarity, dtype=torch.float32).to(device)]
    lnaRNA_adjacency_matrices = [torch.tensor(lnc_gip_similarity, dtype=torch.float32).to(device), 
                          torch.tensor(lnc_func_similarity, dtype=torch.float32).to(device)]
    miRNA_adjacency_matrices = [torch.tensor(mi_gip_similarity, dtype=torch.float32).to(device), 
                          torch.tensor(mi_func_similarity, dtype=torch.float32).to(device)]
    
    disease_fused_features = cal_fused_features(num_diseases, num_views, disease_adjacency_matrices)
    lnc_fused_features = cal_fused_features(num_lnc, num_views, lnaRNA_adjacency_matrices)
    mi_fused_features = cal_fused_features(num_mi, num_views, miRNA_adjacency_matrices)

    A_numpy = concatenate(num_lnc, num_diseases, num_mi, lnc_di_copy, lnc_mi, mi_di, lnc_fused_features, disease_fused_features, mi_fused_features)
    A = torch.Tensor(A_numpy)
    A = A.to(device)

    in_dimension = A.shape[1]
    embedding_dimension = 16
    learn_rate = 1e-2
    weight_decay = 1e-4
    n_epochs = 100
    hidden_dimension=32

    rds = []
    res = []
    vgae_model = VGAE_Model(in_dimension, hidden_dimension, embedding_dimension).to(device) # Move model to device
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    
    d = [] # Lưu kết quả tái tạo của fold hiện tại
    e = [] # Lưu vector tiềm ẩn 'mu' của fold hiện tại

    epoch = 0
    num_nodes = A.shape[0]

    # CẢI TIẾN 2: Tạo ma trận kề với self-loops
    adj_input = A + torch.eye(num_nodes).to(device)
    
    # CẢI TIẾN 3: Sử dụng ma trận đơn vị làm features ban đầu
    features_input = torch.eye(num_nodes).to(device)

    # (Tùy chọn nhưng khuyến khích) Tính pos_weight để xử lý đồ thị thưa
    pos_weight = torch.tensor(float(num_nodes**2 - adj_input.sum()) / adj_input.sum()).to(device)

    while epoch < n_epochs:
        vgae_model.train()
        optimizer.zero_grad()
        
        # Forward pass với input đã được xử lý
        reconstructed_A, mu, log_var = vgae_model(adj_input, features_input)
        
        # CẢI TIẾN 4: Gọi hàm loss với đầy đủ tham số
        loss = vgae_loss_function(reconstructed_A, adj_input, mu, log_var, num_nodes, pos_weight)
        
        loss.backward()
        optimizer.step()
        epoch += 1
        
    # Evaluation after VGAE training
    vgae_model.eval()
    with torch.no_grad():
        # Sử dụng cùng input đã xử lý để đánh giá
        rd, mu, log_var = vgae_model(adj_input, features_input)
        d.append(rd.cpu().numpy()) # Chuyển về numpy để lưu trữ nếu cần
        # Lưu 'mu' làm vector biểu diễn tiềm ẩn
        e.append(mu.cpu().numpy())
    
    # Create datasets for downstream task
    train_dataset = MyDataset(
        d, e, fold, train_positive_ij, train_negative_ij, "训练", dataset
    )
    test_dataset = MyDataset(
         d, e, fold, test_positive_ij, test_negative_ij, "测试", dataset
     )
     
    # Setting Model Parameters for downstream task
    # dataset1
    hidden_dimension = 40
    hiddenLayer_num = 5
    drop_rate = 0.1
    batch_size = 32
    epochs = 5
    use_aggregate = True
    learn_rate = 1e-2
    weight_decay = 1e-4

    for epoch in range(epochs):
        print(f"--- Running iteration {i+1}/10 ---")
        loss_record, model = train(
            train_dataset,
            hidden_dimension,
            hiddenLayer_num,
            drop_rate,
            use_aggregate,
            batch_size,
            epoch,
            device,
            learn_rate,
            weight_decay,
        )
        # test
        test_target, pre_target = test(model, test_dataset, batch_size, device)
        # np.save("./result/" + dataset + "/label", test_target)
        # np.save("./result/" + dataset + "/predict", pre_target)
        test_target = np.array(test_target)
        # Getting a specific score
        AUC = roc_auc_score(test_target, pre_target)
        precision, recall, _ = precision_recall_curve(test_target, pre_target)

        fpr, tpr, thresholds = roc_curve(test_target, pre_target)
        AUPR = auc(recall, precision)
        preds = np.array([1 if p > 0.5 else 0 for p in pre_target])
        # preds = np.array([1 if p > 0.5 else 0 for p in pre_target])
        MCC = matthews_corrcoef(test_target, preds)
        ACC = accuracy_score(test_target, preds)
        P = precision_score(test_target, preds)
        R = recall_score(test_target, preds)
        F1 = f1_score(test_target, preds)
        print(f"Iteration {i+1} scores: {AUC}, {AUPR}, {MCC}, {ACC}, {P}, {R}, {F1}")