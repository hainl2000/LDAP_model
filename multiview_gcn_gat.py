import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from main import MyDataset, train
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

# =============================================================================
# Phần 1: GCN Feature Learning (Theo Eq. 7)
# =============================================================================
class GCNBlock(nn.Module):
    """
    Một khối GCN 2 lớp độc lập cho một view, bám sát mô tả 2.3.1.
    Sử dụng GCNConv của PyG để triển khai đúng công thức (7).
    """
    def __init__(self, in_channels, hidden_channels=128, dropout_rate=0.1):
        super(GCNBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj_matrix):
        # Chuyển ma trận kề dày sang định dạng edge_index của PyG
        edge_index, _ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.to(x.device)
        
        # Lớp 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Lớp 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# =============================================================================
# Phần 2: Attention Fusion (Theo Eq. 8 và 9)
# =============================================================================
# class AttentionFusionWithCNN(nn.Module):
#     """
#     Triển khai chính xác cơ chế attention và tổng hợp bằng CNN 2D.
#     - Eq. (8): ωd = σ(fcd(AvgPool(Hd)))
#     - Eq. (9): Hf_d = CNNd(σ(ωd⋅Hd))
#     """
#     def __init__(self, num_views, num_nodes, hidden_dim=128, attention_dim=64, 
#                  fusion_output_dim=128, cnn_out_channels=32, kernel_size=3):
#         super(AttentionFusionWithCNN, self).__init__()
        
#         # --- Triển khai Eq. 8: Tính trọng số Attention ---
#         self.attention_fc = nn.Sequential(
#             nn.Linear(hidden_dim, attention_dim),
#             nn.Tanh(),
#             nn.Linear(attention_dim, 1)
#         )
        
#         # --- Triển khai Eq. 9: Tổng hợp bằng CNN ---
#         # Lớp CNN 2D. Input sẽ được reshape thành (batch, channels, height, width)
#         # Ta coi: batch=1, channels=1, height=num_views, width=num_nodes * hidden_dim
#         self.conv2d = nn.Conv2d(
#             in_channels=1, 
#             out_channels=cnn_out_channels, 
#             kernel_size=(kernel_size, kernel_size), 
#             padding='same'
#         )

#         # Lớp MLP cuối để map feature về đúng kích thước output
#         # Kích thước sau Conv2d là (cnn_out_channels, num_views, num_nodes * hidden_dim)
#         # Ta cần đưa nó về (num_nodes, fusion_output_dim)
#         final_mlp_in_dim = cnn_out_channels * num_views * hidden_dim
#         self.final_mlp = nn.Sequential(
#             nn.Linear(final_mlp_in_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, fusion_output_dim)
#         )
        
#         self.num_nodes = num_nodes

#     def forward(self, view_embeddings_list):
#         # Input: list các tensor shape (num_nodes, hidden_dim)
#         # Stack lại thành (num_views, num_nodes, hidden_dim)
#         stacked_embeddings = torch.stack(view_embeddings_list, dim=0)
#         num_views, num_nodes, hidden_dim = stacked_embeddings.shape

#         # --- Bước 1: Tính Attention Weights (Eq. 8) ---
#         # AvgPool(Hd): Lấy trung bình theo chiều các node
#         # Shape: (num_views, hidden_dim)
#         view_summaries = torch.mean(stacked_embeddings, dim=1)
        
#         # fcd(AvgPool(Hd)): Đưa qua mạng FC để có score
#         # Shape: (num_views, 1) -> (num_views,)
#         attention_scores = self.attention_fc(view_summaries).squeeze(-1)
        
#         # σ(...): Dùng Softmax để có trọng số
#         # Shape: (num_views,)
#         attention_weights = F.softmax(attention_scores, dim=0)
        
#         # --- Bước 2: Áp dụng Attention và CNN (Eq. 9) ---
#         # ωd ⋅ Hd: Nhân trọng số attention
#         # Reshape weights: (num_views, 1, 1) để nhân broadcasting
#         weighted_embeddings = stacked_embeddings * attention_weights.view(num_views, 1, 1)
        
#         # σ(ωd ⋅ Hd): Áp dụng activation
#         activated_embeddings = F.relu(weighted_embeddings)

#         # Chuẩn bị cho CNNd: Reshape tensor
#         # Shape: (num_views, num_nodes, hidden_dim) -> (1, 1, num_views, num_nodes * hidden_dim)
#         # Đây là một cách diễn giải để sử dụng Conv2d
#         conv_input = activated_embeddings.view(1, 1, num_views, num_nodes * hidden_dim)
        
#         # CNNd(...): Đưa qua lớp Conv2d
#         # Shape: (1, cnn_out_channels, num_views, num_nodes * hidden_dim)
#         conv_output = self.conv2d(conv_input)
        
#         # Đưa về shape cuối cùng
#         # Flatten: (1, cnn_out_channels * num_views * num_nodes * hidden_dim)
#         flattened_output = torch.flatten(conv_output)
        
#         # Reshape và đi qua MLP cuối
#         # Chúng ta cần reshape lại để mỗi node có feature riêng
#         # (num_nodes, cnn_out_channels * num_views * hidden_dim)
#         reshaped_for_mlp = conv_output.permute(0, 2, 3, 1).reshape(num_nodes, -1)
        
#         # (num_nodes, fusion_output_dim)
#         fused_features = self.final_mlp(reshaped_for_mlp)
        
#         return fused_features, attention_weights

class AttentionFusion(nn.Module):
    """
    Phiên bản sửa lỗi, không dùng Conv2D.
    Nó tính toán trọng số attention cho mỗi view và sau đó tổng hợp
    các embedding bằng một tổng có trọng số, rồi đưa qua một lớp MLP.
    """
    def __init__(self, num_views, num_nodes, hidden_dim=128, attention_dim=64, 
                 fusion_output_dim=128):
        super(AttentionFusion, self).__init__()
        
        # --- Tính trọng số Attention cho mỗi view ---
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # --- Lớp MLP để tổng hợp các embedding đã được gán trọng số ---
        # Input là embedding được ghép nối từ các view
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_views, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, fusion_output_dim)
        )

    def forward(self, view_embeddings_list):
        stacked_embeddings = torch.stack(view_embeddings_list, dim=0)
        num_views, num_nodes, hidden_dim = stacked_embeddings.shape

        # --- Bước 1: Tính Attention Weights cho mỗi view ---
        view_summaries = torch.mean(stacked_embeddings, dim=1)
        attention_scores = self.attention_fc(view_summaries).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # --- Bước 2: Áp dụng Attention và tổng hợp bằng MLP ---
        # Nhân trọng số attention
        weighted_embeddings = stacked_embeddings * attention_weights.view(num_views, 1, 1)
        
        # Thay vì đưa qua Conv2D, ta ghép nối các embedding đã được gán trọng số
        # và đưa qua một MLP để học cách kết hợp chúng.
        # Permute để có shape (num_nodes, num_views, hidden_dim)
        # rồi reshape thành (num_nodes, num_views * hidden_dim)
        fusion_input = weighted_embeddings.permute(1, 0, 2).reshape(num_nodes, -1)
        
        fused_features = self.fusion_mlp(fusion_input)
        
        return fused_features, attention_weights

# =============================================================================
# Phần 3: Model Tổng Thể
# =============================================================================
class MultiViewFeatureExtractor(nn.Module):
    # def __init__(self, num_nodes, num_views, gcn_hidden_dim=128, fusion_output_dim=128):
    #     super(MultiViewFeatureExtractor, self).__init__()
        
    #     # Đặc trưng đầu vào là ma trận đơn vị
    #     self.initial_features = nn.Parameter(torch.eye(num_nodes), requires_grad=False)
        
    #     # Các khối GCN độc lập cho mỗi view
    #     self.gcn_blocks = nn.ModuleList([
    #         GCNBlock(in_channels=num_nodes, hidden_channels=gcn_hidden_dim)
    #         for _ in range(num_views)
    #     ])
        
    #     # Module tổng hợp feature bằng Attention và CNN
    #     self.fusion_module = AttentionFusionWithCNN(
    #         num_views=num_views,
    #         num_nodes=num_nodes,
    #         hidden_dim=gcn_hidden_dim,
    #         fusion_output_dim=fusion_output_dim
    #     )

    def __init__(self, num_nodes, num_views, gcn_hidden_dim=128, fusion_output_dim=128):
        super(MultiViewFeatureExtractor, self).__init__()
        
        self.initial_features = nn.Parameter(torch.eye(num_nodes), requires_grad=False)
        
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(in_channels=num_nodes, hidden_channels=gcn_hidden_dim)
            for _ in range(num_views)
        ])
        
        # SỬA Ở ĐÂY: Đổi tên class thành AttentionFusion
        self.fusion_module = AttentionFusion(
            num_views=num_views,
            num_nodes=num_nodes,
            hidden_dim=gcn_hidden_dim,
            fusion_output_dim=fusion_output_dim
        )

    def forward(self, adjacency_matrices_list):
        # 1. Trích xuất feature từ mỗi view bằng GCN
        view_embeddings = [
            gcn(self.initial_features, adj) 
            for gcn, adj in zip(self.gcn_blocks, adjacency_matrices_list)
        ]
        
        # 2. Tổng hợp các feature bằng module fusion
        fused_features, attention_weights = self.fusion_module(view_embeddings)
        
        return fused_features, attention_weights, view_embeddings


def reconstruct_similarity_matrix(embeddings):
    """
    Tái tạo lại ma trận tương đồng từ các vector embedding.
    
    Args:
        embeddings (torch.Tensor): Ma trận embedding có shape (num_nodes, embedding_dim).
        
    Returns:
        torch.Tensor: Ma trận tương đồng mới có shape (num_nodes, num_nodes)
                      với các giá trị trong khoảng [0, 1].
    """
    # Đảm bảo đầu vào là tensor PyTorch
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
        
    # Tính tích vô hướng: H * H^T
    # torch.matmul thực hiện phép nhân ma trận
    dot_product = torch.matmul(embeddings, embeddings.T)
    
    # Áp dụng hàm Sigmoid để chuẩn hóa giá trị về khoảng [0, 1]
    similarity_matrix = torch.sigmoid(dot_product)
    
    return similarity_matrix    


def concatenate(lnclen, dilen, milen, lnc_di, lnc_mi, mi_di, lncSiNet, diSiNet, miSiNet):
    A = np.zeros((lnclen + dilen + milen, lnclen + dilen + milen))
    A[: lnclen, lnclen: lnclen + dilen] = lnc_di
    A[lnclen: lnclen + dilen, : lnclen] = lnc_di.T
    A[: lnclen, lnclen + dilen: ] = lnc_mi
    A[lnclen + dilen: , : lnclen] = lnc_mi.T
    A[lnclen: lnclen + dilen, lnclen + dilen: ] = mi_di.T
    A[lnclen + dilen: , lnclen: lnclen + dilen] = mi_di
    A[: lnclen, : lnclen] = lncSiNet
    A[lnclen: lnclen + dilen, lnclen: lnclen + dilen] = diSiNet
    A[lnclen + dilen: , lnclen + dilen: ] = miSiNet
    return A

def cal_fused_features(num_nodes, num_views, adjacency_matrices, gcn_hidden_dim=128, fusion_output_dim=128):
    model = MultiViewFeatureExtractor(
        num_nodes=num_nodes,
        num_views=num_views,
        gcn_hidden_dim=gcn_hidden_dim,
        fusion_output_dim=fusion_output_dim
    )
    fused_features, attention_weights, individual_embeddings = model(adjacency_matrices)
    return reconstruct_similarity_matrix(fused_features)



# Laplace normalization
class LplsNorm(nn.Module):
    def __init__(self):
        super(LplsNorm, self).__init__()

    def forward(self, adjacency_matrix):
        degree = torch.sum(adjacency_matrix, dim=-1)
        diag = torch.diag(torch.pow(degree, -0.5))
        out = diag.mm(adjacency_matrix).mm(diag)
        return out

# convolutional
class Convolution(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Convolution, self).__init__()
        self.LinearLayer = nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adjacency_matrix, feature):
        mf = feature + torch.mm(adjacency_matrix, feature)
        out = self.LinearLayer(mf)
        return out

class VGAE_Encoder(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, embedding_dimension):
        super(VGAE_Encoder, self).__init__()
        self.lpls_norm = LplsNorm()
        
        # Lớp GCN cơ sở
        self.base_gcn = Convolution(in_dimension, hidden_dimension)
        
        # Hai lớp tuyến tính để tạo ra mean và log_var
        self.gcn_mean = Convolution(hidden_dimension, embedding_dimension)
        self.gcn_log_var = Convolution(hidden_dimension, embedding_dimension)

    def forward(self, adj_matrix, features):
        # 1. Chuẩn hóa ma trận kề
        norm_adj = self.lpls_norm(adj_matrix)
        
        # 2. Đưa qua lớp GCN cơ sở
        hidden = self.base_gcn(norm_adj, features)
        
        # 3. Tính mean và log_var
        mean = self.gcn_mean(norm_adj, hidden)
        log_var = self.gcn_log_var(norm_adj, hidden)
        
        return mean, log_var


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        # Tái tạo ma trận kề bằng cách lấy z nhân với chuyển vị của nó
        z_t = z.t()
        adj_reconstructed = torch.mm(z, z_t)
        return torch.sigmoid(adj_reconstructed)

class VGAE_Model(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, embedding_dimension):
        super(VGAE_Model, self).__init__()
        self.encoder = VGAE_Encoder(in_dimension, hidden_dimension, embedding_dimension)
        self.decoder = InnerProductDecoder()

    def reparameterize(self, mean, log_var):
        # Thủ thuật tái tham số hóa
        if self.training:
            std = torch.exp(0.5 * log_var)
            # Lấy mẫu từ phân phối chuẩn N(0, 1)
            eps = torch.randn_like(std) 
            return eps.mul(std).add_(mean)
        else:
            # Khi kiểm tra hoặc suy luận, chỉ cần dùng mean
            return mean

    def forward(self, adj_matrix, features):
        # Mã hóa để lấy mean và log_var
        mean, log_var = self.encoder(adj_matrix, features)
        
        # Lấy mẫu z từ không gian tiềm ẩn
        z = self.reparameterize(mean, log_var)
        
        # Giải mã để tái tạo ma trận kề
        adj_reconstructed = self.decoder(z)
        
        return adj_reconstructed, mean, log_var

def vgae_loss_function(preds, labels, mean, log_var, num_nodes, pos_weight):
    # 1. Reconstruction Loss
    # Dùng Binary Cross Entropy với Logits để ổn định hơn
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        preds.view(-1), labels.view(-1), pos_weight=pos_weight
    )
    
    # 2. KL Divergence Loss
    # Công thức tính KL divergence giữa N(mean, var) và N(0, 1)
    kl_divergence = -0.5 / num_nodes * torch.mean(torch.sum(
        1 + 2 * log_var - mean.pow(2) - (2 * log_var).exp(), dim=1
    ))
    
    return reconstruction_loss + kl_divergence


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
    for ij in positive_ij[positive5foldsidx[fold]['test']]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0


    lnc_mi = pd.read_csv('./our_dataset/' + dataset + '/interaction/lnc_mi.csv', index_col='0').values

    mi_di = pd.read_csv('./our_dataset/' + dataset + '/interaction/mi_di.csv')
    mi_di.set_index('0', inplace=True)
    mi_di = mi_di.values
    
    # # Giả lập cho bài toán disease
    num_diseases = di_semantic_similarity.shape[0]
    num_lnc = lnc_gip_similarity.shape[0]
    num_mi = mi_gip_similarity.shape[0]
    num_views = 2

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
    vgae_model = VGAE_Model(in_dimension, hidden_dimension, embedding_dimension) # Giả sử dùng VGAE_Model đã định nghĩa
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    
    d = [] # Lưu kết quả tái tạo của fold hiện tại
    e = [] # Lưu vector tiềm ẩn 'mu' của fold hiện tại

    epoch = 0
    num_nodes = A.shape[0]

    # CẢI TIẾN 2: Tạo ma trận kề với self-loops
    adj_input = A + torch.eye(num_nodes)
    
    # CẢI TIẾN 3: Sử dụng ma trận đơn vị làm features ban đầu
    features_input = torch.eye(num_nodes)

    # (Tùy chọn nhưng khuyến khích) Tính pos_weight để xử lý đồ thị thưa
    pos_weight = torch.tensor(float(num_nodes**2 - adj_input.sum()) / adj_input.sum())

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
        
    model.eval()
    with torch.no_grad():
        # Sử dụng cùng input đã xử lý để đánh giá
        rd, mu, log_var = model(adj_input, features_input)
        d.append(rd.cpu().numpy()) # Chuyển về numpy để lưu trữ nếu cần
        # Lưu 'mu' làm vector biểu diễn tiềm ẩn
        e.append(mu.cpu().numpy())
    
        train_dataset = MyDataset(
            network_num, fold, train_positive_ij, train_negative_ij, "训练", dataset, file_name
        )
        test_dataset = MyDataset(
            network_num, fold, test_positive_ij, test_negative_ij, "测试", dataset, file_name
        )
        # Setting Model Parameters
        # dataset1
        hidden_dimension = 40
        hiddenLayer_num = 5
        drop_rate = 0.1
        batch_size = 32
        epochs = 5
        use_aggregate = True
        learn_rate = 1e-2
        weight_decay = 1e-4

        for epoch in epochs:
            for i in range(test_times):
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