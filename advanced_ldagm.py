import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Aggregation of data in the hidden layer
class AggregateLayer(nn.Module):
    def __init__(self, input_dimension):
        super(AggregateLayer, self).__init__()
        self.ForgetGateLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension), nn.Sigmoid()
        )

        self.InputGateLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension), nn.Sigmoid()
        )

        self.UpdateGateLayer = nn.Sequential(
            nn.Linear(input_dimension * 2, input_dimension), nn.Sigmoid()
        )

    def forward(self, current_data, previous_data):
        forget_gate = self.ForgetGateLayer(previous_data)
        forget_data = previous_data * forget_gate

        input_gate = self.InputGateLayer(current_data)
        input_data = current_data * input_gate

        integer_data = torch.cat([input_data, forget_data], dim=-1)
        update_gate = self.UpdateGateLayer(integer_data)
        update_data = update_gate * input_data + (1 - update_gate) * forget_data
        return update_data


class EnhancedAttentionAggregateLayer(nn.Module):
    """
    Enhanced aggregation layer using multi-head attention mechanism
    for better information fusion between current and previous representations.
    """
    def __init__(self, input_dimension, num_heads=4, dropout=0.1):
        super(EnhancedAttentionAggregateLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_heads = num_heads
        self.head_dim = input_dimension // num_heads
        
        # Ensure divisibility
        if input_dimension % num_heads != 0:
            self.head_dim = input_dimension // num_heads + 1
            self.adjusted_dim = self.head_dim * num_heads
            self.input_projection = nn.Linear(input_dimension, self.adjusted_dim)
        else:
            self.adjusted_dim = input_dimension
            self.input_projection = None
        
        # Multi-head attention components
        self.query_projection = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.key_projection = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.value_projection = nn.Linear(self.adjusted_dim, self.adjusted_dim)
        self.output_projection = nn.Linear(self.adjusted_dim, input_dimension)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(input_dimension)
        self.dropout = nn.Dropout(dropout)
        
        # Gate mechanism for residual connection
        self.gate = nn.Sequential(
            nn.Linear(input_dimension * 2, input_dimension),
            nn.Sigmoid()
        )
        
    def forward(self, current_data, previous_data):
        batch_size, seq_len, _ = current_data.shape
        
        # Project to adjusted dimension if needed
        if self.input_projection is not None:
            current_proj = self.input_projection(current_data)
            previous_proj = self.input_projection(previous_data)
        else:
            current_proj = current_data
            previous_proj = previous_data
        
        # Project to query, key, value
        Q = self.query_projection(current_proj)
        K = self.key_projection(previous_proj)
        V = self.value_projection(previous_proj)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.adjusted_dim
        )
        
        # Output projection back to original dimension
        attended_output = self.output_projection(attended_values)
        
        # Gated residual connection
        combined_input = torch.cat([current_data, attended_output], dim=-1)
        gate_weights = self.gate(combined_input)
        
        # Apply gate and layer normalization
        output = gate_weights * attended_output + (1 - gate_weights) * current_data
        output = self.layer_norm(output)
        
        return output


class HiddenLayer(nn.ModuleList):
    def __init__(self, input_dimension, drop_rate):
        super(HiddenLayer, self).__init__()

        self.LinearLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension // 2),
            nn.ReLU(),
            nn.Linear(input_dimension // 2, input_dimension),
            nn.Dropout(drop_rate),
        )

    def forward(self, data):
        data = self.LinearLayer(data)
        return data


class ResidualHiddenLayer(nn.Module):
    """
    Enhanced hidden layer with residual connections and layer normalization
    for better gradient flow and training stability.
    """
    def __init__(self, input_dimension, drop_rate):
        super(ResidualHiddenLayer, self).__init__()
        
        hidden_dim = input_dimension // 2
        
        # Main transformation layers (matching original HiddenLayer structure)
        self.linear1 = nn.Linear(input_dimension, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dimension)
        
        # Normalization and regularization
        self.layer_norm1 = nn.LayerNorm(input_dimension)
        self.layer_norm2 = nn.LayerNorm(input_dimension)
        self.dropout = nn.Dropout(drop_rate)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, data):
        # Store residual connection
        residual = data
        
        # Apply layer normalization before transformation (Pre-LN)
        data = self.layer_norm1(data)
        
        # First linear transformation
        data = self.linear1(data)
        data = self.activation(data)
        data = self.dropout(data)
        
        # Second linear transformation
        data = self.linear2(data)
        data = self.dropout(data)
        
        # Residual connection
        data = data + residual
        
        # Final layer normalization (Post-LN)
        data = self.layer_norm2(data)
        
        return data


class LDAGM(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        feature_num,
        hiddenLayer_num,
        drop_rate,
        use_aggregate,
    ):
        super(LDAGM, self).__init__()
        self.use_aggregate = use_aggregate
        self.hiddenLayer_num = hiddenLayer_num

        self.EmbeddingLayer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension), nn.Dropout(drop_rate)
        )

        self.HiddenLayers = nn.ModuleList(
            [HiddenLayer(hidden_dimension, drop_rate) for _ in range(hiddenLayer_num)]
        )
        self.AggregateLayers = nn.ModuleList(
            [AggregateLayer(hidden_dimension) for _ in range(hiddenLayer_num)]
        )

        # predictive layer
        self.Predict = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dimension * feature_num, 1),
        )

    def forward(self, data):
        data = self.EmbeddingLayer(data)
        if self.use_aggregate:
            agg_data = data
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
                agg_data = self.AggregateLayers[i](data, agg_data)
            predict = self.Predict(agg_data).squeeze(-1)
        else:
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
            predict = self.Predict(data).squeeze(-1)
        return predict


class EnhancedLDAGM(nn.Module):
    """
    Enhanced LDAGM with residual connections, attention-based aggregation,
    and improved architecture for better link prediction performance.
    """
    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        feature_num,
        hiddenLayer_num,
        drop_rate,
        use_aggregate,
        use_residual=True,
        use_attention_aggregate=True,
        num_attention_heads=4,
    ):
        super(EnhancedLDAGM, self).__init__()
        self.use_aggregate = use_aggregate
        self.use_residual = use_residual
        self.use_attention_aggregate = use_attention_aggregate
        self.hiddenLayer_num = hiddenLayer_num

        # Enhanced embedding layer with layer normalization
        self.EmbeddingLayer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

        # Choose between residual and standard hidden layers
        if use_residual:
            self.HiddenLayers = nn.ModuleList(
                [ResidualHiddenLayer(hidden_dimension, drop_rate) for _ in range(hiddenLayer_num)]
            )
        else:
            self.HiddenLayers = nn.ModuleList(
                [HiddenLayer(hidden_dimension, drop_rate) for _ in range(hiddenLayer_num)]
            )
        
        # Choose between attention-based and standard aggregation
        if use_attention_aggregate:
            self.AggregateLayers = nn.ModuleList(
                [EnhancedAttentionAggregateLayer(hidden_dimension, num_attention_heads, drop_rate) 
                 for _ in range(hiddenLayer_num)]
            )
        else:
            self.AggregateLayers = nn.ModuleList(
                [AggregateLayer(hidden_dimension) for _ in range(hiddenLayer_num)]
            )

        # Enhanced predictive layer with additional regularization
        self.Predict = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dimension * feature_num),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dimension * feature_num, hidden_dimension // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(hidden_dimension // 2, 1),
        )

    def forward(self, data):
        # Handle dimension issues for attention mechanism
        if len(data.shape) == 2:
            data = data.unsqueeze(1)  # Add sequence dimension for attention
        
        data = self.EmbeddingLayer(data)
        
        if self.use_aggregate:
            agg_data = data
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
                
                # Handle attention aggregation dimension requirements
                if self.use_attention_aggregate and len(data.shape) == 2:
                    data = data.unsqueeze(1)
                    agg_data = agg_data.unsqueeze(1) if len(agg_data.shape) == 2 else agg_data
                
                agg_data = self.AggregateLayers[i](data, agg_data)
                
                # Squeeze back if needed
                if self.use_attention_aggregate and agg_data.shape[1] == 1:
                    agg_data = agg_data.squeeze(1)
                    
            predict = self.Predict(agg_data).squeeze(-1)
        else:
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
            
            # Squeeze sequence dimension if added
            if data.shape[1] == 1:
                data = data.squeeze(1)
                
            predict = self.Predict(data).squeeze(-1)
            
        return predict
