import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from transformers import BertModel, BertConfig

class GNNBERT(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=256, num_heads=4):
        super(GNNBERT, self).__init__()
        
        # BERT configuration
        self.bert_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4
        )
        self.bert = BertModel(self.bert_config)
        
        # GNN layers
        self.gnn1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.gnn2 = GATConv(hidden_dim * num_heads, hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, attention_mask=None):
        # GNN processing
        gnn_out = self.gnn1(x, edge_index)
        gnn_out = F.relu(gnn_out)
        gnn_out = self.gnn2(gnn_out, edge_index)
        
        # BERT processing
        bert_out = self.bert(inputs_embeds=x, attention_mask=attention_mask)[0]
        
        # Feature fusion
        combined = torch.cat([gnn_out, bert_out], dim=-1)
        output = self.fusion(combined)
        
        return output

class TransactionClassifier(nn.Module):
    def __init__(self, gnn_bert_model, num_classes=2):
        super(TransactionClassifier, self).__init__()
        self.gnn_bert = gnn_bert_model
        self.classifier = nn.Linear(self.gnn_bert.bert_config.hidden_size, num_classes)
        
    def forward(self, x, edge_index, attention_mask=None):
        features = self.gnn_bert(x, edge_index, attention_mask)
        return self.classifier(features)

class AccountAnalyzer(nn.Module):
    def __init__(self, gnn_bert_model, num_classes=3):
        super(AccountAnalyzer, self).__init__()
        self.gnn_bert = gnn_bert_model
        self.analyzer = nn.Sequential(
            nn.Linear(self.gnn_bert.bert_config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, edge_index, attention_mask=None):
        features = self.gnn_bert(x, edge_index, attention_mask)
        return self.analyzer(features) 