import torch
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from web3 import Web3

class TransactionAnalyzer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def preprocess_transaction(self, transaction_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert transaction data to graph format"""
        # Create graph from transaction data
        G = nx.DiGraph()
        
        # Add nodes and edges based on transaction flow
        for tx in transaction_data['transactions']:
            G.add_node(tx['from'], type='address')
            G.add_node(tx['to'], type='address')
            G.add_edge(tx['from'], tx['to'], 
                      weight=float(tx['value']),
                      timestamp=tx['timestamp'])
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        x = torch.tensor([self._get_node_features(node, G) for node in G.nodes()])
        
        return x, edge_index
    
    def _get_node_features(self, node: str, G: nx.DiGraph) -> np.ndarray:
        """Extract features for a node"""
        features = []
        
        # Basic node features
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        total_value_in = sum(d['weight'] for _, _, d in G.in_edges(node, data=True))
        total_value_out = sum(d['weight'] for _, _, d in G.out_edges(node, data=True))
        
        features.extend([in_degree, out_degree, total_value_in, total_value_out])
        
        return np.array(features, dtype=np.float32)
    
    def analyze_transaction(self, transaction_data: Dict) -> Dict:
        """Analyze a transaction for potential threats"""
        x, edge_index = self.preprocess_transaction(transaction_data)
        
        # Move data to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(x, edge_index)
            probabilities = torch.softmax(output, dim=1)
            
        # Interpret results
        threat_level = probabilities[0].cpu().numpy()
        
        return {
            'threat_level': float(threat_level[1]),
            'confidence': float(max(probabilities[0])),
            'analysis': self._interpret_results(threat_level)
        }
    
    def _interpret_results(self, threat_level: np.ndarray) -> str:
        """Interpret the model's output"""
        if threat_level[1] > 0.8:
            return "High risk transaction detected"
        elif threat_level[1] > 0.5:
            return "Suspicious transaction patterns detected"
        else:
            return "Transaction appears normal" 