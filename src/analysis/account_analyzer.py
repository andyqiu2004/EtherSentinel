import torch
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from web3 import Web3
from datetime import datetime, timedelta

class AccountAnalyzer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def analyze_account(self, account_data: Dict) -> Dict:
        """Analyze an account's behavior and risk level"""
        # Extract account features
        features = self._extract_account_features(account_data)
        
        # Create graph representation
        G = self._create_account_graph(account_data)
        
        # Convert to model input format
        x, edge_index = self._prepare_model_input(G, features)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(x, edge_index)
            probabilities = torch.softmax(output, dim=1)
            
        return self._generate_analysis_report(probabilities, account_data)
    
    def _extract_account_features(self, account_data: Dict) -> Dict:
        """Extract relevant features from account data"""
        features = {
            'total_transactions': len(account_data['transactions']),
            'total_value_sent': sum(float(tx['value']) for tx in account_data['transactions'] if tx['from'] == account_data['address']),
            'total_value_received': sum(float(tx['value']) for tx in account_data['transactions'] if tx['to'] == account_data['address']),
            'unique_contracts_interacted': len(set(tx['to'] for tx in account_data['transactions'] if tx['to'].startswith('0x'))),
            'account_age_days': (datetime.now() - datetime.fromtimestamp(account_data['first_seen'])).days
        }
        return features
    
    def _create_account_graph(self, account_data: Dict) -> nx.DiGraph:
        """Create a graph representation of account interactions"""
        G = nx.DiGraph()
        
        # Add main account node
        G.add_node(account_data['address'], type='main_account')
        
        # Add interaction nodes and edges
        for tx in account_data['transactions']:
            if tx['from'] == account_data['address']:
                G.add_node(tx['to'], type='interaction')
                G.add_edge(account_data['address'], tx['to'], 
                          weight=float(tx['value']),
                          timestamp=tx['timestamp'])
            elif tx['to'] == account_data['address']:
                G.add_node(tx['from'], type='interaction')
                G.add_edge(tx['from'], account_data['address'],
                          weight=float(tx['value']),
                          timestamp=tx['timestamp'])
        
        return G
    
    def _prepare_model_input(self, G: nx.DiGraph, features: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare graph data for model input"""
        # Convert graph to PyTorch Geometric format
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        
        # Create node features
        node_features = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'main_account':
                node_features.append(list(features.values()))
            else:
                # Create features for interaction nodes
                node_features.append([
                    G.in_degree(node),
                    G.out_degree(node),
                    sum(d['weight'] for _, _, d in G.in_edges(node, data=True)),
                    sum(d['weight'] for _, _, d in G.out_edges(node, data=True))
                ])
        
        x = torch.tensor(node_features, dtype=torch.float32)
        return x, edge_index
    
    def _generate_analysis_report(self, probabilities: torch.Tensor, account_data: Dict) -> Dict:
        """Generate a comprehensive analysis report"""
        risk_level = probabilities[0].cpu().numpy()
        
        return {
            'risk_score': float(risk_level[2]),  # Assuming 3 classes: low, medium, high risk
            'confidence': float(max(probabilities[0])),
            'risk_level': self._get_risk_level(risk_level),
            'recommendations': self._generate_recommendations(risk_level, account_data)
        }
    
    def _get_risk_level(self, risk_level: np.ndarray) -> str:
        """Convert risk probabilities to risk level"""
        if risk_level[2] > 0.7:
            return "High Risk"
        elif risk_level[2] > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _generate_recommendations(self, risk_level: np.ndarray, account_data: Dict) -> List[str]:
        """Generate recommendations based on risk level and account data"""
        recommendations = []
        
        if risk_level[2] > 0.7:
            recommendations.extend([
                "Immediate investigation recommended",
                "Consider freezing account activity",
                "Review all recent transactions"
            ])
        elif risk_level[2] > 0.4:
            recommendations.extend([
                "Monitor account activity closely",
                "Review transaction patterns",
                "Consider implementing additional security measures"
            ])
        else:
            recommendations.extend([
                "Regular monitoring recommended",
                "Maintain current security measures"
            ])
            
        return recommendations 