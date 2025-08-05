import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score

class CausalMetrics:
    @staticmethod
    def causal_precision(pred_edges, true_edges) -> float:
        """Causal relationship precision"""
        pred_set = set((e["source"], e["target"]) for e in pred_edges)
        true_set = set((e["source"], e["target"]) for e in true_edges)
        
        if not pred_set:
            return 0.0
            
        return len(pred_set & true_set) / len(pred_set)
    
    @staticmethod
    def causal_recall(pred_edges, true_edges) -> float:
        """Causal relationship recall"""
        pred_set = set((e["source"], e["target"]) for e in pred_edges)
        true_set = set((e["source"], e["target"]) for e in true_edges)
        
        if not true_set:
            return 0.0
            
        return len(pred_set & true_set) / len(true_set)
    
    @staticmethod
    def causal_f1(pred_edges, true_edges) -> float:
        """Causal relationship F1 score"""
        precision = CausalMetrics.causal_precision(pred_edges, true_edges)
        recall = CausalMetrics.causal_recall(pred_edges, true_edges)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def graph_edit_distance(pred_graph, true_graph) -> float:
        """Graph edit distance (simplified version)"""
        # Compute node differences
        pred_nodes = set(pred_graph.nodes())
        true_nodes = set(true_graph.nodes())
        node_diff = len(pred_nodes.symmetric_difference(true_nodes))
        
        # Compute edge differences
        pred_edges = set(pred_graph.edges())
        true_edges = set(true_graph.edges())
        edge_diff = len(pred_edges.symmetric_difference(true_edges))
        
        return node_diff + edge_diff
    
    @staticmethod
    def intervention_accuracy(pred_effects, true_effects) -> float:
        """Intervention effect accuracy (Acc-dir)"""
        # Compute effect direction accuracy
        pred_directions = np.sign(list(pred_effects.values()))
        true_directions = np.sign(list(true_effects.values()))
        
        return np.mean(pred_directions == true_directions)
    
    @staticmethod
    def overall_accuracy(pred_effects, true_effects) -> float:
        """Overall intervention accuracy (Acc-overall)"""
        # Compute exact match accuracy
        correct = sum(1 for k in pred_effects if pred_effects[k] == true_effects.get(k, None))
        total = len(true_effects)
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def hallucination_rate(pred_texts, true_texts) -> float:
        """
        Long-text hallucination rate (MIR).
        
        Args:
            pred_texts (list): List of predicted texts.
            true_texts (list): List of ground truth texts.
        
        Returns:
            float: Hallucination rate (lower is better).
        """
        hallucinations = 0
        total = len(true_texts)
        
        for pred, true in zip(pred_texts, true_texts):
            if pred not in true:  # Check if predicted text is hallucinated
                hallucinations += 1
        
        return hallucinations / total if total > 0 else 0.0
