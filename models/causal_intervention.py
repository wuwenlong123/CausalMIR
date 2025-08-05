import networkx as nx
import numpy as np
from typing import List, Dict, Tuple

class CausalIntervention:
    def __init__(self, config):
        self.config = config
        
    def do_calculus(self, causal_graph, target_node, interventions: Dict) -> Dict:
        """
        Perform causal intervention (Do-Calculus).
        
        Args:
            causal_graph: The causal graph (NetworkX DiGraph).
            target_node: The target node for the intervention.
            interventions: A dictionary of interventions {'node_id': value}.
            
        Returns:
            dict: The result of the intervention, including:
                - "target_node": The target node.
                - "interventions": The applied interventions.
                - "effect": The estimated effect of the interventions.
                - "paths": The causal paths from intervention nodes to the target node.
        """
        # 1. Apply interventions to the graph
        modified_graph = self._apply_interventions(causal_graph, interventions)
        
        # 2. Identify causal paths
        causal_paths = self._find_causal_paths(modified_graph, interventions.keys(), target_node)
        
        # 3. Estimate the effect of the interventions
        effect = self._estimate_effect(modified_graph, causal_paths, interventions)
        
        return {
            "target_node": target_node,
            "interventions": interventions,
            "effect": effect,
            "paths": causal_paths
        }
    
    def _apply_interventions(self, graph, interventions):
        """
        Apply interventions to the causal graph.
        
        Args:
            graph: The original causal graph.
            interventions: A dictionary of interventions {'node_id': value}.
            
        Returns:
            NetworkX DiGraph: The modified graph after applying interventions.
        """
        modified_graph = graph.copy()
        
        for node, value in interventions.items():
            if node not in modified_graph.nodes:
                continue
            
            # Remove all incoming edges to the intervened node
            in_edges = list(modified_graph.in_edges(node))
            modified_graph.remove_edges_from(in_edges)
            
            # Update the node with the intervention value
            modified_graph.nodes[node]["intervened"] = True
            modified_graph.nodes[node]["value"] = value
            
        return modified_graph
    
    def _find_causal_paths(self, graph, sources, target) -> List[List[str]]:
        """
        Find all causal paths from source nodes to the target node.
        
        Args:
            graph: The causal graph.
            sources: A list of source nodes.
            target: The target node.
            
        Returns:
            List[List[str]]: A list of causal paths (each path is a list of node IDs).
        """
        paths = []
        for source in sources:
            if source not in graph.nodes:
                continue
                
            for path in nx.all_simple_paths(graph, source=source, target=target):
                paths.append(path)
                
        return paths
    
    def _estimate_effect(self, graph, paths, interventions) -> float:
        """
        Estimate the effect of interventions on the target node.
        
        Args:
            graph: The modified causal graph.
            paths: The causal paths from intervention nodes to the target node.
            interventions: A dictionary of interventions {'node_id': value}.
            
        Returns:
            float: The estimated effect of the interventions.
        """
        total_strength = 0.0
        path_count = 0
        
        for path in paths:
            path_strength = 1.0
            
            # Compute the strength of the path (product of edge weights)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = graph.get_edge_data(u, v)
                if not edge_data:
                    path_strength = 0
                    break
                path_strength *= edge_data["weight"]
                
            if path_strength > 0:
                total_strength += path_strength
                path_count += 1
                
        return total_strength / (path_count + 1e-8)  # Avoid division by zero
    
    def filter_causal_graph(self, graph, threshold: float) -> nx.DiGraph:
        """
        Filter the causal graph to retain only significant causal relationships.
        
        Args:
            graph: The original causal graph.
            threshold: The minimum edge weight to retain.
            
        Returns:
            NetworkX DiGraph: The filtered causal graph.
        """
        filtered_graph = nx.DiGraph()
        
        for u, v, data in graph.edges(data=True):
            if data.get("weight", 0) >= threshold:
                filtered_graph.add_edge(u, v, **data)
        
        for node, data in graph.nodes(data=True):
            filtered_graph.add_node(node, **data)
        
        return filtered_graph
