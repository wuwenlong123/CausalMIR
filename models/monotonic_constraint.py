from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from .causal_scm import CausalAnalyzer  # 导入对应文件中的CausalAnalyzer类

class MonotonicConstraintGenerator:
    
    def __init__(self, config, causal_analyzer):
        self.config = config
        self.causal_analyzer = causal_analyzer  # Depends on causal analysis results
        self.feature_extractor = causal_analyzer.feature_extractor  # Used to extract chunk text content
    
    def generate_constraints(self, processed_chunks):
        """
        Generate monotonicity constraints (core method).
        
        Args:
            processed_chunks (list): Preprocessed document chunks.
        
        Returns:
            list: A list of natural language constraint descriptions.
        """
        # 1. Extract causal chains (from the causal analyzer)
        if not self.causal_analyzer.causal_graph:
            return ["No causal relationships detected, no additional constraints required."]
        
        # 2. Filter strong causal chains related to the current document (strength > threshold)
        strong_chains = self._extract_strong_causal_chains()
        
        # 3. Generate monotonicity constraints based on causal chains
        constraints = self._generate_trend_constraints(strong_chains)
        
        # 4. Add general constraints (e.g., causal direction consistency)
        constraints.append("When A directly causes B, an increase in A should accompany an increase in B, and a decrease in A should accompany a decrease in B.")
        
        return constraints
    
    def _extract_strong_causal_chains(self):
        """
        Extract causal chains with strength exceeding the threshold.
        
        Returns:
            list: A list of strong causal chains and their strengths.
        """
        strong_threshold = self.config.strong_threshold  # Get strong causal threshold from config
        causal_graph = self.causal_analyzer.causal_graph
        
        # Traverse all possible causal paths (simplified example, can be extended to multi-hop paths)
        chains = []
        for u, v, data in causal_graph.edges(data=True):
            if data["weight"] >= strong_threshold:
                chains.append(([u, v], data["weight"]))  # Simple causal chain (u → v)
        
        return chains
    
    def _generate_trend_constraints(self, causal_chains):
        """
        Generate constraints based on trends in causal chains.
        
        Args:
            causal_chains (list): List of causal chains and their strengths.
        
        Returns:
            list: A list of monotonicity constraints in natural language.
        """
        constraints = []
        for chain, strength in causal_chains:
            # Get the text content of the start and end chunks in the causal chain
            start_chunk = next(c for c in self.causal_analyzer.processed_chunks if c["id"] == chain[0])
            end_chunk = next(c for c in self.causal_analyzer.processed_chunks if c["id"] == chain[1])
            
            # Determine trends (using the causal analyzer's _get_trend method)
            start_trend = self.causal_analyzer._get_trend(self.feature_extractor._get_text_content(start_chunk))
            end_trend = self.causal_analyzer._get_trend(self.feature_extractor._get_text_content(end_chunk))
            
            # Generate trend consistency constraints
            if start_trend == 1 and end_trend == 1:
                constraints.append(f"In the causal chain {chain[0]} → {chain[1]}, an increase in the former should accompany an increase in the latter.")
            elif start_trend == -1 and end_trend == -1:
                constraints.append(f"In the causal chain {chain[0]} → {chain[1]}, a decrease in the former should accompany a decrease in the latter.")
            elif start_trend != 0 and end_trend != 0:
                constraints.append(f"In the causal chain {chain[0]} → {chain[1]}, the direction of change in the former and the latter should be consistent.")
        
        return constraints