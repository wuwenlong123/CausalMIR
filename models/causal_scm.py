import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CausalSCMBuilder:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder
        self.scm = None
        
    def build_from_chunks(self, processed_chunks):
        """
        Build the SCM (Structural Causal Model) from document chunks.
        Args:
            processed_chunks (list): List of processed document chunks, each containing:
                - "id": Unique identifier for the chunk.
                - "type": Type of the chunk ("text", "image", "table").
                - "page": Page number of the chunk.
                - "content": Content of the chunk (text, image, or table data).
        Returns:
            dict: The constructed SCM containing nodes and edges.
        """
        # 1. Build nodes
        nodes = []
        node_embeddings = {}
        
        for chunk in processed_chunks:
            # Generate node embeddings based on chunk type
            if chunk["type"] == "text":
                emb = self.encoder.encode_text(chunk["content"])
            elif chunk["type"] == "image":
                emb = self.encoder.encode_image(chunk["content"])
            elif chunk["type"] == "table":
                emb = self.encoder.encode_table(chunk["content"])
            else:
                continue
                
            node_embeddings[chunk["id"]] = emb
            nodes.append({
                "id": chunk["id"],
                "type": chunk["type"],
                "page": chunk["page"],
                "embedding": emb
            })
        
        # 2. Build edges (causal relationships)
        edges = self._detect_causal_relations(processed_chunks, node_embeddings)
        
        # 3. Construct the SCM
        self.scm = {
            "nodes": nodes,
            "edges": edges
        }
        
        return self.scm
    
    def _detect_causal_relations(self, chunks, embeddings):
        """
        Detect causal relationships between chunks based on embeddings.
        Args:
            chunks (list): List of document chunks.
            embeddings (dict): Dictionary of chunk embeddings keyed by chunk ID.
        Returns:
            list: List of causal edges with source, target, and additional features.
        """
        edges = []
        chunk_dict = {c["id"]: c for c in chunks}
        chunk_ids = list(chunk_dict.keys())
        
        # Compute similarity matrix
        emb_list = np.array([embeddings[cid] for cid in chunk_ids])
        sim_matrix = cosine_similarity(emb_list)
        
        # Build causal edges
        for i, source_id in enumerate(chunk_ids):
            for j, target_id in enumerate(chunk_ids):
                if i == j:
                    continue  # Skip self-loops
                    
                source_chunk = chunk_dict[source_id]
                target_chunk = chunk_dict[target_id]
                
                # Filter chunks based on page distance
                if abs(source_chunk["page"] - target_chunk["page"]) > self.config.max_page_distance:
                    continue
                    
                # Compute causal strength
                similarity = sim_matrix[i][j]
                if similarity < self.config.causal_threshold:
                    continue  # Skip weak causal relationships
                    
                # Add causal edge
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "strength": similarity,
                    "features": {
                        "page_distance": abs(source_chunk["page"] - target_chunk["page"]),
                        "source_type": source_chunk["type"],
                        "target_type": target_chunk["type"]
                    }
                })
        
        return edges
    
    def to_graph(self):
        """
        Convert the SCM to a NetworkX directed graph.
        Returns:
            nx.DiGraph: The constructed directed graph.
        """
        if not self.scm:
            raise ValueError("SCM model has not been built yet.")
            
        G = nx.DiGraph()
        for node in self.scm["nodes"]:
            G.add_node(node["id"], type=node["type"], page=node["page"])
            
        for edge in self.scm["edges"]:
            G.add_edge(
                edge["source"], 
                edge["target"], 
                weight=edge["strength"],
                features=edge["features"]
            )
            
        return G