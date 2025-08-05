import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor, SentenceTransformer

class MultimodalEncoder:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.sentence_bert = SentenceTransformer(config.sentence_bert_model)
        
        # Move models to the appropriate device
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using Sentence-BERT"""
        return self.sentence_bert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    
    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        """Encode image using CLIP"""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs.cpu().numpy().squeeze()
    
    def encode_table(self, table_data) -> np.ndarray:
        """Encode table data by converting it to text and using Sentence-BERT"""
        # Convert table data to a textual representation
        table_text = "\n".join([",".join(map(str, row)) for row in table_data])
        return self.encode_text(table_text)
    
    def encode_document(self, document: dict) -> dict:
        """
        Encode a multimodal document and store results in an index.
        Args:
            document (dict): A dictionary containing multimodal data with keys:
                - "text": Text content of the document.
                - "images": List of images in the document.
                - "ocr_texts": List of OCR-extracted texts from images.
                - "tables": List of table data (each table is a list of rows).
        Returns:
            dict: A dictionary containing encoded embeddings with keys:
                - "text_emb": Encoded text embedding.
                - "image_embs": List of encoded image embeddings.
                - "ocr_text_embs": List of encoded OCR text embeddings.
                - "table_embs": List of encoded table embeddings.
        """
        index = {}
        
        # Encode text
        if "text" in document and document["text"]:
            index["text_emb"] = self.encode_text(document["text"])
        
        # Encode images
        if "images" in document and document["images"]:
            index["image_embs"] = [self.encode_image(image) for image in document["images"]]
        
        # Encode OCR texts
        if "ocr_texts" in document and document["ocr_texts"]:
            index["ocr_text_embs"] = [self.encode_text(ocr_text) for ocr_text in document["ocr_texts"]]
        
        # Encode tables
        if "tables" in document and document["tables"]:
            index["table_embs"] = [self.encode_table(table) for table in document["tables"]]
        
        return index
    
    def fuse_features(self, text_emb: np.ndarray, image_emb: np.ndarray, table_emb: np.ndarray = None, alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
        """
        Fuse text, image, and optionally table features.
        Args:
            text_emb (np.ndarray): Text embedding.
            image_emb (np.ndarray): Image embedding.
            table_emb (np.ndarray, optional): Table embedding. Defaults to None.
            alpha (float): Weight for text features. Defaults to 0.5.
            beta (float): Weight for table features. Defaults to 0.3.
        Returns:
            np.ndarray: Fused multimodal embedding.
        """
        # Normalize embeddings
        text_emb = text_emb / np.linalg.norm(text_emb)
        image_emb = image_emb / np.linalg.norm(image_emb)
        
        if table_emb is not None:
            table_emb = table_emb / np.linalg.norm(table_emb)
            # Weighted fusion of text, image, and table features
            fused = alpha * text_emb + (1 - alpha - beta) * image_emb + beta * table_emb
        else:
            # Weighted fusion of text and image features
            fused = alpha * text_emb + (1 - alpha) * image_emb
        
        return fused / np.linalg.norm(fused)
    
    def align_features(self, text_emb: np.ndarray, image_emb: np.ndarray, table_emb: np.ndarray = None) -> np.ndarray:
        """
        Perform multimodal alignment by concatenating normalized embeddings.
        Args:
            text_emb (np.ndarray): Text embedding.
            image_emb (np.ndarray): Image embedding.
            table_emb (np.ndarray, optional): Table embedding. Defaults to None.
        Returns:
            np.ndarray: Aligned multimodal embedding.
        """
        # Normalize embeddings
        text_emb = text_emb / np.linalg.norm(text_emb)
        image_emb = image_emb / np.linalg.norm(image_emb)
        
        if table_emb is not None:
            table_emb = table_emb / np.linalg.norm(table_emb)
            # Concatenate text, image, and table embeddings
            aligned = np.concatenate([text_emb, image_emb, table_emb])
        else:
            # Concatenate text and image embeddings
            aligned = np.concatenate([text_emb, image_emb])
        
        return aligned