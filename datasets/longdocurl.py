import os
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from pre_processer import PDFPreprocessor  # Import PDF preprocessor class

class LongDocURLDataset:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        # Configure multimodal data directories
        self.text_save_dir = config.text_save_dir
        self.table_save_dir = config.table_save_dir
        self.img_save_dir = config.img_save_dir
        
        # Initialize PDF preprocessor
        self.preprocessor = PDFPreprocessor(
            llm_api_base=config.llm_api_base,
            llm_api_key=config.llm_api_key,
            llm_model=config.llm_model
        )
        
    def load_raw_data(self, split="train"):
        """Load raw LongDocURL dataset"""
        if not os.path.exists(os.path.join(self.data_dir, f"{split}.csv")):
            # Download from HuggingFace Hub
            file_path = hf_hub_download(
                repo_id="dengc2023/LongDocURL",
                filename=f"{split}.csv",
                cache_dir=self.cache_dir
            )
        else:
            file_path = os.path.join(self.data_dir, f"{split}.csv")
            
        return pd.read_csv(file_path)
    
    def preprocess(self, df):
        """Data preprocessing: integrate PDF preprocessing results"""
        # Filter invalid samples
        df = df.dropna(subset=["document", "url", "label"])
        # Reset index
        df = df.reset_index(drop=True)
        # Add chunk ID and multimodal data paths
        df["chunk_id"] = df.index.map(lambda x: f"chunk_{x}")
        
        # Integrate PDF preprocessing results
        df["text_chunks"] = df.apply(self._get_text_chunks, axis=1)
        df["tables"] = df.apply(self._get_tables, axis=1)
        df["images"] = df.apply(self._get_images, axis=1)
        
        return df
    
    def _get_text_chunks(self, row):
        """Retrieve text chunk data"""
        doc_id = row["chunk_id"]
        zip_id = doc_id[:4] if len(doc_id) >= 4 else doc_id
        text_path = os.path.join(self.text_save_dir, zip_id, f"{doc_id}_text.json")
        
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _get_tables(self, row):
        """Retrieve table data"""
        doc_id = row["chunk_id"]
        zip_id = doc_id[:4] if len(doc_id) >= 4 else doc_id
        table_path = os.path.join(self.table_save_dir, zip_id, f"{doc_id}_tables.json")
        
        if os.path.exists(table_path):
            with open(table_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _get_images(self, row):
        """Retrieve image paths"""
        doc_id = row["chunk_id"]
        zip_id = doc_id[:4] if len(doc_id) >= 4 else doc_id
        img_dir = os.path.join(self.img_save_dir, zip_id)
        
        if os.path.exists(img_dir):
            return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.startswith(doc_id)]
        return []
    
    def process_pdfs(self):
        """Batch process PDF files"""
        # Ensure output directories exist
        for dir_path in [self.text_save_dir, self.table_save_dir, self.img_save_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Process all PDF files
        self.preprocessor.process_batch(
            pdf_dir=os.path.join(self.data_dir, "pdfs"),
            img_save_dir=self.img_save_dir,
            text_save_dir=self.text_save_dir,
            table_save_dir=self.table_save_dir,
            table_img_save_dir=os.path.join(self.data_dir, "table_images")
        )
    
    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset format"""
        splits = ["train", "validation", "test"]
        dataset_dict = {}
        
        for split in splits:
            df = self.load_raw_data(split)
            df = self.preprocess(df)
            dataset_dict[split] = Dataset.from_pandas(df)
            
        return DatasetDict(dataset_dict)