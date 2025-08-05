import os
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from pre_processer import PDFPreprocessor  # Import PDF preprocessor class

class MMLongBenchDataset:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        # Multimodal data storage directories
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
        """Load the raw MMLongBench dataset (supports CSV/JSON formats)"""
        # Try different file name formats
        possible_paths = [
            os.path.join(self.data_dir, f"{self.config.task}_{split}.json"),
            os.path.join(self.data_dir, f"{split}.csv"),
            os.path.join(self.data_dir, f"{self.config.task}_{split}.csv")
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        # Download from HuggingFace Hub if not found locally
        if file_path is None:
            repo_id = self.config.repo_id if hasattr(self.config, 'repo_id') else "mayubo2333/MMLongBench-Doc"
            try:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{self.config.task}_{split}.json",
                    cache_dir=self.cache_dir
                )
            except:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{split}.csv",
                    cache_dir=self.cache_dir
                )
        
        # Load data based on file format
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:  # CSV format
            return pd.read_csv(file_path)
    
    def preprocess(self, df):
        """Data preprocessing: integrate multimodal data"""
        # Filter invalid samples (retain necessary fields based on task type)
        required_cols = ["context", "target"] if self.config.task_type == "summarization" else \
                        ["context", "question", "target"] if self.config.task_type == "qa" else \
                        ["context", "label"]
        df = df.dropna(subset=required_cols)
        df = df.reset_index(drop=True)
        
        # Add unique identifier
        df["sample_id"] = df.index.map(lambda x: f"{self.config.task}_{x}")
        
        # Integrate multimodal data
        df["text_chunks"] = df.apply(self._get_text_chunks, axis=1)
        df["tables"] = df.apply(self._get_tables, axis=1)
        df["images"] = df.apply(self._get_images, axis=1)
        
        # Process task-specific fields
        if self.config.task_type == "qa":
            df["question_text"] = df["question"].apply(lambda x: x if isinstance(x, str) else " ".join(x))
        elif self.config.task_type == "classification":
            df["label_id"] = df["label"].astype('category').cat.codes
        
        return df
    
    def _get_text_chunks(self, row):
        """Retrieve text chunk data"""
        sample_id = row["sample_id"]
        # Group by ID prefix to reduce directory pressure
        group_id = sample_id[:6] if len(sample_id) >= 6 else sample_id
        text_path = os.path.join(self.text_save_dir, group_id, f"{sample_id}_text.json")
        
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Use raw text if no preprocessing results are available
        return [row["context"]] if "context" in row else []
    
    def _get_tables(self, row):
        """Retrieve table data"""
        sample_id = row["sample_id"]
        group_id = sample_id[:6] if len(sample_id) >= 6 else sample_id
        table_path = os.path.join(self.table_save_dir, group_id, f"{sample_id}_tables.json")
        
        if os.path.exists(table_path):
            with open(table_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _get_images(self, row):
        """Retrieve image path list"""
        sample_id = row["sample_id"]
        group_id = sample_id[:6] if len(sample_id) >= 6 else sample_id
        img_dir = os.path.join(self.img_save_dir, group_id)
        
        if os.path.exists(img_dir):
            # Filter images for the current sample
            return [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                    if f.startswith(sample_id) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Use raw image paths if available
        return row.get("images", []) if "images" in row else []
    
    def process_documents(self):
        """Batch process documents (PDF/images, etc.) to generate multimodal data"""
        # Ensure output directories exist
        for dir_path in [self.text_save_dir, self.table_save_dir, self.img_save_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Process all documents
        self.preprocessor.process_batch(
            doc_dir=os.path.join(self.data_dir, "documents"),  # Supports various document formats
            img_save_dir=self.img_save_dir,
            text_save_dir=self.text_save_dir,
            table_save_dir=self.table_save_dir,
            table_img_save_dir=os.path.join(self.data_dir, "table_images")
        )
    
    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset format"""
        splits = ["train", "validation", "test"] if hasattr(self.config, 'splits') else self.config.splits
        dataset_dict = {}
        
        for split in splits:
            df = self.load_raw_data(split)
            df = self.preprocess(df)
            dataset_dict[split]