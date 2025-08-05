import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    def __init__(self, config, processor_func, max_workers=4):
        self.config = config
        self.processor_func = processor_func  # Function to process a single file
        self.max_workers = max_workers
        self.logger = logging.getLogger("batch_processor")
        
    def process_directory(self, input_dir: str):
        """Process all files in the directory"""
        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return []
            
        # Get all PDF files
        file_paths = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith(".pdf")
        ]
        
        if not file_paths:
            self.logger.warning("No PDF files found")
            return []
            
        # Batch processing
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_single_file, fp): fp for fp in file_paths}
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"File processed successfully: {file_path}")
                except Exception as e:
                    self.logger.error(f"File processing failed {file_path}: {str(e)}")
                    
        return results
    
    def _process_single_file(self, file_path: str):
        """Process a single file"""
        start_time = time.time()
        result = self.processor_func(file_path)
        elapsed = time.time() - start_time
        return {
            "file_path": file_path,
            "result": result,
            "elapsed_time": elapsed
        }
