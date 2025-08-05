import os
import json
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from tqdm import tqdm
import argparse
import requests
from typing import List, Dict, Tuple, Optional


class PDFPreprocessor:
    def __init__(self, llm_api_base: str, llm_api_key: str, llm_model: str = "gpt-4o"):
        """
        Initialize PDF preprocessing tool
        
        Args:
            llm_api_base: Base URL for LLM API
            llm_api_key: API key for LLM authentication
            llm_model: Name of the LLM model to use
        """
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model

    def save_png(self, page: fitz.Page, page_num: int, doc_id: str, img_save_dir: str, zip_id: str) -> str:
        """Save PDF page as PNG image"""
        # Create image save directory (organized by zip_id)
        save_dir = f"{img_save_dir}/{zip_id}"
        os.makedirs(save_dir, exist_ok=True)
        img_path = f"{save_dir}/{doc_id}_{page_num}.png"
        
        # Save page as image
        pix = page.get_pixmap()
        pix.save(img_path)
        return img_path

    def resize_coords(self, img_size: Tuple[int, int], bbox: List[float]) -> List[float]:
        """Normalize text coordinates to image size ratio"""
        ori_w, ori_h = img_size
        return [
            round(bbox[0] / ori_w, 3),
            round(bbox[1] / ori_h, 3),
            round(bbox[2] / ori_w, 3),
            round(bbox[3] / ori_h, 3)
        ]

    def detect_and_extract_tables(self, pdf_path: str, page_num: int) -> List[Dict]:
        """
        Detect and extract tables from specified PDF page
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (starting from 0)
            
        Returns:
            List of table information, each containing content, bounding box, etc.
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return tables
                    
                page = pdf.pages[page_num]
                # Find tables
                found_tables = page.find_tables()
                
                for table_idx, table in enumerate(found_tables):
                    # Extract table content
                    table_data = page.extract_table(table_settings=table)
                    
                    # Convert bounding box coordinates (pdfplumber uses bottom-left origin, need to convert to top-left)
                    bbox = [
                        table.bbox[0],  # x0
                        page.height - table.bbox[3],  # y0 (convert origin)
                        table.bbox[2],  # x1
                        page.height - table.bbox[1]   # y1 (convert origin)
                    ]
                    
                    tables.append({
                        "table_id": f"table_{page_num}_{table_idx}",
                        "page_num": page_num,
                        "bbox": bbox,
                        "data": table_data,
                        "rows": len(table_data) if table_data else 0,
                        "cols": len(table_data[0]) if table_data and len(table_data) > 0 else 0
                    })
                    
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            
        return tables

    def call_llm_api(self, prompt: str) -> Optional[str]:
        """Call LLM API to generate response"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a table processing expert skilled at analyzing table content and generating valuable information."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.llm_api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"LLM API request failed: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return None

    def enhance_table_data(self, table_data: List[List[str]]) -> Dict:
        """
        Enhance table data using LLM
        
        Args:
            table_data: Raw table data
            
        Returns:
            Enhanced table information including description, summary, potential issues, etc.
        """
        # Convert table data to string
        table_str = "\n".join(["|".join(row) for row in table_data])
        
        # Generate prompt
        prompt = f"""Please analyze the following table data and provide:
1. A concise table description (1-2 sentences)
2. Key findings or main trends in the table
3. Potential insights or conclusions

Table data:
{table_str}
"""
        
        # Call LLM API
        llm_response = self.call_llm_api(prompt)
        
        if not llm_response:
            return {
                "description": "Failed to generate table description",
                "key_findings": [],
                "insights": []
            }
        
        # Parse LLM response (simplified processing, may need more complex parsing in practice)
        sections = llm_response.split('\n\n')
        
        return {
            "description": sections[0] if len(sections) > 0 else "No description",
            "key_findings": sections[1].split('\n') if len(sections) > 1 else [],
            "insights": sections[2].split('\n') if len(sections) > 2 else [],
            "raw_llm_response": llm_response
        }

    def is_text_in_table(self, text_bbox: List[float], table_bbox: List[float]) -> bool:
        """Determine if text is inside a table"""
        tx0, ty0, tx1, ty1 = text_bbox
        tb_x0, tb_y0, tb_x1, tb_y1 = table_bbox
        
        # Check if text bounding box is inside table bounding box
        return (
            tx0 >= tb_x0 and ty0 >= tb_y0 and
            tx1 <= tb_x1 and ty1 <= tb_y1
        )

    def process_single_pdf(self, pdf_path: str, img_save_dir: str, text_save_dir: str, 
                          table_save_dir: str, table_img_save_dir: str) -> None:
        """
        Process single PDF file: extract text, images, and tables, and perform table enhancement
        
        Args:
            pdf_path: Path to the PDF file
            img_save_dir: Directory to save page images
            text_save_dir: Directory to save text blocks
            table_save_dir: Directory to save table data
            table_img_save_dir: Directory to save table images
        """
        # Parse document ID and zip ID
        doc_id = os.path.basename(pdf_path).split(".")[0]
        zip_id = doc_id[:4] if len(doc_id) >= 4 else doc_id
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open PDF file {pdf_path}, error: {e}")
            return
        
        # Create various save directories
        for dir_path in [
            f"{img_save_dir}/{zip_id}",
            f"{text_save_dir}/{zip_id}",
            f"{table_save_dir}/{zip_id}",
            f"{table_img_save_dir}/{zip_id}"
        ]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Store content information for all pages
        all_texts = []
        all_tables = []
        img_size = None  # Image size (all pages have the same size, get from first page)
        
        for page_num, page in enumerate(doc):
            try:
                # Save current page as image
                img_path = self.save_png(page, page_num, doc_id, img_save_dir, zip_id)
                # Get image size (only need to get once)
                if img_size is None:
                    with Image.open(img_path) as img:
                        img_size = img.size
            except Exception as e:
                print(f"Failed to process page {page_num} image, error: {e}")
                continue
            
            # Extract tables from page
            tables = self.detect_and_extract_tables(pdf_path, page_num)
            
            # Save screenshot for each table
            for table in tables:
                # Table bounding box
                bbox = table["bbox"]
                # Convert to fitz format
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                
                # Capture table area
                pix = page.get_pixmap(clip=rect)
                table_img_path = f"{table_img_save_dir}/{zip_id}/{doc_id}_{table['table_id']}.png"
                pix.save(table_img_path)
                table["image_path"] = table_img_path
                
                # If table has data, perform enhancement
                if table["data"]:
                    enhanced_data = self.enhance_table_data(table["data"])
                    table["enhanced"] = enhanced_data
                
                all_tables.append(table)
            
            # Extract page text (with coordinate information)
            # Format: (x0, y0, x1, y1, text, line_no, block_no, word_no)
            word_list = page.get_text("words")
            page_texts = []
            
            for word_info in word_list:
                # Original coordinates
                bbox = list(word_info[:4])
                # Check if text belongs to any table, skip if it does (to avoid duplication)
                in_table = False
                for table in tables:
                    if self.is_text_in_table(bbox, table["bbox"]):
                        in_table = True
                        break
                
                if in_table:
                    continue
                    
                # Normalized coordinates
                normalized_bbox = self.resize_coords(img_size, bbox)
                page_texts.append({
                    "coordi": normalized_bbox,
                    "word": word_info[4].encode('utf-8', 'ignore').decode('utf-8'),  # Handle special characters
                    "line_no": word_info[5],
                    "block_no": word_info[6],
                    "word_no": word_info[7],
                    "page_no": page_num
                })
            
            all_texts.extend(page_texts)
        
        # Save text information
        if img_size and all_texts:
            text_data = {
                "zip_no": zip_id,
                "doc_no": doc_id,
                "pdf_path": pdf_path,
                "img_size": img_size,
                "contents": all_texts
            }
            text_path = f"{text_save_dir}/{zip_id}/{doc_id}_text.json"
            with open(text_path, "w", encoding="utf-8") as f:
                json.dump(text_data, f, ensure_ascii=False, indent=2)
        
        # Save table information
        if all_tables:
            table_data = {
                "zip_no": zip_id,
                "doc_no": doc_id,
                "pdf_path": pdf_path,
                "tables": all_tables
            }
            table_path = f"{table_save_dir}/{zip_id}/{doc_id}_tables.json"
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(table_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processing completed: {pdf_path}, extracted {len(all_tables)} tables")

    def process_batch(self, pdf_dir: str, img_save_dir: str, text_save_dir: str, 
                     table_save_dir: str, table_img_save_dir: str) -> None:
        """Batch process PDF files"""
        # Get all PDF file paths
        pdf_files = [
            os.path.join(pdf_dir, f) 
            for f in os.listdir(pdf_dir) 
            if f.lower().endswith(".pdf")
        ]
        print(f"Found {len(pdf_files)} PDF files, starting processing...")
        
        # Batch processing
        for pdf_path in tqdm(pdf_files, desc="Processing progress"):
            self.process_single_pdf(pdf_path, img_save_dir, text_save_dir, 
                                  table_save_dir, table_img_save_dir)
        print("Preprocessing completed!")


def main():
    parser = argparse.ArgumentParser(description="PDF preprocessing: extract and chunk text, images, and tables, and enhance table data")

    args = parser.parse_args()
    
    # 初始化预处理工具
    preprocessor = PDFPreprocessor(
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model
    )
    
    # 批量处理
    preprocessor.process_batch(
        pdf_dir=args.pdf_dir,
        img_save_dir=args.img_save_dir,
        text_save_dir=args.text_save_dir,
        table_save_dir=args.table_save_dir,
        table_img_save_dir=args.table_img_save_dir
    )


if __name__ == "__main__":
    main()
