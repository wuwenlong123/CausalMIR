from config import Config
from utils.batch_processor import BatchProcessor
from datasets.pre_processer import PDFPreprocessor
from models.causal_scm import CausalSCMBuilder
from utils.metrics import CausalMetrics  # Import evaluation metrics
import logging
import os

def main():
    config = Config()
    logger = logging.getLogger("batch_processor")
    logger.addHandler(logging.FileHandler(f"{config.logs_dir}/batch.log"))
    logger.setLevel(logging.INFO)
    
    # Verify required directories exist
    for dir_path in [config.data_dir, config.output_dir, config.logs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    processor = BatchProcessor(config)
    docs = [f"{config.data_dir}/{f}" for f in os.listdir(config.data_dir) if f.endswith(".pdf")]
    
    for doc in docs:
        try:
            logger.info(f"Processing document: {doc}")
            preprocessor = PDFPreprocessor(config)
            raw_chunks = preprocessor.parse_pdf(doc)
            processed_chunks = preprocessor.preprocess_chunks(raw_chunks)
            
            scm_builder = CausalSCMBuilder(config, preprocessor)
            candidate_pairs = scm_builder.causal_initial_screening(processed_chunks)
            scm = scm_builder.build_initial_scm(candidate_pairs, processed_chunks)
            
            # Evaluation
            logger.info("Evaluating causal model...")
            true_edges = scm_builder.get_ground_truth_edges()  # Replace with actual ground truth retrieval
            pred_edges = scm["edges"]
            
            acc_overall = CausalMetrics.overall_accuracy(pred_edges, true_edges)
            acc_dir = CausalMetrics.intervention_accuracy(pred_edges, true_edges)
            mir = CausalMetrics.hallucination_rate(
                [chunk["content"] for chunk in processed_chunks if chunk["type"] == "text"],
                [chunk["ground_truth"] for chunk in processed_chunks if chunk["type"] == "text"]
            )
            
            logger.info(f"Evaluation Results for {doc}:")
            logger.info(f"  Acc-overall: {acc_overall:.2f}")
            logger.info(f"  Acc-dir: {acc_dir:.2f}")
            logger.info(f"  MIR: {mir:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to process document {doc}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()