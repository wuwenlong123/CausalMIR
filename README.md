# Multimodal Causal Reasoning Framework
A computational framework for extracting and analyzing causal relationships from multimodal documents, developed for AAAI 2026 submission.

## Overview

This framework processes PDF documents containing text, tables, and images to identify causal relationships and build causal models using structural causal modeling (SCM) techniques. The system implements monotonicity constraints and causal intervention analysis to enhance reasoning capabilities across multiple modalities.

## Installation

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements_extended.txt`

### Setup

```bash
git clone <repository-url>
cd AAAI-2026-CausalMIR
pip install -r requirements_extended.txt
```

## Configuration

Modify parameters in `config.py` to adjust system behavior:

- Directory paths for input/output operations
- Causal analysis thresholds
- Model hyperparameters
- Batch processing settings

## Usage

### Basic Workflow

1. Place PDF documents in the designated data directory.
2. Run the batch processing pipeline:

    ```bash
    python main_batch.py
    ```

3. Results will be generated in the output directory specified in the configuration.

### Key Components

#### Document Processing

- `DocumentPreprocessor`: Handles PDF parsing and chunk extraction.
- Supports text, table, and image extraction from multimodal documents.

#### Causal Analysis

- `CausalSCMBuilder`: Constructs initial structural causal models.
- `MonotonicConstraintGenerator`: Generates reasoning constraints based on causal chains.
- Implements causal strength thresholds and trend analysis.

#### Batch Processing

- `BatchProcessor`: Manages parallel processing of multiple documents.
- Comprehensive error logging and recovery mechanisms.

## Project Structure

```
AAAI-2026-CausalMIR/
├── datasets/                # Data processing and dataset management
│   ├── longdocurl.py        # Handles LongDocURL dataset loading and preprocessing
│   ├── mmlongbench.py       # Handles MMLongBench dataset loading and preprocessing
│   ├── pre_processer.py     # PDF preprocessing: text, table, and image extraction
├── models/                  # Core causal analysis and reasoning models
│   ├── causal_scm.py        # Constructs structural causal models (SCM)
│   ├── causal_intervention.py # Performs causal interventions and effect estimation
│   ├── monotonic_constraint.py # Generates monotonicity constraints based on causal chains
├── utils/                   # Utility modules for processing and encoding
│   ├── batch_processor.py   # Manages parallel processing of documents
│   ├── multimodal_encoder.py # Encodes multimodal data (text, images, tables)
├── config.py                # Configuration file for system parameters
├── requirements_extended.txt # List of required Python packages
├── main_batch.py            # Entry point for batch processing pipeline
├── README.md                # Project documentation
└── output/                  # Directory for storing generated results
```