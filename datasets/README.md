# 使用API模型处理
python preprocessor.py \
    --json_dir /path/to/json_files \
    --img_dir /path/to/images \
    --output_dir /path/to/output \
    --table_desc_model gpt-4o \
    --api_base_url https://api.openai.com/v1 \
    --api_key your_api_key

# 使用本地模型处理
python preprocessor.py \
    --json_dir /path/to/json_files \
    --img_dir /path/to/images \
    --output_dir /path/to/output \
    --table_desc_model lmsys/vicuna-7b-v1.5 \
    --device cuda