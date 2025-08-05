class Config:
    def __init__(self):
        # 路径配置（扩展）
        self.data_dir = "./data"          # 数据目录（原无，新增）
        self.cache_dir = "./cache"        # 缓存目录（新增）
        self.logs_dir = "./logs"          # 日志目录（新增）
        self.output_dir = "./output"      # 输出目录（保留原功能）
        self.temp_dir = f"{self.cache_dir}/temp"  # 临时文件（原config.py迁移）
        
        # 分块配置（保留原功能）
        self.max_chunk_length = 300
        self.page_intervals = [(1, 16), (17, 32), (33, 48), (49, 64), (65, 80)]
        
        # 因果分析阈值（保留原功能）
        self.causal_threshold = 0.4
        self.strong_threshold = 0.6
        
        # 模型配置（保留原功能）
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.sentence_bert_model = "all-MiniLM-L6-v2"
        
        # 新增：批量处理配置
        self.batch_size = 10              # 每次处理文档数
        self.batch_workers = 4            # 并行线程数
        
        # 新增：API配置
        self.api_port = 8000
        self.api_host = "0.0.0.0"
        
        # 创建必要目录（扩展）
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
