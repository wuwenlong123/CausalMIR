import os
import logging
import requests
from typing import Dict, Optional

class LLMClient:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.logger = logging.getLogger("llm_client")
        
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[Dict]:
        """调用LLM生成文本"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {str(e)}")
            return None
    
    def extract_answer(self, response: Dict) -> str:
        """从LLM响应中提取答案文本"""
        if not response or "choices" not in response:
            return ""
        return response["choices"][0]["message"]["content"].strip()
