"""
Gemini Deep Research API 客户端

功能：
- 调用 Gemini Deep Research API
- 处理 API 响应和错误
- 支持异步和同步调用
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class GeminiDeepResearchClient:
    """
    Gemini Deep Research API 客户端
    
    使用示例：
        >>> client = GeminiDeepResearchClient(api_key='your_api_key')
        >>> result = client.deep_research('分析特斯拉2024年Q4财报')
        >>> print(result['content'])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 300,
        max_retries: int = 3
    ):
        """
        初始化 Gemini Deep Research 客户端
        
        Args:
            api_key: Gemini API Key，如果不提供则从环境变量 GEMINI_API_KEY 读取
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            
        Raises:
            ValueError: 如果 API Key 未提供或未找到
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', "")
        if not self.api_key:
            raise ValueError(
                "API Key 未提供。请通过参数传入或设置环境变量 GEMINI_API_KEY"
            )
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Gemini Deep Research 客户端初始化完成，base_url={self.base_url}")
    
    def deep_research(
        self,
        query: str,
        model: str = "gemini-2.0-flash-thinking-exp",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行深度研究查询
        
        Args:
            query: 研究问题或主题
            model: 使用的模型名称
            temperature: 生成温度 (0.0-1.0)
            max_output_tokens: 最大输出 token 数
            metadata: 额外的元数据信息
            
        Returns:
            Dict 包含以下字段：
                - content: 研究报告内容
                - query: 原始查询
                - model: 使用的模型
                - timestamp: 生成时间戳
                - metadata: 相关元数据
                
        Raises:
            Exception: API 调用失败时抛出
        """
        logger.info(f"开始深度研究查询: query='{query}', model={model}")
        
        start_time = time.time()
        
        # 构建请求
        request_data = {
            "query": query,
            "model": model,
            "temperature": temperature,
        }
        
        if max_output_tokens:
            request_data["max_output_tokens"] = max_output_tokens
        
        # 调用 API（这里使用实际的 API 调用逻辑）
        try:
            result = self._call_api(request_data)
            elapsed_time = time.time() - start_time
            
            logger.info(
                f"深度研究完成: "
                f"query='{query}', "
                f"elapsed_time={elapsed_time:.2f}s, "
                f"content_length={len(result.get('content', ''))}"
            )
            
            # 组装返回结果
            response = {
                'content': result.get('content', ''),
                'query': query,
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed_time,
                'metadata': metadata or {},
                'thinking_process': result.get('thinking_process', ''),
            }
            
            return response
            
        except Exception as e:
            logger.error(f"深度研究失败: query='{query}', error={str(e)}", exc_info=True)
            raise
    
    def _call_api(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        实际调用 Gemini API
        
        Args:
            request_data: API 请求数据
            
        Returns:
            API 响应数据
            
        Raises:
            Exception: API 调用失败
        """
        import requests
        
        endpoint = f"{self.base_url}/models/{request_data['model']}:generateContent"
        
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.api_key,
        }
        
        # 构建 Gemini API 格式的请求体
        payload = {
            "contents": [{
                "parts": [{
                    "text": request_data['query']
                }]
            }],
            "generationConfig": {
                "temperature": request_data.get('temperature', 0.7),
            }
        }
        
        if 'max_output_tokens' in request_data:
            payload['generationConfig']['maxOutputTokens'] = request_data['max_output_tokens']
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API 调用尝试 {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                
                # 解析 Gemini API 响应
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    content_parts = candidate.get('content', {}).get('parts', [])
                    
                    if content_parts:
                        content = content_parts[0].get('text', '')
                        
                        return {
                            'content': content,
                            'thinking_process': '',  # Gemini 可能有思考过程
                            'finish_reason': candidate.get('finishReason', 'STOP'),
                        }
                
                raise ValueError("API 响应格式不正确")
                
            except requests.exceptions.Timeout:
                logger.warning(f"API 调用超时 (尝试 {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("API 调用达到最大重试次数")
    
    def batch_research(
        self,
        queries: List[str],
        model: str = "gemini-2.0-flash-thinking-exp",
        temperature: float = 0.7,
        delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        批量执行深度研究查询
        
        Args:
            queries: 查询列表
            model: 使用的模型名称
            temperature: 生成温度
            delay: 每次请求之间的延迟（秒）
            
        Returns:
            研究结果列表
        """
        logger.info(f"开始批量研究: count={len(queries)}")
        
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"处理查询 {i}/{len(queries)}: '{query}'")
            
            try:
                result = self.deep_research(
                    query=query,
                    model=model,
                    temperature=temperature
                )
                results.append(result)
                
                # 延迟以避免 API 限流
                if i < len(queries):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"查询失败: '{query}', error={str(e)}")
                results.append({
                    'content': '',
                    'query': query,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
        
        logger.info(f"批量研究完成: total={len(queries)}, success={sum(1 for r in results if 'error' not in r)}")
        return results
    
    def chat(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行普通的 Gemini 对话（非 Deep Research）
        
        Args:
            prompt: 用户提示词
            model: 使用的模型名称
            temperature: 生成温度 (0.0-1.0)
            max_output_tokens: 最大输出 token 数
            system_instruction: 系统指令（可选）
            
        Returns:
            Dict 包含以下字段：
                - content: 回复内容
                - prompt: 原始提示词
                - model: 使用的模型
                - timestamp: 生成时间戳
                - usage: token 使用情况（如果有）
                
        Raises:
            Exception: API 调用失败时抛出
        """
        logger.info(f"开始 Gemini 对话: prompt='{prompt[:50]}...', model={model}")
        
        start_time = time.time()
        
        try:
            import requests
            
            endpoint = f"{self.base_url}/models/{model}:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key,
            }
            
            # 构建请求体
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                }
            }
            
            if max_output_tokens:
                payload['generationConfig']['maxOutputTokens'] = max_output_tokens
            
            if system_instruction:
                payload['systemInstruction'] = {
                    "parts": [{"text": system_instruction}]
                }
            
            # 发送请求
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            elapsed_time = time.time() - start_time
            
            # 解析响应
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                content_parts = candidate.get('content', {}).get('parts', [])
                
                if content_parts:
                    content = content_parts[0].get('text', '')
                    
                    # 提取 token 使用情况
                    usage = {}
                    if 'usageMetadata' in data:
                        usage = {
                            'prompt_tokens': data['usageMetadata'].get('promptTokenCount', 0),
                            'candidates_tokens': data['usageMetadata'].get('candidatesTokenCount', 0),
                            'total_tokens': data['usageMetadata'].get('totalTokenCount', 0),
                        }
                    
                    logger.info(
                        f"Gemini 对话完成: "
                        f"elapsed_time={elapsed_time:.2f}s, "
                        f"content_length={len(content)}, "
                        f"tokens={usage.get('total_tokens', 'N/A')}"
                    )
                    
                    return {
                        'content': content,
                        'prompt': prompt,
                        'model': model,
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_time': elapsed_time,
                        'usage': usage,
                        'finish_reason': candidate.get('finishReason', 'STOP'),
                    }
            
            raise ValueError("API 响应格式不正确或无内容")
            
        except Exception as e:
            logger.error(f"Gemini 对话失败: prompt='{prompt[:50]}...', error={str(e)}", exc_info=True)
            raise

