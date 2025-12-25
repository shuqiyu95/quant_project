"""
LLM 模块 - 支持 Gemini Deep Research API 调用

核心功能：
- Gemini Deep Research API 集成
- 研究报告生成与保存
- 报告管理与检索
"""

from .gemini_client import GeminiDeepResearchClient
from .report_manager import ReportManager

__all__ = [
    'GeminiDeepResearchClient',
    'ReportManager',
]

