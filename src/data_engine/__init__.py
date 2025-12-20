"""
数据引擎模块
支持美股（yfinance）和A股（AkShare）的数据获取
"""

from .base import BaseDataFetcher, MarketType
from .us_fetcher import USFetcher
from .cn_fetcher import CNFetcher
from .data_manager import DataManager

__all__ = [
    'BaseDataFetcher',
    'MarketType',
    'USFetcher',
    'CNFetcher',
    'DataManager'
]

