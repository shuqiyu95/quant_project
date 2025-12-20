"""
数据引擎基础类定义
提供统一的数据接口抽象
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd


class BaseDataFetcher(ABC):
    """数据获取器基类"""
    
    @abstractmethod
    def fetch_daily_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            date is index, timezone-aware
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """验证股票代码格式是否正确"""
        pass


class MarketType:
    """市场类型枚举"""
    US = "US"
    CN = "CN"
    UNKNOWN = "UNKNOWN"

