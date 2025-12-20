"""
数据管理器
统一管理不同市场的数据获取和缓存
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .base import MarketType
from .us_fetcher import USFetcher
from .cn_fetcher import CNFetcher


class DataManager:
    """数据管理器 - 自动识别市场并获取数据"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据缓存目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化不同市场的数据获取器
        self.us_fetcher = USFetcher()
        self.cn_fetcher = CNFetcher()
        
    def identify_market(self, symbol: str) -> str:
        """
        识别股票代码属于哪个市场
        
        Args:
            symbol: 股票代码
            
        Returns:
            市场类型 (US/CN/UNKNOWN)
        """
        symbol = symbol.strip().upper()
        
        # 检查是否为A股（6位数字）
        if self.cn_fetcher.validate_symbol(symbol):
            return MarketType.CN
        
        # 检查是否为美股（1-5个大写字母）
        if self.us_fetcher.validate_symbol(symbol):
            return MarketType.US
        
        return MarketType.UNKNOWN
    
    def get_fetcher(self, symbol: str):
        """根据股票代码获取对应的数据获取器"""
        market = self.identify_market(symbol)
        
        if market == MarketType.US:
            return self.us_fetcher
        elif market == MarketType.CN:
            return self.cn_fetcher
        else:
            raise ValueError(f"Unable to identify market for symbol: {symbol}")
    
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取股票数据（自动识别市场）
        
        Args:
            symbol: 股票代码（如 AAPL 或 600519）
            start_date: 开始日期（默认为一年前）
            end_date: 结束日期（默认为今天）
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.strip().upper()
        
        # 处理日期参数
        if end_date is None:
            end_date = pd.Timestamp.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        else:
            end_date = pd.Timestamp(end_date)
            
        if start_date is None:
            # 默认获取一年数据
            start_date = end_date - timedelta(days=365)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        else:
            start_date = pd.Timestamp(start_date)
        
        # 移除时区信息用于比较（缓存比较时使用）
        start_date_naive = start_date.tz_localize(None) if hasattr(start_date, 'tz') and start_date.tz else start_date
        end_date_naive = end_date.tz_localize(None) if hasattr(end_date, 'tz') and end_date.tz else end_date
        
        # 检查缓存
        cache_file = self._get_cache_path(symbol)
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                
                # 获取缓存日期范围（移除时区用于比较）
                cache_min = df.index.min().tz_localize(None) if df.index.min().tz else df.index.min()
                cache_max = df.index.max().tz_localize(None) if df.index.max().tz else df.index.max()
                
                # 检查缓存是否涵盖所需日期范围
                if cache_min <= start_date_naive and cache_max >= end_date_naive:
                    print(f"Using cached data for {symbol}")
                    # 使用原始日期范围进行切片
                    return df.loc[start_date_naive:end_date_naive]
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        # 从远程获取数据
        print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        fetcher = self.get_fetcher(symbol)
        df = fetcher.fetch_daily_data(symbol, start_date, end_date)
        
        # 保存到缓存
        if not df.empty:
            self._save_to_cache(df, symbol)
        
        return df
    
    def fetch_multiple(
        self,
        symbols: list,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> dict:
        """
        批量获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            字典 {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, start_date, end_date, use_cache)
                results[symbol] = df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def _get_cache_path(self, symbol: str) -> Path:
        """获取缓存文件路径"""
        market = self.identify_market(symbol)
        market_dir = self.data_dir / market.lower()
        market_dir.mkdir(parents=True, exist_ok=True)
        return market_dir / f"{symbol}.parquet"
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str):
        """保存数据到缓存"""
        try:
            cache_file = self._get_cache_path(symbol)
            df.to_parquet(cache_file, compression='snappy')
            print(f"Data cached to {cache_file}")
        except Exception as e:
            print(f"Failed to cache data: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清除缓存"""
        if symbol:
            cache_file = self._get_cache_path(symbol)
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cache cleared for {symbol}")
        else:
            # 清除所有缓存
            for market_dir in self.data_dir.iterdir():
                if market_dir.is_dir():
                    for cache_file in market_dir.glob("*.parquet"):
                        cache_file.unlink()
            print("All cache cleared")

