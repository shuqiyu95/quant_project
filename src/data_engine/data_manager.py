"""
数据管理器
统一管理不同市场的数据获取和缓存
支持日线/分钟线数据、增量更新、行业数据等
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List, Dict
import pandas as pd
import json

from .base import MarketType
from .us_fetcher import USFetcher
from .cn_fetcher import CNFetcher


class DataManager:
    """数据管理器 - 自动识别市场并获取数据，支持增量更新"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据缓存目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建元数据目录
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def fetch_data_incremental(
        self,
        symbol: str,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        增量更新数据 - 只获取缓存之后的新数据
        
        Args:
            symbol: 股票代码
            end_date: 结束日期（默认为今天）
            
        Returns:
            完整的历史数据（包含新增部分）
        """
        symbol = symbol.strip().upper()
        
        if end_date is None:
            end_date = pd.Timestamp.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        else:
            end_date = pd.Timestamp(end_date)
        
        cache_file = self._get_cache_path(symbol)
        metadata_file = self._get_metadata_path(symbol)
        
        # 检查缓存和元数据
        if cache_file.exists() and metadata_file.exists():
            try:
                # 读取缓存数据
                df_cached = pd.read_parquet(cache_file)
                df_cached.index = pd.to_datetime(df_cached.index)
                
                # 读取元数据
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                last_update = pd.to_datetime(metadata['last_update'])
                cache_end = df_cached.index.max()
                
                # 移除时区信息用于比较
                cache_end_naive = cache_end.tz_localize(None) if cache_end.tz else cache_end
                end_date_naive = end_date.tz_localize(None) if end_date.tz else end_date
                
                # 如果缓存已是最新，直接返回
                if cache_end_naive >= end_date_naive:
                    print(f"Cache is up-to-date for {symbol}")
                    return df_cached
                
                # 获取增量数据（从缓存结束日期的下一天开始）
                start_date_incremental = cache_end + timedelta(days=1)
                print(f"Fetching incremental data for {symbol} from {start_date_incremental.date()} to {end_date.date()}")
                
                fetcher = self.get_fetcher(symbol)
                df_new = fetcher.fetch_daily_data(symbol, start_date_incremental, end_date)
                
                if not df_new.empty:
                    # 合并数据
                    df_combined = pd.concat([df_cached, df_new])
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                    df_combined = df_combined.sort_index()
                    
                    # 更新缓存
                    self._save_to_cache(df_combined, symbol)
                    self._save_metadata(symbol, df_combined)
                    
                    print(f"Added {len(df_new)} new records for {symbol}")
                    return df_combined
                else:
                    print(f"No new data available for {symbol}")
                    return df_cached
                
            except Exception as e:
                print(f"Error in incremental update: {e}, falling back to full fetch")
        
        # 如果没有缓存，执行完整获取（默认一年）
        start_date = end_date - timedelta(days=365)
        return self.fetch_data(symbol, start_date, end_date, use_cache=False)
    
    def fetch_intraday_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "5",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取分钟线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 分钟周期（1/5/15/30/60）
            use_cache: 是否使用缓存
            
        Returns:
            分钟级别DataFrame
        """
        symbol = symbol.strip().upper()
        
        # 处理日期
        if end_date is None:
            end_date = pd.Timestamp.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 默认30天分钟数据
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # 分钟数据缓存路径
        cache_file = self._get_intraday_cache_path(symbol, period)
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                
                cache_min = df.index.min()
                cache_max = df.index.max()
                
                # 检查缓存覆盖范围
                if cache_min <= start_date and cache_max >= end_date:
                    print(f"Using cached intraday data for {symbol}")
                    return df.loc[start_date:end_date]
            except Exception as e:
                print(f"Failed to load intraday cache: {e}")
        
        # 获取新数据
        print(f"Fetching intraday data for {symbol} ({period}min)")
        fetcher = self.get_fetcher(symbol)
        
        # 只有A股支持分钟数据
        if isinstance(fetcher, CNFetcher):
            df = fetcher.fetch_intraday_data(symbol, start_date, end_date, period)
            
            if not df.empty:
                self._save_intraday_cache(df, symbol, period)
            
            return df
        else:
            raise NotImplementedError("Intraday data only supported for CN market")
    
    def fetch_industry_data(self, symbol: str) -> Optional[Dict]:
        """
        获取股票行业数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            行业信息字典
        """
        fetcher = self.get_fetcher(symbol)
        
        if isinstance(fetcher, CNFetcher):
            return fetcher.fetch_industry_data(symbol)
        else:
            raise NotImplementedError("Industry data only supported for CN market")
    
    def fetch_auction_data(self, symbol: str, trade_date: str) -> Optional[Dict]:
        """
        获取集合竞价数据
        
        Args:
            symbol: 股票代码
            trade_date: 交易日期 'YYYYMMDD'
            
        Returns:
            竞价数据字典
        """
        fetcher = self.get_fetcher(symbol)
        
        if isinstance(fetcher, CNFetcher):
            return fetcher.fetch_auction_data(symbol, trade_date)
        else:
            raise NotImplementedError("Auction data only supported for CN market")
    
    def _get_intraday_cache_path(self, symbol: str, period: str) -> Path:
        """获取分钟线缓存路径"""
        market = self.identify_market(symbol)
        intraday_dir = self.data_dir / market.lower() / "intraday" / f"{period}min"
        intraday_dir.mkdir(parents=True, exist_ok=True)
        return intraday_dir / f"{symbol}.parquet"
    
    def _save_intraday_cache(self, df: pd.DataFrame, symbol: str, period: str):
        """保存分钟线数据到缓存"""
        try:
            cache_file = self._get_intraday_cache_path(symbol, period)
            df.to_parquet(cache_file, compression='snappy')
            print(f"Intraday data cached to {cache_file}")
        except Exception as e:
            print(f"Failed to cache intraday data: {e}")
    
    def _get_metadata_path(self, symbol: str) -> Path:
        """获取元数据文件路径"""
        market = self.identify_market(symbol)
        return self.metadata_dir / f"{symbol}_{market}.json"
    
    def _save_metadata(self, symbol: str, df: pd.DataFrame):
        """保存数据元信息"""
        try:
            metadata = {
                'symbol': symbol,
                'market': self.identify_market(symbol),
                'last_update': pd.Timestamp.now().isoformat(),
                'data_start': df.index.min().isoformat(),
                'data_end': df.index.max().isoformat(),
                'record_count': len(df)
            }
            
            metadata_file = self._get_metadata_path(symbol)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save metadata: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清除缓存"""
        if symbol:
            cache_file = self._get_cache_path(symbol)
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cache cleared for {symbol}")
            
            # 清除元数据
            metadata_file = self._get_metadata_path(symbol)
            if metadata_file.exists():
                metadata_file.unlink()
        else:
            # 清除所有缓存
            for market_dir in self.data_dir.iterdir():
                if market_dir.is_dir() and market_dir.name != "metadata":
                    for cache_file in market_dir.rglob("*.parquet"):
                        cache_file.unlink()
            
            # 清除所有元数据
            for metadata_file in self.metadata_dir.glob("*.json"):
                metadata_file.unlink()
                
            print("All cache cleared")

