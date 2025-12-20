"""
美股数据获取器
使用 yfinance 获取美股市场数据
"""
import re
from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf
from .base import BaseDataFetcher, MarketType


class USFetcher(BaseDataFetcher):
    """美股数据获取器"""
    
    def __init__(self):
        self.market_type = MarketType.US
        
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证美股代码格式
        美股代码通常是1-5个大写字母
        """
        pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(pattern, symbol.upper()))
    
    def fetch_daily_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取美股日线数据
        
        Args:
            symbol: 股票代码（如 AAPL）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with standardized columns
        """
        try:
            # 使用 yfinance 下载数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # 标准化列名
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # 确保索引是日期类型并设置时区（美股使用 US/Eastern）
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('US/Eastern')
            else:
                df.index = df.index.tz_convert('US/Eastern')
            
            df.index.name = 'date'
            
            # 添加市场标识
            df['market'] = self.market_type
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch US data for {symbol}: {str(e)}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception:
            return None

