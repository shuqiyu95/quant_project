"""
A股数据获取器
使用 AkShare 获取A股市场数据
"""
import re
from datetime import datetime
from typing import Optional
import pandas as pd
import akshare as ak
from .base import BaseDataFetcher, MarketType


class CNFetcher(BaseDataFetcher):
    """A股数据获取器"""
    
    def __init__(self):
        self.market_type = MarketType.CN
        
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证A股代码格式
        A股代码是6位数字
        """
        pattern = r'^\d{6}$'
        return bool(re.match(pattern, symbol))
    
    def fetch_daily_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取A股日线数据
        
        Args:
            symbol: 股票代码（如 600519）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with standardized columns
        """
        try:
            # 使用 akshare 获取A股数据
            # ak.stock_zh_a_hist 函数获取历史行情数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # AkShare 返回的列名是中文，需要转换
            # 日期、开盘、收盘、最高、最低、成交量
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 只保留需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # 设置日期为索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 设置时区为中国时区
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            else:
                df.index = df.index.tz_convert('Asia/Shanghai')
            
            df.index.name = 'date'
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 添加市场标识
            df['market'] = self.market_type
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch CN data for {symbol}: {str(e)}")
    
    def get_stock_info(self, symbol: str) -> Optional[dict]:
        """获取股票基本信息"""
        try:
            # 获取实时行情数据
            df = ak.stock_zh_a_spot_em()
            stock_info = df[df['代码'] == symbol]
            if not stock_info.empty:
                return stock_info.iloc[0].to_dict()
            return None
        except Exception:
            return None

