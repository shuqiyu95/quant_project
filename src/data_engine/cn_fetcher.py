"""
A股数据获取器
使用 AkShare 获取A股市场数据
支持日线、分钟线、行业数据、微观数据等
"""
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import akshare as ak
from .base import BaseDataFetcher, MarketType


class CNFetcher(BaseDataFetcher):
    """A股数据获取器 - 全面支持各类A股数据"""
    
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
        end_date: datetime,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取A股日线数据
        
        Args:
            symbol: 股票代码（如 600519）
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型 "qfq"前复权/"hfq"后复权/""不复权
            
        Returns:
            DataFrame with standardized columns
        """
        try:
            # 使用 akshare 获取A股数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust=adjust
            )
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # AkShare 返回的列名是中文，需要转换
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 保留所有可用列
            available_cols = [col for col in column_mapping.values() if col in df.columns]
            df = df[available_cols]
            
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
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'amplitude', 'pct_change', 'change', 'turnover']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 添加市场标识
            df['market'] = self.market_type
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch CN daily data for {symbol}: {str(e)}")
    
    def fetch_intraday_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        period: str = "5"
    ) -> pd.DataFrame:
        """
        获取A股分钟线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 分钟周期 "1"/"5"/"15"/"30"/"60"
            
        Returns:
            分钟级别的OHLCV数据
        """
        try:
            # akshare 分钟数据获取
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                end_date=end_date.strftime('%Y-%m-%d %H:%M:%S'),
                adjust="qfq"
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            column_mapping = {
                '时间': 'datetime',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }
            
            df = df.rename(columns=column_mapping)
            available_cols = [col for col in column_mapping.values() if col in df.columns]
            df = df[available_cols]
            
            # 设置时间索引
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # 设置时区
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['symbol'] = symbol
            df['period'] = period
            
            return df
            
        except Exception as e:
            print(f"Warning: Failed to fetch intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_auction_data(self, symbol: str, trade_date: str) -> Optional[dict]:
        """
        获取集合竞价数据（9:25）
        
        Args:
            symbol: 股票代码
            trade_date: 交易日期 'YYYYMMDD'
            
        Returns:
            包含竞价量、竞价价格等信息的字典
        """
        try:
            # 获取当日分钟数据，提取9:25-9:30的数据
            end_time = datetime.strptime(trade_date, '%Y%m%d') + timedelta(days=1)
            df = self.fetch_intraday_data(
                symbol=symbol,
                start_date=datetime.strptime(trade_date, '%Y%m%d'),
                end_date=end_time,
                period="1"
            )
            
            if df.empty:
                return None
            
            # 筛选9:25-9:31的数据（集合竞价和开盘）
            df_auction = df.between_time('09:25', '09:31')
            
            if df_auction.empty:
                return None
            
            auction_info = {
                'date': trade_date,
                'symbol': symbol,
                'auction_price': df_auction.iloc[0]['open'] if len(df_auction) > 0 else None,
                'auction_volume': df_auction.iloc[0]['volume'] if len(df_auction) > 0 else None,
                'first_minute_volume': df_auction['volume'].sum(),
                'open_price': df_auction.iloc[0]['open'] if len(df_auction) > 0 else None
            }
            
            return auction_info
            
        except Exception as e:
            print(f"Failed to fetch auction data: {str(e)}")
            return None
    
    def fetch_industry_data(self, symbol: str) -> Optional[dict]:
        """
        获取股票所属行业信息（申万行业）
        
        Args:
            symbol: 股票代码
            
        Returns:
            行业信息字典
        """
        try:
            # 获取申万行业分类
            df_industry = ak.stock_sector_spot()
            
            # 获取个股行业归属
            df_stock_industry = ak.stock_individual_info_em(symbol=symbol)
            
            if df_stock_industry.empty:
                return None
            
            # 提取行业信息
            industry_info = {
                'symbol': symbol,
                'industry': df_stock_industry[df_stock_industry['item'] == '行业']['value'].values[0] 
                           if '行业' in df_stock_industry['item'].values else None,
                'sector': df_stock_industry[df_stock_industry['item'] == '所属概念']['value'].values[0]
                         if '所属概念' in df_stock_industry['item'].values else None,
            }
            
            return industry_info
            
        except Exception as e:
            print(f"Failed to fetch industry data: {str(e)}")
            return None
    
    def fetch_sw_industry_index(
        self,
        industry_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取申万行业指数数据
        
        Args:
            industry_code: 申万行业代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            行业指数数据
        """
        try:
            # 获取申万行业指数历史数据
            df = ak.index_hist_sw(
                symbol=industry_code,
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Shanghai')
            
            df['industry_code'] = industry_code
            
            return df
            
        except Exception as e:
            print(f"Failed to fetch SW industry index: {str(e)}")
            return pd.DataFrame()
    
    def fetch_turnover_quantile(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int = 100
    ) -> Optional[float]:
        """
        计算当前换手率在历史中的分位数
        
        Args:
            symbol: 股票代码
            current_date: 当前日期
            lookback_days: 回溯天数
            
        Returns:
            换手率分位数 (0-1)
        """
        try:
            start_date = current_date - timedelta(days=lookback_days * 2)  # 多取一些以确保有足够交易日
            df = self.fetch_daily_data(symbol, start_date, current_date)
            
            if df.empty or 'turnover' not in df.columns:
                return None
            
            # 取最近lookback_days的交易日
            df = df.tail(lookback_days)
            
            if len(df) < 2:
                return None
            
            # 计算当前换手率的分位数
            current_turnover = df.iloc[-1]['turnover']
            quantile = (df['turnover'] < current_turnover).sum() / len(df)
            
            return quantile
            
        except Exception as e:
            print(f"Failed to calculate turnover quantile: {str(e)}")
            return None
    
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
    
    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        批量获取实时行情
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时行情DataFrame
        """
        try:
            df_all = ak.stock_zh_a_spot_em()
            df_filtered = df_all[df_all['代码'].isin(symbols)]
            
            # 标准化列名
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'pre_close',
                '换手率': 'turnover'
            }
            
            df_filtered = df_filtered.rename(columns=column_mapping)
            available_cols = [col for col in column_mapping.values() if col in df_filtered.columns]
            
            return df_filtered[available_cols]
            
        except Exception as e:
            print(f"Failed to fetch realtime quotes: {str(e)}")
            return pd.DataFrame()

