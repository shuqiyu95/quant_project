"""
同花顺热股榜数据抓取与热度因子生成
功能：
1. 每日抓取同花顺热股榜前10名单
2. 数据存储到 data/hot_stock 文件夹，按日期命名
3. 每周统计在榜次数和排名，生成股票热度因子
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import akshare as ak
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class HotStockTracker:
    """同花顺热股榜跟踪器"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化热股跟踪器
        
        Args:
            data_dir: 数据存储目录，默认为 project_root/data/hot_stock
        """
        if data_dir is None:
            self.data_dir = project_root / "data" / "hot_stock"
        else:
            self.data_dir = Path(data_dir)
        
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"数据存储目录: {self.data_dir}")
    
    def calculate_stock_returns(self, symbol: str, current_price: float) -> Dict[str, float]:
        """
        计算股票的多周期涨幅
        
        Args:
            symbol: 股票代码（6位数字）
            current_price: 当前价格
            
        Returns:
            字典包含: return_1d, return_3d, return_5d, return_10d
        """
        returns = {
            'return_1d': None,
            'return_3d': None,
            'return_5d': None,
            'return_10d': None
        }
        
        try:
            # 获取最近30天的历史数据（确保有足够数据）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=40)).strftime('%Y%m%d')
            
            # 使用 akshare 获取A股历史数据
            df_hist = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df_hist is None or df_hist.empty or len(df_hist) < 2:
                return returns
            
            # 获取收盘价列
            if '收盘' in df_hist.columns:
                prices = df_hist['收盘'].values
            elif 'close' in df_hist.columns:
                prices = df_hist['close'].values
            else:
                return returns
            
            # 计算不同周期的涨幅
            # 使用历史收盘价而不是当前价格更准确
            periods = {
                'return_1d': 1,
                'return_3d': 3,
                'return_5d': 5,
                'return_10d': 10
            }
            
            for key, days in periods.items():
                if len(prices) > days:
                    # 涨幅 = (最新价 - N日前价) / N日前价 * 100
                    old_price = prices[-(days+1)]
                    new_price = prices[-1]
                    if old_price > 0:
                        returns[key] = round((new_price - old_price) / old_price * 100, 2)
            
        except Exception as e:
            # 静默处理错误，返回 None 值
            pass
        
        return returns
    
    def fetch_hot_stocks(self, top_n: int = 10) -> pd.DataFrame:
        """
        抓取同花顺热股榜数据
        
        Args:
            top_n: 获取前N名，默认10
            
        Returns:
            DataFrame with columns: [rank, symbol, name, price, change_pct, ...]
        """
        try:
            print(f"正在抓取热股榜前{top_n}名...")
            
            # 使用 akshare 获取热股榜数据
            # stock_hot_rank_em: 东方财富-个股人气榜-实时变动（推荐）
            # stock_hot_rank_latest_em: 东方财富-个股人气榜-最新排名
            # stock_rank_cxg_ths: 同花顺-持续关注
            df = ak.stock_hot_rank_em()
            print("使用接口: stock_hot_rank_em (东方财富-个股人气榜)")
            
            if df is None or df.empty:
                print("警告: 未获取到数据")
                return pd.DataFrame()
            
            # 打印原始列名以便调试
            print(f"原始列名: {df.columns.tolist()}")
            
            # 只保留前N名
            df = df.head(top_n).copy()
            
            # 标准化列名 - 根据实际返回的列名进行调整
            column_mapping = {
                '序号': 'rank',
                '排名': 'rank',
                '当前排名': 'rank',
                '股票代码': 'symbol',
                '代码': 'symbol',
                '股票简称': 'name',
                '股票名称': 'name',
                '名称': 'name',
                '最新价': 'price',
                '现价': 'price',
                '涨跌幅': 'change_pct',
                '涨跌': 'change_pct',
                '涨幅': 'change_pct',
                '涨跌额': 'change_amount'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # 添加抓取时间
            df['fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # 确保有 rank 列
            if 'rank' not in df.columns:
                df['rank'] = range(1, len(df) + 1)
            
            # 确保 symbol 列是字符串格式（6位数字）
            # 处理 SH600118 或 SZ000665 格式，提取后6位数字
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str).str.replace('SH', '').str.replace('SZ', '').str.zfill(6)
            
            print(f"成功抓取 {len(df)} 只热股")
            print(f"标准化后列名: {df.columns.tolist()}")
            
            # 添加多周期涨幅数据
            print("\n正在获取各股票的历史涨幅数据...")
            returns_list = []
            for idx, row in df.iterrows():
                symbol = row['symbol']
                price = row.get('price', 0)
                
                print(f"  [{idx+1}/{len(df)}] {symbol} {row.get('name', '')}", end='')
                
                returns = self.calculate_stock_returns(symbol, price)
                returns_list.append(returns)
                
                # 显示获取结果
                if returns['return_1d'] is not None:
                    print(f" ✓ (1d: {returns['return_1d']:.2f}%)")
                else:
                    print(f" ✗ (数据不足)")
            
            # 将涨幅数据添加到 DataFrame
            returns_df = pd.DataFrame(returns_list)
            df = pd.concat([df, returns_df], axis=1)
            
            print(f"\n完成！共获取 {len(df)} 只热股的涨幅数据")
            
            return df
            
        except Exception as e:
            print(f"抓取热股榜失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_daily_data(self, df: pd.DataFrame, date: Optional[str] = None) -> str:
        """
        保存每日热股数据
        
        Args:
            df: 热股数据
            date: 日期字符串 (YYYY-MM-DD)，默认为今天
            
        Returns:
            保存的文件路径
        """
        if df.empty:
            print("数据为空，跳过保存")
            return ""
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # 复制数据框避免修改原始数据
        df_save = df.copy()
        
        # 去掉不需要的列
        columns_to_drop = ['fetch_time', 'fetch_date', 'change_amount']
        for col in columns_to_drop:
            if col in df_save.columns:
                df_save = df_save.drop(columns=[col])
        
        # 对所有浮点数列保留两位小数
        for col in df_save.columns:
            if df_save[col].dtype in ['float64', 'float32']:
                df_save[col] = df_save[col].round(2)
        
        # 文件名格式: YYYY-MM-DD.csv
        filename = f"{date}.csv"
        filepath = self.data_dir / filename
        
        # 保存为CSV
        df_save.to_csv(filepath, index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"数据已保存到: {filepath}")
        
        return str(filepath)
    
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """
        加载指定日期的热股数据
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            DataFrame
        """
        filename = f"{date}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"文件不存在: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # 确保 symbol 列是6位字符串格式
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        
        return df
    
    def load_date_range_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载日期范围内的所有热股数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            合并后的DataFrame
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            df = self.load_daily_data(date_str)
            if not df.empty:
                all_data.append(df)
            current += timedelta(days=1)
        
        if not all_data:
            print(f"日期范围 {start_date} 到 {end_date} 内没有数据")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"加载了 {len(all_data)} 天的数据，共 {len(combined_df)} 条记录")
        
        return combined_df
    
    def calculate_weekly_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算周统计数据：在榜次数和平均排名
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [symbol, name, appearance_count, avg_rank, min_rank, max_rank]
        """
        # 加载日期范围内的数据
        df = self.load_date_range_data(start_date, end_date)
        
        if df.empty:
            print("没有数据可供统计")
            return pd.DataFrame()
        
        # 确保必要的列存在
        if 'symbol' not in df.columns or 'rank' not in df.columns:
            print("数据缺少必要的列 (symbol, rank)")
            return pd.DataFrame()
        
        # 按股票代码分组统计
        stats = []
        for symbol, group in df.groupby('symbol'):
            # 获取股票名称
            name = group['name'].iloc[0] if 'name' in group.columns else ''
            
            # 在榜次数
            appearance_count = len(group['fetch_date'].unique())
            
            # 平均排名
            avg_rank = group['rank'].mean()
            
            # 最好排名（数字越小越好）
            min_rank = group['rank'].min()
            
            # 最差排名
            max_rank = group['rank'].max()
            
            stats.append({
                'symbol': symbol,
                'name': name,
                'appearance_count': appearance_count,
                'avg_rank': avg_rank,
                'min_rank': min_rank,
                'max_rank': max_rank
            })
        
        stats_df = pd.DataFrame(stats)
        
        # 按在榜次数和平均排名排序
        stats_df = stats_df.sort_values(
            by=['appearance_count', 'avg_rank'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        print(f"\n周统计完成，共 {len(stats_df)} 只股票上榜")
        
        return stats_df
    
    def generate_heat_factor(
        self, 
        start_date: str, 
        end_date: str,
        method: str = 'weighted'
    ) -> pd.DataFrame:
        """
        生成股票热度因子
        
        热度因子计算方法：
        1. weighted: 综合考虑在榜次数和排名的加权得分
           heat_score = appearance_count * weight1 + (1/avg_rank) * weight2
        2. simple: 简单方法，只考虑在榜次数
        3. rank_based: 基于排名的方法，排名越高得分越高
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            method: 计算方法 ('weighted', 'simple', 'rank_based')
            
        Returns:
            DataFrame with columns: [symbol, name, heat_score, heat_rank, ...]
        """
        # 获取周统计数据
        stats_df = self.calculate_weekly_stats(start_date, end_date)
        
        if stats_df.empty:
            print("没有统计数据，无法生成热度因子")
            return pd.DataFrame()
        
        # 计算热度得分
        if method == 'weighted':
            # 加权方法：在榜次数权重60%，排名权重40%
            # 归一化在榜次数
            max_appearance = stats_df['appearance_count'].max()
            normalized_appearance = stats_df['appearance_count'] / max_appearance
            
            # 归一化排名得分（排名越小越好，所以用倒数）
            # 为避免除零，加一个小值
            rank_score = 1.0 / (stats_df['avg_rank'] + 0.1)
            max_rank_score = rank_score.max()
            normalized_rank_score = rank_score / max_rank_score
            
            # 加权计算
            stats_df['heat_score'] = (
                normalized_appearance * 0.6 + 
                normalized_rank_score * 0.4
            ) * 100  # 乘以100使得分更直观
            
        elif method == 'simple':
            # 简单方法：只看在榜次数
            stats_df['heat_score'] = stats_df['appearance_count'] * 10
            
        elif method == 'rank_based':
            # 基于排名：考虑排名的倒数和在榜次数
            stats_df['heat_score'] = (
                stats_df['appearance_count'] * (10.0 / stats_df['avg_rank'])
            )
        else:
            raise ValueError(f"未知的计算方法: {method}")
        
        # 计算热度排名（得分越高排名越靠前）
        stats_df['heat_rank'] = stats_df['heat_score'].rank(
            ascending=False, 
            method='min'
        ).astype(int)
        
        # 按热度排名排序
        stats_df = stats_df.sort_values('heat_rank').reset_index(drop=True)
        
        # 添加元数据
        stats_df['start_date'] = start_date
        stats_df['end_date'] = end_date
        stats_df['method'] = method
        stats_df['generate_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n热度因子生成完成 (方法: {method})")
        print(f"前5名最热股票:")
        print(stats_df[['heat_rank', 'symbol', 'name', 'heat_score', 
                        'appearance_count', 'avg_rank']].head())
        
        return stats_df
    
    def save_heat_factor(self, heat_df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        保存热度因子数据
        
        Args:
            heat_df: 热度因子数据
            filename: 文件名，默认为 heat_factor_YYYY-MM-DD.csv
            
        Returns:
            保存的文件路径
        """
        if heat_df.empty:
            print("热度因子数据为空，跳过保存")
            return ""
        
        if filename is None:
            end_date = heat_df['end_date'].iloc[0] if 'end_date' in heat_df.columns else datetime.now().strftime('%Y-%m-%d')
            filename = f"heat_factor_{end_date}.csv"
        
        filepath = self.data_dir / filename
        heat_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"热度因子已保存到: {filepath}")
        
        return str(filepath)
    
    def run_daily_task(self, top_n: int = 10) -> bool:
        """
        执行每日任务：抓取并保存热股数据
        
        Args:
            top_n: 抓取前N名
            
        Returns:
            是否成功
        """
        print(f"\n{'='*60}")
        print(f"开始执行每日热股榜抓取任务")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 抓取数据
        df = self.fetch_hot_stocks(top_n=top_n)
        
        if df.empty:
            print("抓取失败，任务终止")
            return False
        
        # 保存数据
        filepath = self.save_daily_data(df)
        
        if filepath:
            print(f"\n每日任务完成！")
            return True
        else:
            print(f"\n保存失败，任务未完成")
            return False
    
    def run_weekly_task(self, days: int = 7, method: str = 'weighted') -> bool:
        """
        执行每周任务：统计并生成热度因子
        
        Args:
            days: 统计最近N天，默认7天
            method: 热度因子计算方法
            
        Returns:
            是否成功
        """
        print(f"\n{'='*60}")
        print(f"开始执行每周热度因子生成任务")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"统计周期: 最近{days}天")
        print(f"{'='*60}\n")
        
        # 计算日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days-1)).strftime('%Y-%m-%d')
        
        print(f"统计日期范围: {start_date} 到 {end_date}")
        
        # 生成热度因子
        heat_df = self.generate_heat_factor(start_date, end_date, method=method)
        
        if heat_df.empty:
            print("生成热度因子失败，任务终止")
            return False
        
        # 保存热度因子
        filepath = self.save_heat_factor(heat_df)
        
        if filepath:
            print(f"\n每周任务完成！")
            return True
        else:
            print(f"\n保存失败，任务未完成")
            return False


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='同花顺热股榜数据抓取与热度因子生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 执行每日抓取任务（抓取前10名）
  python hot_stock.py --daily
  
  # 执行每日抓取任务（抓取前20名）
  python hot_stock.py --daily --top 20
  
  # 执行每周统计任务（统计最近7天）
  python hot_stock.py --weekly
  
  # 执行每周统计任务（统计最近14天）
  python hot_stock.py --weekly --days 14
  
  # 生成指定日期范围的热度因子
  python hot_stock.py --generate --start 2024-01-01 --end 2024-01-07
  
  # 使用不同的计算方法
  python hot_stock.py --weekly --method rank_based
        """
    )
    
    parser.add_argument(
        '--daily', 
        action='store_true',
        help='执行每日抓取任务'
    )
    
    parser.add_argument(
        '--weekly', 
        action='store_true',
        help='执行每周统计任务'
    )
    
    parser.add_argument(
        '--generate',
        action='store_true',
        help='生成热度因子（需指定日期范围）'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='抓取热股榜前N名（默认: 10）'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='统计最近N天（默认: 7）'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='开始日期 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='结束日期 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='weighted',
        choices=['weighted', 'simple', 'rank_based'],
        help='热度因子计算方法（默认: weighted）'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='数据存储目录（默认: data/hot_stock）'
    )
    
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = HotStockTracker(data_dir=args.data_dir)
    
    # 执行任务
    if args.daily:
        success = tracker.run_daily_task(top_n=args.top)
        sys.exit(0 if success else 1)
    
    elif args.weekly:
        success = tracker.run_weekly_task(days=args.days, method=args.method)
        sys.exit(0 if success else 1)
    
    elif args.generate:
        if not args.start or not args.end:
            print("错误: 生成热度因子需要指定 --start 和 --end 参数")
            sys.exit(1)
        
        heat_df = tracker.generate_heat_factor(args.start, args.end, method=args.method)
        if not heat_df.empty:
            tracker.save_heat_factor(heat_df)
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

