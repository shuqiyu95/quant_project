"""
A股次日高点预测策略

预测次日开盘后30分钟内的最高涨幅，用于开盘买入、高点卖出的交易策略

核心功能：
1. 标签生成：计算次日开盘后30分钟最高涨幅
2. A股特色因子：竞价强度、量能、板块动量等
3. 多分类模型：将涨幅分为5个桶 (+6%, +3%, 0%, -3%, -6%)
4. 回测：评估策略收益

作者：Quant Team
日期：2025-12-23
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import DataManager
from src.factors import Alpha158
from src.backtester import BacktestEngine


class CNIntradayHighPredictor:
    """
    A股次日高点预测器
    
    预测次日开盘后30分钟内的最高涨幅区间
    """
    
    # 涨幅桶定义（百分比）
    BINS = [-np.inf, -3.0, 0.0, 3.0, 6.0, np.inf]
    LABELS = ['<-3%', '-3%~0%', '0%~3%', '3%~6%', '>6%']
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict] = None,
        data_dir: str = 'data'
    ):
        """
        Args:
            model_type: 模型类型 ('random_forest' or 'gbdt')
            model_params: 模型参数
            data_dir: 数据目录
        """
        self.model_type = model_type
        self.model_params = model_params or self._default_params()
        self.data_dir = data_dir
        
        # 初始化组件
        self.dm = DataManager(data_dir=data_dir)
        self.alpha158 = Alpha158()
        self.scaler = StandardScaler()
        
        # 模型
        self.model = self._create_model()
        
        # 特征相关
        self.feature_names_ = None
        self.feature_importance_ = None
        
        print(f"✅ CNIntradayHighPredictor initialized ({model_type})")
    
    def _default_params(self) -> Dict:
        """默认模型参数"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        else:  # gbdt
            return {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'random_state': 42
            }
    
    def _create_model(self):
        """创建分类模型"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == 'gbdt':
            return GradientBoostingClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_label(
        self,
        symbol: str,
        date: pd.Timestamp,
        intraday_df: pd.DataFrame
    ) -> Optional[float]:
        """
        生成标签：次日开盘后30分钟最高涨幅
        
        Args:
            symbol: 股票代码
            date: 当前日期
            intraday_df: 分钟线数据
            
        Returns:
            max_return: 开盘后30分钟最高涨幅（百分比）
        """
        try:
            # 找到次日的数据
            next_day = date + timedelta(days=1)
            
            # 筛选次日的数据
            next_day_data = intraday_df[
                (intraday_df.index.date == next_day.date())
            ]
            
            if len(next_day_data) == 0:
                return None
            
            # 获取开盘价（9:30）
            open_price = next_day_data.iloc[0]['open']
            
            # 获取开盘后30分钟内的数据（9:30-10:00）
            morning_data = next_day_data.between_time('09:30', '10:00')
            
            if len(morning_data) == 0:
                return None
            
            # 计算30分钟内的最高涨幅
            high_price = morning_data['high'].max()
            max_return = (high_price - open_price) / open_price * 100  # 转换为百分比
            
            return max_return
            
        except Exception as e:
            print(f"Warning: Failed to generate label for {symbol} on {date}: {e}")
            return None
    
    def calculate_cn_special_factors(
        self,
        daily_df: pd.DataFrame,
        intraday_df: Optional[pd.DataFrame] = None,
        date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        计算A股特色因子
        
        包括：
        1. 竞价强度：竞价量占全天比例、竞价涨幅
        2. 量能因子：换手率分位数、量比
        3. 板块动量：近期涨跌幅、振幅
        4. 情绪因子：连续涨跌天数、是否涨停/跌停
        
        Args:
            daily_df: 日线数据
            intraday_df: 分钟线数据（可选，用于竞价数据）
            date: 当前日期（可选，用于计算实时因子）
            
        Returns:
            factors_df: 因子DataFrame
        """
        factors = pd.DataFrame(index=daily_df.index)
        
        # ========== 1. 基础价格因子 ==========
        close = daily_df['close']
        high = daily_df['high']
        low = daily_df['low']
        volume = daily_df['volume']
        
        # 涨跌幅序列
        returns = close.pct_change()
        
        # ========== 2. 量能因子 ==========
        if 'turnover' in daily_df.columns:
            # 换手率分位数（近100天）
            factors['turnover_quantile'] = daily_df['turnover'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
            factors['turnover_ma5'] = daily_df['turnover'].rolling(5).mean()
            factors['turnover_ma20'] = daily_df['turnover'].rolling(20).mean()
        else:
            # 用成交量代替
            factors['turnover_quantile'] = volume.rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
        
        # 量比（今日量/5日均量）
        volume_ma5 = volume.rolling(5).mean()
        factors['volume_ratio'] = volume / (volume_ma5 + 1e-10)
        
        # 成交量变化率
        factors['volume_change_5d'] = volume.pct_change(5)
        factors['volume_change_20d'] = volume.pct_change(20)
        
        # ========== 3. 价格动量因子 ==========
        # 多周期收益率
        for d in [1, 3, 5, 10, 20]:
            factors[f'return_{d}d'] = returns.rolling(d).sum()
        
        # 振幅
        for d in [5, 10, 20]:
            factors[f'amplitude_{d}d'] = ((high - low) / close).rolling(d).mean()
        
        # 价格强度（收盘价在当日范围内的位置）
        factors['price_position'] = (close - low) / (high - low + 1e-10)
        factors['price_position_ma5'] = factors['price_position'].rolling(5).mean()
        
        # ========== 4. 情绪因子 ==========
        # 连续涨跌天数
        factors['consecutive_up'] = (returns > 0).astype(int).groupby(
            (returns <= 0).astype(int).cumsum()
        ).cumsum()
        factors['consecutive_down'] = (returns < 0).astype(int).groupby(
            (returns >= 0).astype(int).cumsum()
        ).cumsum()
        
        # 近期创新高/新低
        factors['is_high_20d'] = (close == close.rolling(20).max()).astype(int)
        factors['is_low_20d'] = (close == close.rolling(20).min()).astype(int)
        
        # ========== 5. 波动率因子 ==========
        # 历史波动率
        for d in [5, 10, 20]:
            factors[f'volatility_{d}d'] = returns.rolling(d).std() * np.sqrt(252)
        
        # 上行/下行波动率
        upside_vol = returns[returns > 0].rolling(20).std()
        downside_vol = returns[returns < 0].rolling(20).std()
        factors['upside_volatility'] = upside_vol.fillna(0)
        factors['downside_volatility'] = downside_vol.fillna(0)
        
        # ========== 6. 竞价因子（需要分钟线数据）==========
        if intraday_df is not None and len(intraday_df) > 0:
            # 对每个日期计算竞价因子
            for idx in daily_df.index:
                try:
                    day_minute_data = intraday_df[
                        intraday_df.index.date == idx.date()
                    ]
                    
                    if len(day_minute_data) > 0:
                        # 竞价量（9:25-9:30）
                        auction_data = day_minute_data.between_time('09:25', '09:30')
                        if len(auction_data) > 0:
                            auction_volume = auction_data['volume'].sum()
                            day_total_volume = day_minute_data['volume'].sum()
                            factors.loc[idx, 'auction_volume_ratio'] = auction_volume / (day_total_volume + 1e-10)
                            
                            # 竞价涨幅
                            if idx in daily_df.index:
                                prev_close = daily_df.loc[:idx]['close'].iloc[-2] if len(daily_df.loc[:idx]) > 1 else None
                                if prev_close is not None:
                                    auction_price = auction_data.iloc[0]['open']
                                    factors.loc[idx, 'auction_return'] = (auction_price - prev_close) / prev_close
                except Exception:
                    continue
        
        # 填充缺失值
        factors = factors.fillna(0)
        
        return factors
    
    def prepare_features(
        self,
        daily_df: pd.DataFrame,
        intraday_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        准备完整特征集
        
        结合 Alpha158 因子 + A股特色因子
        
        Args:
            daily_df: 日线数据
            intraday_df: 分钟线数据（可选）
            
        Returns:
            features_df: 完整特征DataFrame
        """
        # 1. 计算 Alpha158 因子
        alpha158_factors = self.alpha158.calculate(daily_df)
        
        # 2. 计算 A股特色因子
        cn_factors = self.calculate_cn_special_factors(daily_df, intraday_df)
        
        # 3. 合并
        features = pd.concat([alpha158_factors, cn_factors], axis=1)
        
        # 4. 移除无效行（前面因滚动窗口产生的NaN）
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features
    
    def prepare_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        min_periods: int = 60
    ) -> Tuple[pd.DataFrame, pd.Series, List]:
        """
        准备训练数据集
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            min_periods: 最小周期数（用于计算因子）
            
        Returns:
            X: 特征矩阵
            y: 标签（涨幅桶）
            dates: 日期列表
        """
        print(f"\n{'='*60}")
        print(f"📊 Preparing dataset for {symbol}")
        print(f"{'='*60}")
        
        # 1. 获取日线数据（扩展时间范围以确保有足够的历史数据）
        extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print(f"\n📥 Fetching daily data...")
        daily_df = self.dm.fetch_data(
            symbol=symbol,
            start_date=extended_start,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if daily_df is None or len(daily_df) < min_periods:
            raise ValueError(f"Insufficient daily data for {symbol}")
        
        print(f"✅ Daily data: {len(daily_df)} days")
        
        # 2. 获取分钟线数据（用于生成标签和竞价因子）
        print(f"\n📥 Fetching minute data...")
        try:
            minute_df = self.dm.cn_fetcher.fetch_intraday_data(
                symbol=symbol,
                start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                period="1"
            )
            print(f"✅ Minute data: {len(minute_df)} bars")
        except Exception as e:
            print(f"⚠️ Warning: Failed to fetch minute data: {e}")
            minute_df = pd.DataFrame()
        
        # 3. 计算特征
        print(f"\n🔧 Calculating features...")
        features = self.prepare_features(daily_df, minute_df if len(minute_df) > 0 else None)
        print(f"✅ Features: {features.shape[1]} columns")
        
        # 4. 生成标签
        print(f"\n🏷️  Generating labels...")
        labels = []
        valid_dates = []
        
        for date in features.index:
            # 只处理目标时间范围内的数据
            if date < pd.Timestamp(start_date, tz=date.tz):
                continue
            
            if len(minute_df) > 0:
                label = self.generate_label(symbol, date, minute_df)
            else:
                # 如果没有分钟线数据，用次日开盘价作为近似
                try:
                    next_day_idx = daily_df.index.get_loc(date) + 1
                    if next_day_idx < len(daily_df):
                        next_open = daily_df.iloc[next_day_idx]['open']
                        curr_close = daily_df.loc[date, 'close']
                        label = (next_open - curr_close) / curr_close * 100
                    else:
                        label = None
                except Exception:
                    label = None
            
            if label is not None:
                labels.append(label)
                valid_dates.append(date)
        
        print(f"✅ Labels: {len(labels)} samples")
        
        # 5. 对齐特征和标签
        features = features.loc[valid_dates]
        
        # 6. 将连续标签转换为分类桶
        y_continuous = pd.Series(labels, index=valid_dates)
        y_categorical = pd.cut(
            y_continuous,
            bins=self.BINS,
            labels=range(len(self.LABELS))
        ).astype(int)
        
        # 7. 保存特征名称
        self.feature_names_ = features.columns.tolist()
        
        # 打印标签分布
        print(f"\n📊 Label distribution:")
        for i, label in enumerate(self.LABELS):
            count = (y_categorical == i).sum()
            pct = count / len(y_categorical) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        return features, y_categorical, valid_dates
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签（分类桶）
            validation_split: 验证集比例
        """
        print(f"\n{'='*60}")
        print(f"🤖 Training model")
        print(f"{'='*60}")
        
        # 划分训练集和验证集（按时间顺序）
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n📊 Dataset split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        
        # 特征标准化
        print(f"\n⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 训练模型
        print(f"\n🎯 Training {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # 评估
        print(f"\n📈 Evaluation:")
        
        # 训练集
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"   Training Accuracy: {train_acc:.4f}")
        
        # 验证集
        y_val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"   Validation Accuracy: {val_acc:.4f}")
        
        # 详细报告
        print(f"\n📊 Classification Report (Validation):")
        # 只显示实际存在的类别
        unique_classes = sorted(y_val.unique())
        class_labels = [self.LABELS[i] for i in unique_classes]
        print(classification_report(
            y_val,
            y_val_pred,
            labels=unique_classes,
            target_names=class_labels,
            zero_division=0
        ))
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
            
            # 打印 Top 20 特征
            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance': self.feature_importance_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 20 Important Features:")
            print(importance_df.head(20).to_string(index=False))
        
        print(f"\n✅ Training completed!")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            y_pred: 预测类别
            y_proba: 预测概率
        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        return y_pred, y_proba
    
    def backtest(
        self,
        symbol: str,
        X_test: pd.DataFrame,
        dates_test: List,
        daily_df: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        回测策略
        
        策略：每天预测次日高点，如果预测涨幅 > 3%，则开盘买入，30分钟后卖出
        
        Args:
            symbol: 股票代码
            X_test: 测试集特征
            dates_test: 测试集日期
            daily_df: 日线数据（用于获取价格）
            initial_capital: 初始资金
            
        Returns:
            backtest_results: 回测结果字典
        """
        print(f"\n{'='*60}")
        print(f"📊 Backtesting strategy")
        print(f"{'='*60}")
        
        # 预测
        y_pred, y_proba = self.predict(X_test)
        
        # 回测引擎
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=0.0003,  # A股佣金 0.03%
            slippage_rate=0.001,     # 滑点 0.1%
            market='CN'
        )
        
        # 模拟交易
        trades = []
        positions = []
        
        for i, date in enumerate(dates_test):
            pred_class = y_pred[i]
            pred_proba_max = y_proba[i].max()
            
            # 策略：如果预测涨幅 >= 3% (class >= 3) 且置信度 > 0.4，则交易
            if pred_class >= 3 and pred_proba_max > 0.4:
                # 获取次日开盘价
                try:
                    next_day_idx = daily_df.index.get_loc(date) + 1
                    if next_day_idx >= len(daily_df):
                        continue
                    
                    next_day_date = daily_df.index[next_day_idx]
                    open_price = daily_df.iloc[next_day_idx]['open']
                    
                    # 假设30分钟后卖出（简化版：用当日高点的一部分作为卖出价）
                    # 实际应该用分钟线数据
                    high_price = daily_df.iloc[next_day_idx]['high']
                    sell_price = open_price + (high_price - open_price) * 0.5  # 保守估计
                    
                    # 计算收益
                    actual_return = (sell_price - open_price) / open_price
                    
                    # 全仓买入
                    buy_value = engine.cash * 0.95  # 留一点余量
                    
                    # 买入
                    success = engine.buy(symbol, open_price, buy_value, next_day_date)
                    
                    if success:
                        # 卖出
                        engine.sell(symbol, sell_price, next_day_date)
                        
                        trades.append({
                            'date': date,
                            'trade_date': next_day_date,
                            'pred_class': pred_class,
                            'pred_label': self.LABELS[pred_class],
                            'confidence': pred_proba_max,
                            'open_price': open_price,
                            'sell_price': sell_price,
                            'return': actual_return * 100
                        })
                
                except Exception as e:
                    continue
            
            # 更新投资组合
            if date in daily_df.index:
                prices = {symbol: daily_df.loc[date, 'close']}
                engine.update_portfolio(date, prices)
        
        # 统计结果
        print(f"\n📊 Backtest Results:")
        print(f"   Total trades: {len(trades)}")
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            avg_return = trades_df['return'].mean()
            win_rate = (trades_df['return'] > 0).sum() / len(trades_df)
            
            print(f"   Average return per trade: {avg_return:.2f}%")
            print(f"   Win rate: {win_rate:.2%}")
            print(f"\n   Recent trades:")
            print(trades_df.tail(10).to_string(index=False))
        
        # 获取投资组合统计
        stats = engine.get_portfolio_stats()
        
        print(f"\n💰 Portfolio Performance:")
        print(f"   Initial capital: ¥{stats['initial_capital']:,.2f}")
        print(f"   Final value: ¥{stats['final_value']:,.2f}")
        print(f"   Total return: {stats['total_return']:.2%}")
        print(f"   Annual return: {stats['annual_return']:.2%}")
        print(f"   Sharpe ratio: {stats['sharpe_ratio']:.4f}")
        print(f"   Max drawdown: {stats['max_drawdown']:.2%}")
        
        return {
            'trades': trades_df if len(trades) > 0 else pd.DataFrame(),
            'portfolio_stats': stats,
            'engine': engine
        }
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'bins': self.BINS,
            'labels': self.LABELS
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names_ = model_data['feature_names']
        self.feature_importance_ = model_data.get('feature_importance')
        print(f"\n✅ Model loaded from {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("A股次日高点预测策略")
    print("=" * 60)
    print("\n使用方法:")
    print("  from cn_intraday_high_strategy import CNIntradayHighPredictor")
    print("  predictor = CNIntradayHighPredictor()")
    print("  X, y, dates = predictor.prepare_dataset('000858', '2023-01-01', '2024-12-31')")
    print("  predictor.train(X, y)")
    print("\n详见 demo_cn_intraday_high.py")

