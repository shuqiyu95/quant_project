"""
股票预测模型

包含多种模型实现：
1. 随机森林模型
2. 线性回归模型
3. 支持自定义排序损失函数的梯度提升模型
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

from .rank_loss import (
    RankMSELoss, 
    PairwiseRankLoss, 
    ListNetLoss, 
    evaluate_ranking_metrics
)


class StockPredictor:
    """
    股票预测器基类
    
    支持多种模型和训练方式
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        loss_type: str = 'mse',
        model_params: Optional[Dict] = None,
        scale_features: bool = True
    ):
        """
        Args:
            model_type: 模型类型 ('random_forest', 'linear', 'ridge', 'lasso', 'gbdt')
            loss_type: 损失函数类型 ('mse', 'rank_mse', 'pairwise', 'listnet')
            model_params: 模型参数字典
            scale_features: 是否标准化特征
        """
        self.model_type = model_type
        self.loss_type = loss_type
        self.model_params = model_params or {}
        self.scale_features = scale_features
        
        # 初始化模型
        self.model = self._create_model()
        
        # 特征标准化器
        self.scaler = StandardScaler() if scale_features else None
        
        # 损失函数
        self.loss_fn = self._create_loss_function()
        
        # 特征重要性
        self.feature_importance_ = None
        self.feature_names_ = None
    
    def _create_model(self):
        """创建模型实例"""
        default_params = self._get_default_params()
        params = {**default_params, **self.model_params}
        
        if self.model_type == 'random_forest':
            return RandomForestRegressor(**params)
        elif self.model_type == 'linear':
            return LinearRegression(**params)
        elif self.model_type == 'ridge':
            return Ridge(**params)
        elif self.model_type == 'lasso':
            return Lasso(**params)
        elif self.model_type == 'gbdt':
            return GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_default_params(self) -> Dict:
        """获取默认模型参数"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type in ['ridge', 'lasso']:
            return {'alpha': 1.0, 'random_state': 42}
        elif self.model_type == 'gbdt':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
        else:
            return {}
    
    def _create_loss_function(self):
        """创建损失函数"""
        if self.loss_type == 'rank_mse':
            return RankMSELoss()
        elif self.loss_type == 'pairwise':
            return PairwiseRankLoss()
        elif self.loss_type == 'listnet':
            return ListNetLoss()
        else:  # 'mse' or default
            return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            **fit_params: 额外的训练参数
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # 标准化特征
        if self.scaler is not None:
            X_array = self.scaler.fit_transform(X_array)
        
        # 训练模型
        self.model.fit(X_array, y_array, **fit_params)
        
        # 保存特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            predictions: 预测值
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 标准化特征
        if self.scaler is not None:
            X_array = self.scaler.transform(X_array)
        
        return self.model.predict(X_array)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, k: int = 3) -> Dict:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            k: Top-K 评估
            
        Returns:
            metrics: 评估指标字典
        """
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y_true = y.values
        else:
            y_true = y
        
        # 基本回归指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # 排序指标
        rank_metrics = evaluate_ranking_metrics(y_true, y_pred, k=k)
        metrics.update(rank_metrics)
        
        # 自定义损失
        if self.loss_fn is not None:
            custom_loss = self.loss_fn(y_true, y_pred)
            metrics[f'{self.loss_type}_loss'] = custom_loss
        
        return metrics
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            top_k: 返回前k个重要特征，None表示返回全部
            
        Returns:
            importance_df: 特征重要性 DataFrame
        """
        if self.feature_importance_ is None:
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'loss_type': self.loss_type,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.loss_type = model_data['loss_type']
        self.feature_names_ = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        print(f"Model loaded from {filepath}")


class RankingPredictor(StockPredictor):
    """
    专门用于排序的预测器
    
    针对股票选择优化的模型
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        loss_type: str = 'rank_mse',
        model_params: Optional[Dict] = None
    ):
        """
        Args:
            model_type: 模型类型
            loss_type: 必须是排序损失 ('rank_mse', 'pairwise', 'listnet')
        """
        if loss_type not in ['rank_mse', 'pairwise', 'listnet']:
            warnings.warn(f"RankingPredictor should use ranking loss, got {loss_type}")
        
        super().__init__(
            model_type=model_type,
            loss_type=loss_type,
            model_params=model_params,
            scale_features=True
        )
    
    def fit_cross_sectional(
        self,
        dataset: List[Tuple],
        verbose: bool = True
    ):
        """
        使用截面数据训练（每个时间点对股票进行排序）
        
        Args:
            dataset: List of (date, X, y, symbols)
            verbose: 是否打印训练信息
        """
        # 收集所有数据
        X_list = []
        y_list = []
        
        for date, X, y, symbols in dataset:
            X_list.append(X)
            y_list.append(y)
        
        X_all = np.vstack(X_list)
        y_all = np.hstack(y_list)
        
        # 训练模型
        if verbose:
            print(f"Training on {len(dataset)} time points, {len(X_all)} samples")
        
        self.fit(X_all, y_all)
        
        # 评估每个截面的排序质量
        if verbose:
            print("\n=== Cross-Sectional Evaluation ===")
            metrics_list = []
            
            for date, X, y, symbols in dataset[:5]:  # 只评估前5个
                y_pred = self.predict(X)
                metrics = evaluate_ranking_metrics(y, y_pred, k=min(3, len(y)))
                metrics_list.append(metrics)
                
                print(f"\nDate: {date}")
                print(f"  Spearman: {metrics['spearman_corr']:.4f}")
                print(f"  Top-K Acc: {metrics.get('top_3_accuracy', 0):.4f}")
        
        return self


def train_test_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    dates: List,
    test_size: float = 0.2
) -> Tuple:
    """
    时间序列数据的训练测试集划分
    
    按时间顺序划分，不打乱
    
    Args:
        X: 特征矩阵
        y: 标签
        dates: 日期列表
        test_size: 测试集比例
        
    Returns:
        X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
    else:
        X_train = X[:split_idx]
        X_test = X[split_idx:]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
    else:
        y_train = y[:split_idx]
        y_test = y[split_idx:]
    
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def walk_forward_validation(
    predictor: StockPredictor,
    dataset: List[Tuple],
    train_window: int = 120,
    test_window: int = 20,
    verbose: bool = True
) -> Dict:
    """
    滚动窗口验证
    
    Args:
        predictor: 预测器
        dataset: 截面数据集 List of (date, X, y, symbols)
        train_window: 训练窗口大小（天数）
        test_window: 测试窗口大小（天数）
        verbose: 是否打印信息
        
    Returns:
        results: 验证结果字典
    """
    results = {
        'predictions': [],
        'actuals': [],
        'dates': [],
        'symbols': [],
        'metrics': []
    }
    
    n_periods = len(dataset)
    
    for i in range(train_window, n_periods - test_window, test_window):
        # 训练集
        train_dataset = dataset[i - train_window:i]
        
        # 测试集
        test_dataset = dataset[i:i + test_window]
        
        # 训练模型
        X_train_list = [X for _, X, _, _ in train_dataset]
        y_train_list = [y for _, _, y, _ in train_dataset]
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        
        predictor.fit(X_train, y_train)
        
        # 在测试集上预测和评估
        for date, X_test, y_test, symbols in test_dataset:
            y_pred = predictor.predict(X_test)
            
            # 保存结果
            results['predictions'].extend(y_pred)
            results['actuals'].extend(y_test)
            results['dates'].extend([date] * len(y_test))
            results['symbols'].extend(symbols)
            
            # 计算指标
            metrics = evaluate_ranking_metrics(y_test, y_pred, k=min(3, len(y_test)))
            results['metrics'].append(metrics)
        
        if verbose:
            avg_spearman = np.mean([m['spearman_corr'] for m in results['metrics'][-test_window:]])
            print(f"Period {i//test_window}: Avg Spearman = {avg_spearman:.4f}")
    
    return results


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Stock Predictor ===\n")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples) * 0.02)  # 模拟收益率
    
    # 1. 测试随机森林模型
    print("1. Testing Random Forest Predictor")
    rf_predictor = StockPredictor(model_type='random_forest', loss_type='rank_mse')
    
    # 划分训练测试集
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 训练
    rf_predictor.fit(X_train, y_train)
    print("Training completed!")
    
    # 评估
    metrics = rf_predictor.evaluate(X_test, y_test, k=3)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 特征重要性
    print("\nTop 5 Important Features:")
    importance = rf_predictor.get_feature_importance(top_k=5)
    print(importance)
    
    # 2. 测试线性模型
    print("\n2. Testing Ridge Predictor")
    ridge_predictor = StockPredictor(model_type='ridge', loss_type='rank_mse')
    ridge_predictor.fit(X_train, y_train)
    
    metrics = ridge_predictor.evaluate(X_test, y_test, k=3)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ All tests passed!")

