"""
测试模型模块

测试：
- RankLoss 函数
- FeatureEngineer
- StockPredictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.models.rank_loss import (
    RankMSELoss,
    PairwiseRankLoss,
    ListNetLoss,
    BinaryClassificationLoss,
    evaluate_ranking_metrics
)
from src.models.feature_engineering import FeatureEngineer, create_weekly_trading_dates
from src.models.predictor import StockPredictor, train_test_split_time_series


class TestRankLoss:
    """测试 RankLoss 函数"""
    
    def test_rank_mse_loss(self):
        """测试 RankMSE"""
        y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01])
        y_pred = np.array([0.04, 0.02, 0.07, -0.01, 0.02])
        
        loss_fn = RankMSELoss()
        loss = loss_fn(y_true, y_pred)
        
        assert loss >= 0, "Loss should be non-negative"
        assert isinstance(loss, float), "Loss should be float"
        print(f"✅ RankMSE Loss: {loss:.4f}")
    
    def test_pairwise_rank_loss(self):
        """测试 Pairwise Loss"""
        y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01])
        y_pred = np.array([0.04, 0.02, 0.07, -0.01, 0.02])
        
        loss_fn = PairwiseRankLoss(margin=0.01)
        loss = loss_fn(y_true, y_pred)
        
        assert loss >= 0, "Loss should be non-negative"
        print(f"✅ Pairwise Loss: {loss:.4f}")
    
    def test_listnet_loss(self):
        """测试 ListNet Loss"""
        y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01])
        y_pred = np.array([0.04, 0.02, 0.07, -0.01, 0.02])
        
        loss_fn = ListNetLoss()
        loss = loss_fn(y_true, y_pred)
        
        assert loss >= 0, "Loss should be non-negative"
        print(f"✅ ListNet Loss: {loss:.4f}")
    
    def test_binary_classification_loss(self):
        """测试二分类 Loss"""
        y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01])
        y_pred_proba = np.array([0.3, 0.2, 0.8, 0.1, 0.4])
        
        loss_fn = BinaryClassificationLoss(top_k=1)
        loss = loss_fn(y_true, y_pred_proba)
        
        assert loss >= 0, "Loss should be non-negative"
        print(f"✅ Binary Classification Loss: {loss:.4f}")
    
    def test_ranking_metrics(self):
        """测试排序指标"""
        y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01, 0.06, -0.01])
        y_pred = np.array([0.04, 0.02, 0.07, -0.01, 0.02, 0.05, 0.00])
        
        metrics = evaluate_ranking_metrics(y_true, y_pred, k=3)
        
        assert 'spearman_corr' in metrics
        assert 'kendall_tau' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'ndcg@3' in metrics
        
        print("✅ Ranking Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")


class TestFeatureEngineer:
    """测试特征工程"""
    
    def setup_method(self):
        """准备测试数据"""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        n = len(dates)
        
        self.df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'high': 100 + np.cumsum(np.random.randn(n) * 2) + 2,
            'low': 100 + np.cumsum(np.random.randn(n) * 2) - 2,
            'volume': 1000000 + np.random.randn(n) * 100000
        }, index=dates)
    
    def test_create_features(self):
        """测试特征创建"""
        fe = FeatureEngineer()
        features = fe.create_features(self.df)
        
        assert len(features) == len(self.df), "Features length should match data"
        assert features.shape[1] > 0, "Should have features"
        
        print(f"✅ Created {features.shape[1]} features")
        print(f"   Shape: {features.shape}")
    
    def test_create_labels(self):
        """测试标签创建"""
        fe = FeatureEngineer()
        labels = fe.create_labels(self.df, forward_days=5)
        
        assert len(labels) == len(self.df), "Labels length should match data"
        
        # 检查最后5天的标签是NaN（因为没有未来数据）
        assert labels.iloc[-5:].isna().all(), "Last 5 labels should be NaN"
        
        print(f"✅ Created labels with {labels.notna().sum()} valid values")
    
    def test_prepare_dataset(self):
        """测试数据集准备"""
        fe = FeatureEngineer()
        
        data_dict = {
            'AAPL': self.df.copy(),
            'MSFT': self.df.copy() * 1.1,
            'GOOGL': self.df.copy() * 0.9
        }
        
        X, y, dates, symbols = fe.prepare_dataset(data_dict, forward_days=5)
        
        assert X.shape[0] == y.shape[0], "X and y should have same length"
        assert len(dates) == len(X), "Dates should match X length"
        assert len(symbols) == len(X), "Symbols should match X length"
        assert set(symbols) == set(data_dict.keys()), "Should contain all symbols"
        
        print(f"✅ Prepared dataset:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Unique symbols: {len(set(symbols))}")
    
    def test_cross_sectional_dataset(self):
        """测试截面数据集"""
        fe = FeatureEngineer()
        
        data_dict = {
            'AAPL': self.df.copy(),
            'MSFT': self.df.copy() * 1.1,
            'GOOGL': self.df.copy() * 0.9
        }
        
        dataset = fe.create_cross_sectional_dataset(data_dict, forward_days=5)
        
        assert len(dataset) > 0, "Should have time points"
        
        date, X, y, symbols = dataset[0]
        assert X.shape[0] == len(y), "X and y should match"
        assert len(symbols) == len(y), "Symbols should match y"
        
        print(f"✅ Cross-sectional dataset:")
        print(f"   Time points: {len(dataset)}")
        print(f"   First date: {date}")
        print(f"   Stocks per date: {len(symbols)}")


class TestStockPredictor:
    """测试预测模型"""
    
    def setup_method(self):
        """准备测试数据"""
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y = pd.Series(np.random.randn(n_samples) * 0.02)
        
        # 划分训练测试集
        split_idx = int(n_samples * 0.8)
        self.X_train = self.X.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_train = self.y.iloc[:split_idx]
        self.y_test = self.y.iloc[split_idx:]
    
    def test_random_forest_predictor(self):
        """测试随机森林"""
        predictor = StockPredictor(
            model_type='random_forest',
            loss_type='rank_mse',
            model_params={'n_estimators': 50, 'max_depth': 5}
        )
        
        predictor.fit(self.X_train, self.y_train)
        y_pred = predictor.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test), "Predictions should match test size"
        
        metrics = predictor.evaluate(self.X_test, self.y_test, k=3)
        
        print("✅ Random Forest Predictor:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
    
    def test_linear_predictor(self):
        """测试线性回归"""
        predictor = StockPredictor(
            model_type='ridge',
            loss_type='rank_mse'
        )
        
        predictor.fit(self.X_train, self.y_train)
        y_pred = predictor.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test), "Predictions should match test size"
        
        metrics = predictor.evaluate(self.X_test, self.y_test, k=3)
        
        print("✅ Ridge Predictor:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
    
    def test_feature_importance(self):
        """测试特征重要性"""
        predictor = StockPredictor(
            model_type='random_forest',
            model_params={'n_estimators': 50}
        )
        
        predictor.fit(self.X_train, self.y_train)
        importance = predictor.get_feature_importance(top_k=5)
        
        assert len(importance) == 5, "Should return top 5 features"
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        
        print("✅ Feature Importance (Top 5):")
        print(importance)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TESTING MODELS MODULE".center(60))
    print("=" * 60)
    
    # Test RankLoss
    print("\n--- Testing RankLoss ---")
    test_rank = TestRankLoss()
    test_rank.test_rank_mse_loss()
    test_rank.test_pairwise_rank_loss()
    test_rank.test_listnet_loss()
    test_rank.test_binary_classification_loss()
    test_rank.test_ranking_metrics()
    
    # Test FeatureEngineer
    print("\n--- Testing FeatureEngineer ---")
    test_fe = TestFeatureEngineer()
    test_fe.setup_method()
    test_fe.test_create_features()
    test_fe.test_create_labels()
    test_fe.test_prepare_dataset()
    test_fe.test_cross_sectional_dataset()
    
    # Test StockPredictor
    print("\n--- Testing StockPredictor ---")
    test_pred = TestStockPredictor()
    test_pred.setup_method()
    test_pred.test_random_forest_predictor()
    test_pred.test_linear_predictor()
    test_pred.test_feature_importance()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!".center(60))
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

