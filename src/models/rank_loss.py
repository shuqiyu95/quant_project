"""
Ranking Loss Functions for Stock Selection

实现多种排序损失函数：
1. RankMSE: 基于排名的均方误差
2. ListNet: 基于概率分布的排序损失
3. PairwiseRankLoss: 成对比较的排序损失
"""

import numpy as np
from scipy.stats import rankdata
from typing import Optional


class RankMSELoss:
    """
    RankMSE Loss - 排名均方误差损失
    
    计算预测值排名与真实值排名之间的MSE
    适用于关注相对排序而非绝对数值的场景
    
    公式: Loss = mean((rank_pred - rank_true)^2)
    """
    
    def __init__(self, method='average'):
        """
        Args:
            method: 排名方法 ('average', 'min', 'max', 'dense', 'ordinal')
                   'average': 相同值的平均排名
                   'min': 相同值的最小排名
                   'ordinal': 按出现顺序排名
        """
        self.method = method
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算 RankMSE 损失
        
        Args:
            y_true: 真实值 (n_samples,)
            y_pred: 预测值 (n_samples,)
            
        Returns:
            loss: RankMSE 损失值
        """
        # 计算排名（值越大排名越高）
        rank_true = rankdata(-y_true, method=self.method)
        rank_pred = rankdata(-y_pred, method=self.method)
        
        # 计算 MSE
        loss = np.mean((rank_pred - rank_true) ** 2)
        return loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        计算梯度（用于自定义优化）
        
        注意：rankdata 不可微，这里返回近似梯度
        """
        rank_true = rankdata(-y_true, method=self.method)
        rank_pred = rankdata(-y_pred, method=self.method)
        
        # 近似梯度：使用预测值的顺序
        n = len(y_pred)
        order = np.argsort(-y_pred)
        grad = np.zeros(n)
        
        for i in range(n):
            idx = order[i]
            grad[idx] = 2 * (rank_pred[idx] - rank_true[idx]) / n
        
        return grad


class PairwiseRankLoss:
    """
    Pairwise Ranking Loss - 成对排序损失
    
    对所有样本对进行比较，惩罚排序错误的对
    类似于 LambdaRank 的思想
    
    公式: Loss = sum(max(0, margin - (y_pred_i - y_pred_j)) * I(y_true_i > y_true_j))
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: 边界值，要求预测差异至少大于 margin
        """
        self.margin = margin
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算成对排序损失
        
        Args:
            y_true: 真实值 (n_samples,)
            y_pred: 预测值 (n_samples,)
            
        Returns:
            loss: Pairwise ranking loss
        """
        n = len(y_true)
        loss = 0.0
        count = 0
        
        # 遍历所有样本对
        for i in range(n):
            for j in range(i + 1, n):
                # 如果 i 应该排在 j 前面（真实值更大）
                if y_true[i] > y_true[j]:
                    # 计算违反的程度
                    violation = self.margin - (y_pred[i] - y_pred[j])
                    loss += max(0, violation)
                    count += 1
                # 如果 j 应该排在 i 前面
                elif y_true[j] > y_true[i]:
                    violation = self.margin - (y_pred[j] - y_pred[i])
                    loss += max(0, violation)
                    count += 1
        
        # 返回平均损失
        return loss / count if count > 0 else 0.0


class ListNetLoss:
    """
    ListNet Loss - 基于概率分布的排序损失
    
    将排序问题转化为概率分布匹配问题
    使用 Plackett-Luce 模型计算概率分布
    
    公式: Loss = -sum(P_true * log(P_pred))
         其中 P(i) = exp(score_i) / sum(exp(score_j))
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: 温度参数，控制分布的平滑程度
        """
        self.temperature = temperature
    
    def _to_probability(self, scores: np.ndarray) -> np.ndarray:
        """
        将分数转换为概率分布 (Plackett-Luce model)
        
        Args:
            scores: 分数数组
            
        Returns:
            probs: 概率分布
        """
        # 归一化防止数值溢出
        scores = scores / self.temperature
        scores = scores - np.max(scores)
        
        # 计算 softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        return probs
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算 ListNet 交叉熵损失
        
        Args:
            y_true: 真实值 (n_samples,)
            y_pred: 预测值 (n_samples,)
            
        Returns:
            loss: ListNet cross-entropy loss
        """
        # 转换为概率分布
        p_true = self._to_probability(y_true)
        p_pred = self._to_probability(y_pred)
        
        # 计算交叉熵（添加小值避免 log(0)）
        epsilon = 1e-10
        loss = -np.sum(p_true * np.log(p_pred + epsilon))
        
        return loss


class BinaryClassificationLoss:
    """
    简化版：二分类损失函数
    
    将排序问题简化为分类问题：预测哪只股票是"赢家"
    适用于 top-k 选股场景
    """
    
    def __init__(self, top_k: int = 1):
        """
        Args:
            top_k: 前 k 只股票标记为正类
        """
        self.top_k = top_k
    
    def create_labels(self, y_true: np.ndarray) -> np.ndarray:
        """
        创建二分类标签
        
        Args:
            y_true: 真实收益率
            
        Returns:
            labels: 二分类标签 (1: 赢家, 0: 其他)
        """
        n = len(y_true)
        labels = np.zeros(n)
        
        # 找到 top-k 的索引
        top_k_indices = np.argsort(-y_true)[:self.top_k]
        labels[top_k_indices] = 1
        
        return labels
    
    def __call__(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        Args:
            y_true: 真实收益率
            y_pred_proba: 预测为正类的概率
            
        Returns:
            loss: Binary cross-entropy loss
        """
        labels = self.create_labels(y_true)
        
        # 二分类交叉熵
        epsilon = 1e-10
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            labels * np.log(y_pred_proba) + 
            (1 - labels) * np.log(1 - y_pred_proba)
        )
        
        return loss


def evaluate_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> dict:
    """
    评估排序质量的多个指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        k: Top-K 评估
        
    Returns:
        metrics: 包含多个评估指标的字典
    """
    n = len(y_true)
    
    # 1. Spearman 秩相关系数
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    # 2. Kendall's Tau
    from scipy.stats import kendalltau
    kendall_tau, _ = kendalltau(y_true, y_pred)
    
    # 3. Top-K 准确率（预测的前K与真实的前K的重叠）
    true_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    top_k_accuracy = len(true_top_k & pred_top_k) / k
    
    # 4. NDCG (Normalized Discounted Cumulative Gain)
    # NDCG 需要非负值，对于股票收益率（可能为负），我们转换为排名分数
    from sklearn.metrics import ndcg_score
    try:
        # 将值转换为非负排名分数 (值越大，分数越高)
        # 使用 min-max 归一化到 [0, 1]
        if y_true.min() < 0 or y_pred.min() < 0:
            y_true_normalized = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-10)
            y_pred_normalized = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
            ndcg = ndcg_score([y_true_normalized], [y_pred_normalized], k=k)
        else:
            ndcg = ndcg_score([y_true], [y_pred], k=k)
    except Exception as e:
        # 如果 NDCG 计算失败，设为 NaN
        ndcg = np.nan
    
    # 5. RankMSE
    rank_mse = RankMSELoss()(y_true, y_pred)
    
    return {
        'spearman_corr': spearman_corr,
        'kendall_tau': kendall_tau,
        f'top_{k}_accuracy': top_k_accuracy,
        f'ndcg@{k}': ndcg,
        'rank_mse': rank_mse
    }


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Ranking Loss Functions ===\n")
    
    # 模拟数据：7支股票的真实收益和预测收益
    np.random.seed(42)
    y_true = np.array([0.05, 0.03, 0.08, -0.02, 0.01, 0.06, -0.01])  # 真实收益
    y_pred = np.array([0.04, 0.02, 0.07, -0.01, 0.02, 0.05, 0.00])   # 预测收益
    
    print("True returns:", y_true)
    print("Pred returns:", y_pred)
    print()
    
    # 1. RankMSE Loss
    rank_mse_loss = RankMSELoss()
    loss1 = rank_mse_loss(y_true, y_pred)
    print(f"RankMSE Loss: {loss1:.4f}")
    
    # 2. Pairwise Rank Loss
    pairwise_loss = PairwiseRankLoss(margin=0.01)
    loss2 = pairwise_loss(y_true, y_pred)
    print(f"Pairwise Rank Loss: {loss2:.4f}")
    
    # 3. ListNet Loss
    listnet_loss = ListNetLoss()
    loss3 = listnet_loss(y_true, y_pred)
    print(f"ListNet Loss: {loss3:.4f}")
    
    # 4. 评估指标
    print("\n=== Ranking Metrics ===")
    metrics = evaluate_ranking_metrics(y_true, y_pred, k=3)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

