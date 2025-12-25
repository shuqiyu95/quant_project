# A股次日高点预测策略 - 快速开始

## 📝 简介

本策略预测A股个股**次日开盘后30分钟内的最高涨幅**，用于短线交易决策。

### 核心特点

✅ **多分类预测**：将涨幅分为5个桶（<-3%, -3%~0%, 0~3%, 3~6%, >6%）  
✅ **丰富的特征**：Alpha158 (158个) + A股特色因子 (80+个)  
✅ **完整的回测**：包含A股交易规则（T+1、佣金、印花税）  
✅ **即开即用**：金风科技演示脚本，一键运行  

---

## 🚀 快速开始

### 1. 运行演示脚本

```bash
cd /Users/yushuqi/Desktop/code/quant_project
python demo_cn_intraday_high.py
```

**默认配置**：
- 股票：金风科技 (002202)
- 时间范围：2022-01-01 至 2024-12-20
- 模型：随机森林（200棵树）
- 初始资金：10万元

### 2. 输出结果

运行完成后会生成：

```
output/
├── cn_intraday_high_002202.pkl  # 训练好的模型
├── trades_002202.csv            # 交易记录
└── portfolio_002202.csv         # 投资组合历史
```

### 3. 查看结果

```python
import pandas as pd

# 查看交易记录
trades = pd.read_csv('output/trades_002202.csv')
print(trades)

# 查看投资组合历史
portfolio = pd.read_csv('output/portfolio_002202.csv')
print(portfolio[['date', 'total_value', 'daily_return']])
```

---

## 🎯 策略逻辑

### 预测目标

预测次日开盘后30分钟内的**最高涨幅**（相对于开盘价）

### 交易规则

```
IF 预测涨幅 >= 3% AND 置信度 > 0.4:
    次日开盘价买入（全仓）
    30分钟后卖出
```

### 适用场景

- ✅ 短线交易参考
- ✅ 日内波动预测
- ⚠️ 需要T+0交易能力（融券等）

---

## 🔧 自定义使用

### 更换股票

编辑 `demo_cn_intraday_high.py`：

```python
SYMBOL = '600519'      # 贵州茅台
STOCK_NAME = '贵州茅台'
START_DATE = '2023-01-01'
END_DATE = '2024-12-20'
```

常用股票代码：
- `600519`: 贵州茅台
- `000858`: 五粮液
- `600036`: 招商银行
- `300750`: 宁德时代

### 调整模型参数

```python
predictor = CNIntradayHighPredictor(
    model_type='gbdt',  # 使用GBDT替代随机森林
    model_params={
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.03
    }
)
```

### Python API使用

```python
from cn_intraday_high_strategy import CNIntradayHighPredictor

# 1. 初始化
predictor = CNIntradayHighPredictor()

# 2. 准备数据
X, y, dates = predictor.prepare_dataset(
    symbol='002202',
    start_date='2022-01-01',
    end_date='2024-12-20'
)

# 3. 训练模型
predictor.train(X, y, validation_split=0.2)

# 4. 保存模型
predictor.save('my_model.pkl')

# 5. 回测（可选）
backtest_results = predictor.backtest(
    symbol='002202',
    X_test=X_test,
    dates_test=dates_test,
    daily_df=daily_df
)
```

---

## 📊 特征工程

### Alpha158 因子（158个）

标准的技术指标因子：
- 价格动量（ROC, MA, STD等）
- 技术指标（RSI, MACD, KDJ等）
- 量价关系（CORR, Beta, Rsquare等）

### A股特色因子（80+个）

#### 1. 量能因子
- `turnover_quantile`: 换手率在近100天的分位数
- `volume_ratio`: 今日成交量 / 5日均量
- `volume_change_5d/20d`: 成交量变化率

#### 2. 竞价因子（需分钟线数据）
- `auction_volume_ratio`: 竞价量占全天比例
- `auction_return`: 竞价涨幅

#### 3. 情绪因子
- `consecutive_up/down`: 连续涨跌天数
- `is_high_20d/low_20d`: 是否创新高/新低

#### 4. 波动率因子
- `volatility_5d/10d/20d`: 历史波动率
- `upside/downside_volatility`: 上下行波动率

**共239个特征**（Alpha158 + A股特色因子）

---

## 📈 回测指标

运行完成后会显示：

```
💰 Portfolio Performance:
   Initial Capital: ¥100,000.00
   Final Value: ¥XXX,XXX.XX
   Total Return: X.XX%
   Annual Return: X.XX%
   Sharpe Ratio: X.XXXX
   Max Drawdown: -X.XX%
   Win Rate: XX.X%
   Total Trades: XX
```

### 指标说明

- **Total Return**: 总收益率
- **Annual Return**: 年化收益率
- **Sharpe Ratio**: 夏普比率（风险调整后收益）
  - \> 1.0: 优秀
  - 0.5~1.0: 良好
  - < 0.5: 一般
- **Max Drawdown**: 最大回撤
- **Win Rate**: 胜率

---

## 🎨 演示输出示例

```
================================================================================
🚀 A股次日高点预测策略 - 金风科技演示
================================================================================

📋 Configuration:
   Stock: 金风科技 (002202)
   Date range: 2022-01-01 to 2024-12-20
   Model: random_forest
   Initial capital: ¥100,000.00

============================================================
📊 Preparing dataset for 002202
============================================================

✅ Daily data: 962 days
✅ Minute data: 0 bars
✅ Features: 239 columns
✅ Labels: 718 samples

📊 Label distribution:
   <-3%: 4 (0.6%)
   -3%~0%: 467 (65.0%)
   0%~3%: 242 (33.7%)
   3%~6%: 4 (0.6%)
   >6%: 1 (0.1%)

============================================================
🤖 Training model
============================================================

   Training Accuracy: 0.9686
   Validation Accuracy: 0.6250

🔝 Top 20 Important Features:
          feature  importance
  TURNOVER_std_20    0.027848
   price_position    0.027491
    VOLUME_std_20    0.027484
          RSQR_60    0.022651
           KURT_5    0.019486
          ...

✅ Training completed!
```

---

## ⚠️ 重要提示

### 数据限制

1. **分钟线数据**：
   - AkShare的分钟线数据可能不稳定
   - 没有分钟线时，用次日开盘价近似，误差较大
   - 建议使用付费数据源（Wind、同花顺等）

2. **样本不平衡**：
   - 高涨幅（>3%）样本较少
   - 模型可能倾向预测中等涨幅
   - 可以使用SMOTE等方法改善

### 交易限制

1. **T+1制度**：策略假设次日开盘买入、30分钟后卖出
   - 需要融券或期权等工具
   - 或仅作为交易参考

2. **涨跌停**：涨停无法买入，跌停无法卖出

3. **流动性**：小盘股可能难以快速成交

---

## 📚 详细文档

更多详细信息请查看：

- **策略详解**: `docs/CN_INTRADAY_HIGH_STRATEGY.md`
- **更新日志**: `docs/CHANGELOG.md`
- **待办事项**: `docs/TODO.md`

---

## 🔗 快速链接

| 文件 | 说明 |
|------|------|
| `cn_intraday_high_strategy.py` | 策略核心代码 |
| `demo_cn_intraday_high.py` | 演示脚本 |
| `docs/CN_INTRADAY_HIGH_STRATEGY.md` | 详细文档 |
| `output/` | 输出目录 |

---

## 💡 改进建议

### 提升预测准确率

1. **获取分钟线数据**：准确计算30分钟最高涨幅
2. **增加样本**：更长的历史数据（3-5年）
3. **特征优化**：加入行业、市场、资金流向等因子
4. **模型调优**：GridSearch优化超参数

### 策略优化

1. **动态仓位**：根据置信度调整仓位大小
2. **止损止盈**：设置合理的止损和止盈点
3. **多股票池**：同时预测多只股票，选择最优的
4. **时间过滤**：避开重大公告、业绩发布日

---

## 📞 联系方式

如有问题或建议，欢迎反馈！

**最后更新**: 2025-12-25  
**版本**: v1.3.0  
**维护者**: Quant Team

