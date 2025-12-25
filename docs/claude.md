# 🤖 Quant Project - 开发指南

## 📋 项目简介

本项目是一个支持 **A股 (CN)** 与 **美股 (US)** 的跨市场量化分析框架，具备从日线到分钟线的多尺度数据处理能力。

### 核心目标
- **美股策略**: Mag7（七巨头）每周轮动择股策略
- **A股策略**: 特征工程驱动的个股次日开盘预测

---

## 🏗️ 项目架构

### 目录结构

```
quant_project/
├── .cursorrules                    # 🎯 开发规范（重要！）
├── docs/                           # 📚 文档中心
│   ├── claude.md                   # 本开发指南
│   ├── CHANGELOG.md                # 版本更新日志
│   ├── TODO.md                     # 待实现功能（增量需求）
│   ├── QUICKSTART*.md              # 快速开始文档
│   └── ...                         # 其他详细文档
├── src/
│   ├── data_engine/                # 数据获取与管理
│   ├── factors/                    # 因子计算层
│   ├── models/                     # 机器学习模型
│   ├── backtester/                 # 回测引擎
│   └── utils/                      # 工具函数
├── data/                           # 本地数据缓存
├── test/                           # 测试套件
└── output/                         # 输出结果
```

### 模块状态

| 模块 | 状态 | 职责 |
|------|------|------|
| **data_engine** | ✅ 完成 | 数据获取、缓存、增量更新 |
| **factors** | ✅ 完成 | 因子计算、Alpha158/360 |
| **models** | ✅ 完成 | RankLoss、特征工程、预测 |
| **backtester** | ✅ 完成 | 回测引擎、策略、性能分析 |
| **utils** | 🔧 基础 | 日志、时区、交易日历等 |

---

## 🚀 快速开始

### 1. 环境搭建

```bash
cd /Users/yushuqi/Desktop/code/quant_project
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 运行 Mag7 策略
python main_mag7_strategy.py

# 运行测试
pytest test/ -v
```
---

## 📚 完整文档导航

### 快速开始（5分钟上手）
- [项目总览](../README.md)
- [项目快速开始](QUICKSTART.md)
- [Mag7 策略快速开始](QUICKSTART_MAG7.md)
- [因子库快速开始](QUICKSTART_FACTORS.md)
- [A股数据快速开始](CN_DATA_QUICKSTART.md)

### 详细文档
- [因子库使用指南](factors_guide.md) - 算子、Alpha158、Alpha360
- [数据集管理指南](DATASET_USAGE.md) - 保存/加载数据集
- [A股数据完整文档](CN_DATA_MODULE.md) - A股数据全部功能
- [A股数据实现总结](CN_DATA_IMPLEMENTATION.md) - 技术实现细节

### 版本管理
- [更新日志](CHANGELOG.md) - 版本历史和变更记录
- [待实现功能](TODO.md) - 计划中的功能和改进

---

## 🎯 开发规范（重要！）

### ⚠️ 开发前必读

**所有开发规范、代码风格、架构原则请查看：**
👉 [`.cursorrules`](../.cursorrules) 文件

### 📝 增量更新流程

**当你需要添加新功能或改进时：**

1. **记录需求** 📝
   - 在 `docs/TODO.md` 中记录待实现功能
   - 格式：`- [ ] 功能描述 (优先级: 高/中/低)`

2. **开发功能** 💻
   - 遵循 `.cursorrules` 中的规范
   - 在对应模块实现功能
   - 编写测试用例

3. **更新文档** 📚
   - 更新 `CHANGELOG.md` 记录变更
   - 重要功能更新 `README.md`

4. **代码审查** ✅
   - 运行 `pytest test/ -v` 确保测试通过
   - 检查 `.cursorrules` 中的审查清单

**示例 - 添加新因子库：**
```bash
# 1. 记录到 TODO.md
echo "- [ ] 实现 Alpha101 因子库 (优先级: 中)" >> docs/TODO.md

# 2. 实现功能
# 在 src/factors/alpha101.py 实现

# 3. 编写测试
# 在 test/test_factors.py 添加测试

# 4. 更新文档
# 更新 CHANGELOG.md 和 factors_guide.md

# 5. 提交
git add .
git commit -m "feat: 添加 Alpha101 因子库"
```

---

## 🎯 开发路线图

### 已完成 ✅
- [x] 数据引擎（美股 + A股）
- [x] 因子计算库（Alpha158 + Alpha360）
- [x] RankLoss 函数
- [x] Mag7 每周轮动策略
- [x] 回测引擎
- [x] 数据集管理功能
- [x] 完整测试套件
- [x] 文档体系

### 进行中 🔧
- [ ] 可视化模块（见 `docs/TODO.md`）

### 计划中 📋
- [ ] A股次日高点预测策略
- [ ] 实时交易接口对接
- [ ] 策略组合管理
- [ ] 风险控制模块

详细规划见 `docs/TODO.md`

---

## 📖 关键概念速查

### 数据格式
所有数据统一为 timezone-aware DataFrame，包含：
- 美股：US/Eastern 时区
- A股：Asia/Shanghai 时区
- 列：open, high, low, close, volume, symbol, market

### 因子命名规范
```python
'MA_5'              # 5日移动平均
'RSI_14'            # 14日RSI
'MACD_12_26_9'      # MACD指标
```

### 回测规则
- **美股**：T+0，佣金0.1%，滑点0.05%
- **A股**：T+1，佣金0.03%，印花税0.05%（卖出）

详见 `.cursorrules` 文件。

---

## 🤝 贡献指南

1. 阅读 `.cursorrules` 了解开发规范
2. 在 `docs/TODO.md` 中记录需求
3. 遵循测试驱动开发（TDD）
4. 更新相关文档
5. 提交 Pull Request

---

## 🔗 快速链接

| 需求 | 文档 |
|------|------|
| 5分钟快速上手 | [QUICKSTART.md](QUICKSTART.md) |
| 运行 Mag7 策略 | [QUICKSTART_MAG7.md](QUICKSTART_MAG7.md) |
| 计算因子 | [QUICKSTART_FACTORS.md](QUICKSTART_FACTORS.md) |
| 获取A股数据 | [CN_DATA_QUICKSTART.md](CN_DATA_QUICKSTART.md) |
| 开发规范 | [.cursorrules](../.cursorrules) |
| 版本历史 | [CHANGELOG.md](CHANGELOG.md) |
| 待实现功能 | [TODO.md](TODO.md) |

---

**最后更新**: 2025-12-23  
**项目状态**: 生产就绪 ✅  
**维护者**: Quant Team
