🤖 Claude 开发指南：跨市场多层级量化分析系统
1. 项目愿景
构建一个支持 A股 (CN) 与 美股 (US) 的量化框架。系统应具备处理日线 (L1) 到高频 (L2/L3) 数据扩展的能力，并能集成 Qlib 风格的因子表达。首个里程碑是实现 Mag7（美股七巨头）的 5 日周期择股策略。
2. 目录架构与职责 (Layered Architecture)
所有代码生成必须严格遵守以下结构：

quant_project/
├── docs/
│   └── claude.md         # 本开发说明文件
├── src/
│   ├── data_engine/      # 数据获取层 (yfinance/AkShare)
│   ├── factors/          # 因子计算层 (Qlib风格算子)
│   ├── models/           # 机器学习模型与 Loss 函数
│   ├── backtester/       # 回测引擎 (Backtrader封装)
│   └── utils/            # 交易日历、时区转换等工具
├── data/                 # 本地缓存 (Parquet/CSV)
├── test/                 # 测试脚本路径
└── main.py               # 策略运行脚本
3. 技术规范 (Technology Stack)
Python: 3.10+

数据驱动: 使用 Pandas (Vectorized) 进行运算，使用parquet进行数据存储

因子逻辑:

参考 Qlib 的表达式语法 (Expression Engine)。

在 src/factors/ 中实现类似 Ref(close, n), Std(close, n), Slope(close, n) 的通用算子。

数据源:

US: yfinance

CN: AkShare

L2/L3: 预留 tick_data 处理接口，支持大单流分析。

每当完成一个核心模块，请自动在 test/ 目录下生成对应的测试文件。确保所有测试都能通过 pytest 指令一键运行。

4. 关键逻辑约束 (Core Logic)
    4.1 跨市场处理A股: 考虑 T+1 制度，印花税 (0.05%)，涨跌停限制。美股: 考虑 T+0 制度，最小交易单位 1 股。时区: 统一处理为 Timestamp 对象，避免硬编码时区偏移。
    4.2 第一个策略：Mag7 5-Day Rotation标的池: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA。目标: 预测 $T+5$ 收益率排名。损失函数 (Loss Function):实现 RankMSE：不仅关注收益率数值，更关注股票间的相对排名顺序。公式参考：$Loss = \sum (y_{pred\_rank} - y_{true\_rank})^2$ 或对排名逆转进行惩罚。
    4.3 Qlib 集成方式轻量化方案: 在 src/factors/ 中直接封装 Qlib 的经典 Alpha158/360 算子逻辑，无需完整安装 Qlib 环境。扩展性: 接口设计需满足 df_ohlcv -> df_factors 的纯函数转换。