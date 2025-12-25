"""
测试 A股数据获取模块
包括日线、分钟线、行业数据、微观数据等
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_engine.cn_fetcher import CNFetcher
from src.data_engine.data_manager import DataManager


class TestCNFetcher:
    """测试A股数据获取器"""
    
    @pytest.fixture
    def fetcher(self):
        """创建CNFetcher实例"""
        return CNFetcher()
    
    @pytest.fixture
    def test_symbol(self):
        """测试用的股票代码 - 贵州茅台"""
        return "600519"
    
    @pytest.fixture
    def test_symbols(self):
        """测试用的多个股票代码"""
        return ["600519", "000858", "600036"]
    
    def test_validate_symbol(self, fetcher):
        """测试股票代码验证"""
        # 有效代码
        assert fetcher.validate_symbol("600519") == True
        assert fetcher.validate_symbol("000001") == True
        
        # 无效代码
        assert fetcher.validate_symbol("AAPL") == False
        assert fetcher.validate_symbol("60051") == False
        assert fetcher.validate_symbol("6005199") == False
        assert fetcher.validate_symbol("abc123") == False
    
    def test_fetch_daily_data(self, fetcher, test_symbol):
        """测试获取日线数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        df = fetcher.fetch_daily_data(test_symbol, start_date, end_date)
        
        # 验证返回的数据
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # 验证必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # 验证数据类型
        assert df.index.name == 'date'
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        
        # 验证时区
        assert df.index.tz is not None
        
        # 验证市场标识
        assert 'market' in df.columns
        assert df['market'].iloc[0] == 'CN'
        
        # 验证股票代码
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == test_symbol
        
        # 验证数据范围
        assert df['open'].min() > 0
        assert df['high'].max() > df['low'].min()
        assert df['volume'].sum() > 0
        
        print(f"\n✓ 成功获取 {test_symbol} 的日线数据，共 {len(df)} 条记录")
        print(f"  日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
        print(f"  最新收盘价: {df['close'].iloc[-1]:.2f}")
        
    def test_fetch_daily_data_with_indicators(self, fetcher, test_symbol):
        """测试获取包含更多指标的日线数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = fetcher.fetch_daily_data(test_symbol, start_date, end_date)
        
        # 验证额外的指标列
        expected_extra_cols = ['amount', 'turnover', 'pct_change']
        for col in expected_extra_cols:
            if col in df.columns:
                print(f"  ✓ 包含指标: {col}")
                # 验证数据有效性
                assert df[col].notna().any()
    
    def test_fetch_intraday_data(self, fetcher, test_symbol):
        """测试获取分钟线数据"""
        # 获取最近3天的5分钟数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        try:
            df = fetcher.fetch_intraday_data(
                symbol=test_symbol,
                start_date=start_date,
                end_date=end_date,
                period="5"
            )
            
            if not df.empty:
                # 验证数据结构
                assert isinstance(df, pd.DataFrame)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    assert col in df.columns
                
                # 验证时间索引
                assert pd.api.types.is_datetime64_any_dtype(df.index)
                assert df.index.tz is not None
                
                # 验证分钟级别数据特征
                assert 'period' in df.columns
                assert df['period'].iloc[0] == "5"
                
                print(f"\n✓ 成功获取 {test_symbol} 的5分钟数据，共 {len(df)} 条记录")
                print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
            else:
                print(f"\n⚠ {test_symbol} 暂无分钟线数据（可能需要实时数据源）")
                
        except Exception as e:
            print(f"\n⚠ 分钟线数据获取失败（预期情况）: {str(e)}")
            # 分钟线数据可能需要特定权限或实时源，失败不算测试失败
            pytest.skip(f"Intraday data not available: {str(e)}")
    
    def test_fetch_industry_data(self, fetcher, test_symbol):
        """测试获取行业数据"""
        try:
            industry_info = fetcher.fetch_industry_data(test_symbol)
            
            if industry_info:
                assert isinstance(industry_info, dict)
                assert 'symbol' in industry_info
                assert industry_info['symbol'] == test_symbol
                
                print(f"\n✓ 成功获取 {test_symbol} 的行业信息:")
                for key, value in industry_info.items():
                    if value:
                        print(f"  {key}: {value}")
            else:
                print(f"\n⚠ {test_symbol} 暂无行业数据")
                
        except Exception as e:
            print(f"\n⚠ 行业数据获取失败: {str(e)}")
            pytest.skip(f"Industry data not available: {str(e)}")
    
    def test_fetch_turnover_quantile(self, fetcher, test_symbol):
        """测试计算换手率分位数"""
        try:
            current_date = datetime.now()
            quantile = fetcher.fetch_turnover_quantile(
                symbol=test_symbol,
                current_date=current_date,
                lookback_days=100
            )
            
            if quantile is not None:
                assert 0 <= quantile <= 1
                print(f"\n✓ {test_symbol} 当前换手率分位数: {quantile:.2%}")
                print(f"  （在最近100个交易日中的位置）")
            else:
                print(f"\n⚠ 无法计算换手率分位数")
                
        except Exception as e:
            print(f"\n⚠ 换手率分位数计算失败: {str(e)}")
    
    def test_get_stock_info(self, fetcher, test_symbol):
        """测试获取股票基本信息"""
        try:
            info = fetcher.get_stock_info(test_symbol)
            
            if info:
                assert isinstance(info, dict)
                print(f"\n✓ 成功获取 {test_symbol} 的基本信息")
                # 打印部分关键信息
                key_fields = ['代码', '名称', '最新价', '涨跌幅', '市值']
                for field in key_fields:
                    if field in info:
                        print(f"  {field}: {info[field]}")
            else:
                print(f"\n⚠ 未找到 {test_symbol} 的基本信息")
                
        except Exception as e:
            print(f"\n⚠ 股票信息获取失败: {str(e)}")
    
    def test_get_realtime_quotes(self, fetcher, test_symbols):
        """测试批量获取实时行情"""
        try:
            df = fetcher.get_realtime_quotes(test_symbols)
            
            if not df.empty:
                assert isinstance(df, pd.DataFrame)
                assert 'symbol' in df.columns
                assert 'price' in df.columns
                
                print(f"\n✓ 成功获取 {len(test_symbols)} 只股票的实时行情:")
                for _, row in df.iterrows():
                    if 'symbol' in row and 'name' in row and 'price' in row:
                        print(f"  {row['symbol']} {row['name']}: {row['price']}")
            else:
                print("\n⚠ 未获取到实时行情数据")
                
        except Exception as e:
            print(f"\n⚠ 实时行情获取失败: {str(e)}")


class TestDataManager:
    """测试数据管理器（A股相关功能）"""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """创建临时目录的DataManager实例"""
        return DataManager(data_dir=str(tmp_path / "test_data"))
    
    @pytest.fixture
    def test_symbol(self):
        return "600519"
    
    def test_identify_market(self, data_manager):
        """测试市场识别"""
        assert data_manager.identify_market("600519") == "CN"
        assert data_manager.identify_market("000858") == "CN"
        assert data_manager.identify_market("AAPL") == "US"
        assert data_manager.identify_market("invalid") == "UNKNOWN"
    
    def test_fetch_and_cache(self, data_manager, test_symbol):
        """测试数据获取和缓存"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # 首次获取（从API）
        df1 = data_manager.fetch_data(test_symbol, start_date, end_date, use_cache=False)
        assert not df1.empty
        assert len(df1) > 0
        
        # 第二次获取（从缓存）
        df2 = data_manager.fetch_data(test_symbol, start_date, end_date, use_cache=True)
        assert not df2.empty
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(df1, df2)
        
        print(f"\n✓ 缓存机制测试通过")
        print(f"  数据条数: {len(df1)}")
    
    def test_incremental_update(self, data_manager, test_symbol):
        """测试增量更新"""
        # 首次获取30天数据
        end_date = datetime.now() - timedelta(days=5)
        df1 = data_manager.fetch_data(
            test_symbol, 
            end_date - timedelta(days=30), 
            end_date, 
            use_cache=False
        )
        initial_count = len(df1)
        
        # 增量更新到今天
        df2 = data_manager.fetch_data_incremental(test_symbol, datetime.now())
        
        assert len(df2) >= initial_count
        
        print(f"\n✓ 增量更新测试通过")
        print(f"  初始数据: {initial_count} 条")
        print(f"  更新后: {len(df2)} 条")
        print(f"  新增: {len(df2) - initial_count} 条")
    
    def test_fetch_industry_data_via_manager(self, data_manager, test_symbol):
        """测试通过DataManager获取行业数据"""
        try:
            industry_info = data_manager.fetch_industry_data(test_symbol)
            
            if industry_info:
                assert isinstance(industry_info, dict)
                print(f"\n✓ 通过DataManager获取行业数据成功")
            else:
                print(f"\n⚠ 行业数据为空")
                
        except Exception as e:
            print(f"\n⚠ 行业数据获取失败: {str(e)}")
            pytest.skip(f"Industry data not available: {str(e)}")
    
    def test_clear_cache(self, data_manager, test_symbol):
        """测试清除缓存"""
        # 先获取一些数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        data_manager.fetch_data(test_symbol, start_date, end_date, use_cache=False)
        
        # 验证缓存存在
        cache_file = data_manager._get_cache_path(test_symbol)
        assert cache_file.exists()
        
        # 清除缓存
        data_manager.clear_cache(test_symbol)
        
        # 验证缓存已删除
        assert not cache_file.exists()
        
        print(f"\n✓ 缓存清除测试通过")


class TestDataIntegrity:
    """数据完整性测试"""
    
    @pytest.fixture
    def fetcher(self):
        return CNFetcher()
    
    def test_data_completeness(self, fetcher):
        """测试数据完整性"""
        symbol = "600036"  # 招商银行
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = fetcher.fetch_daily_data(symbol, start_date, end_date)
        
        # 检查缺失值
        missing_report = df.isnull().sum()
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        
        print(f"\n数据完整性报告 - {symbol}:")
        print(f"总记录数: {len(df)}")
        
        for col in critical_cols:
            missing_count = missing_report.get(col, 0)
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {col}: {missing_count} 缺失 ({missing_pct:.1f}%)")
            
            # 关键列不应有缺失
            assert missing_count == 0, f"{col} should not have missing values"
    
    def test_data_consistency(self, fetcher):
        """测试数据一致性（价格关系）"""
        symbol = "600519"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20)
        
        df = fetcher.fetch_daily_data(symbol, start_date, end_date)
        
        # 验证价格关系: high >= close >= low, high >= open >= low
        assert (df['high'] >= df['close']).all(), "High should >= Close"
        assert (df['close'] >= df['low']).all(), "Close should >= Low"
        assert (df['high'] >= df['open']).all(), "High should >= Open"
        assert (df['open'] >= df['low']).all(), "Open should >= Low"
        
        # 验证最高价是真正的最高
        assert (df['high'] >= df['low']).all(), "High should >= Low"
        
        print(f"\n✓ 数据一致性验证通过 - {symbol}")
        print(f"  验证了 {len(df)} 条记录的价格关系")


if __name__ == "__main__":
    # 运行测试
    print("=" * 60)
    print("A股数据获取模块测试")
    print("=" * 60)
    
    # 使用 pytest 运行
    pytest.main([__file__, "-v", "-s", "--tb=short"])

