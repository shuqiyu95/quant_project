"""
LLM 模块测试
"""

import os
import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.llm import GeminiDeepResearchClient, ReportManager


class TestGeminiDeepResearchClient:
    """测试 Gemini Deep Research 客户端"""
    
    def test_init_with_api_key(self):
        """测试使用 API Key 初始化"""
        client = GeminiDeepResearchClient(api_key='test_api_key')
        assert client.api_key == 'test_api_key'
        assert client.timeout == 300
        assert client.max_retries == 3
    
    def test_init_from_env(self, monkeypatch):
        """测试从环境变量读取 API Key"""
        monkeypatch.setenv('GEMINI_API_KEY', 'env_api_key')
        client = GeminiDeepResearchClient()
        assert client.api_key == 'env_api_key'
    
    def test_init_no_api_key(self, monkeypatch):
        """测试未提供 API Key 时抛出异常"""
        # 使用 monkeypatch 删除环境变量
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        
        # 确保不通过参数传递 api_key
        with pytest.raises(ValueError, match="API Key 未提供"):
            GeminiDeepResearchClient(api_key=None)
    
    @patch('requests.post')
    def test_deep_research_success(self, mock_post):
        """测试成功执行深度研究"""
        # Mock API 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'This is a research report.'}]
                },
                'finishReason': 'STOP'
            }]
        }
        mock_post.return_value = mock_response
        
        client = GeminiDeepResearchClient(api_key='test_key')
        result = client.deep_research('Test query')
        
        assert 'content' in result
        assert result['content'] == 'This is a research report.'
        assert result['query'] == 'Test query'
        assert 'timestamp' in result
        assert 'elapsed_time' in result
    
    @patch('requests.post')
    def test_deep_research_with_metadata(self, mock_post):
        """测试带元数据的深度研究"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'Report content'}]
                },
                'finishReason': 'STOP'
            }]
        }
        mock_post.return_value = mock_response
        
        client = GeminiDeepResearchClient(api_key='test_key')
        result = client.deep_research(
            'Test query',
            metadata={'ticker': 'AAPL'}
        )
        
        assert result['metadata']['ticker'] == 'AAPL'
    
    @patch('requests.post')
    def test_deep_research_api_error(self, mock_post):
        """测试 API 错误处理"""
        mock_post.side_effect = Exception("API Error")
        
        client = GeminiDeepResearchClient(api_key='test_key')
        with pytest.raises(Exception, match="API Error"):
            client.deep_research('Test query')
    
    @patch('requests.post')
    def test_batch_research(self, mock_post):
        """测试批量研究"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'Report content'}]
                },
                'finishReason': 'STOP'
            }]
        }
        mock_post.return_value = mock_response
        
        client = GeminiDeepResearchClient(api_key='test_key')
        queries = ['Query 1', 'Query 2']
        
        results = client.batch_research(queries, delay=0.1)
        
        assert len(results) == 2
        assert all('content' in r for r in results)


class TestReportManager:
    """测试报告管理器"""
    
    @pytest.fixture
    def temp_report_dir(self, tmp_path):
        """创建临时报告目录"""
        report_dir = tmp_path / "reports"
        return str(report_dir)
    
    @pytest.fixture
    def sample_report_data(self):
        """示例报告数据"""
        return {
            'content': 'This is a test report content.',
            'query': 'Test query',
            'model': 'gemini-2.0-flash-thinking-exp',
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': 1.5,
            'metadata': {'ticker': 'AAPL'},
        }
    
    def test_init(self, temp_report_dir):
        """测试初始化"""
        manager = ReportManager(base_dir=temp_report_dir)
        assert manager.base_dir.exists()
    
    def test_save_report_basic(self, temp_report_dir, sample_report_data):
        """测试基础报告保存"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        report_path = manager.save_report(
            report_data=sample_report_data,
            filename='test_report'
        )
        
        assert os.path.exists(report_path)
        assert 'test_report.txt' in report_path
        
        # 验证内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == sample_report_data['content']
    
    def test_save_report_with_date(self, temp_report_dir, sample_report_data):
        """测试指定日期保存报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        custom_date = '2024-12-20'
        report_path = manager.save_report(
            report_data=sample_report_data,
            filename='test_report',
            date=custom_date
        )
        
        assert custom_date in report_path
    
    def test_save_report_auto_filename(self, temp_report_dir, sample_report_data):
        """测试自动生成文件名"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        report_path = manager.save_report(
            report_data=sample_report_data
        )
        
        assert os.path.exists(report_path)
        assert 'Test_query' in report_path or 'test_query' in report_path.lower()
    
    def test_save_report_with_metadata(self, temp_report_dir, sample_report_data):
        """测试保存带元数据的报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        report_path = manager.save_report(
            report_data=sample_report_data,
            filename='test_report',
            save_metadata=True
        )
        
        # 检查元数据文件
        metadata_path = report_path.replace('.txt', '.json')
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata['query'] == 'Test query'
        assert metadata['model'] == 'gemini-2.0-flash-thinking-exp'
    
    def test_save_report_invalid_data(self, temp_report_dir):
        """测试保存无效数据"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        with pytest.raises(ValueError, match="必须是字典类型"):
            manager.save_report("invalid data")
        
        with pytest.raises(ValueError, match="必须包含 'content' 字段"):
            manager.save_report({})
    
    def test_load_report(self, temp_report_dir, sample_report_data):
        """测试加载报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        # 先保存
        report_path = manager.save_report(
            report_data=sample_report_data,
            filename='test_report'
        )
        
        # 再加载
        loaded = manager.load_report(filename='test_report')
        
        assert loaded['content'] == sample_report_data['content']
        assert 'metadata' in loaded
        assert loaded['metadata']['query'] == 'Test query'
    
    def test_load_report_not_found(self, temp_report_dir):
        """测试加载不存在的报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        with pytest.raises(FileNotFoundError):
            manager.load_report(filename='nonexistent')
    
    def test_list_reports(self, temp_report_dir, sample_report_data):
        """测试列出报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        # 保存多个报告
        manager.save_report(sample_report_data, filename='report1')
        manager.save_report(sample_report_data, filename='report2')
        
        reports = manager.list_reports()
        
        assert len(reports) == 2
        assert any(r['filename'] == 'report1' for r in reports)
        assert any(r['filename'] == 'report2' for r in reports)
    
    def test_list_reports_with_metadata(self, temp_report_dir, sample_report_data):
        """测试列出报告包含元数据"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        manager.save_report(sample_report_data, filename='report1')
        
        reports = manager.list_reports(include_metadata=True)
        
        assert len(reports) == 1
        assert 'metadata' in reports[0]
        assert reports[0]['metadata']['query'] == 'Test query'
    
    def test_list_dates(self, temp_report_dir, sample_report_data):
        """测试列出所有日期"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        # 保存到不同日期
        manager.save_report(sample_report_data, filename='r1', date='2024-12-20')
        manager.save_report(sample_report_data, filename='r2', date='2024-12-21')
        
        dates = manager.list_dates()
        
        assert len(dates) == 2
        assert '2024-12-20' in dates
        assert '2024-12-21' in dates
    
    def test_delete_report(self, temp_report_dir, sample_report_data):
        """测试删除报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        # 保存报告
        manager.save_report(sample_report_data, filename='test_report')
        
        # 删除报告
        result = manager.delete_report(filename='test_report')
        
        assert result is True
        
        # 验证已删除
        with pytest.raises(FileNotFoundError):
            manager.load_report(filename='test_report')
    
    def test_delete_report_not_found(self, temp_report_dir):
        """测试删除不存在的报告"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        result = manager.delete_report(filename='nonexistent')
        assert result is False
    
    def test_search_reports_metadata(self, temp_report_dir):
        """测试搜索报告元数据"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        # 保存不同的报告
        report1 = {
            'content': 'Report about AAPL',
            'query': 'Analyze AAPL stock',
            'model': 'test',
            'timestamp': datetime.now().isoformat(),
        }
        
        report2 = {
            'content': 'Report about TSLA',
            'query': 'Analyze TSLA stock',
            'model': 'test',
            'timestamp': datetime.now().isoformat(),
        }
        
        manager.save_report(report1, filename='aapl_report')
        manager.save_report(report2, filename='tsla_report')
        
        # 搜索
        results = manager.search_reports(keyword='AAPL', search_content=False)
        
        assert len(results) == 1
        assert results[0]['filename'] == 'aapl_report'
    
    def test_search_reports_content(self, temp_report_dir):
        """测试搜索报告内容"""
        manager = ReportManager(base_dir=temp_report_dir)
        
        report = {
            'content': 'This report discusses Tesla and SpaceX',
            'query': 'General tech report',
            'model': 'test',
            'timestamp': datetime.now().isoformat(),
        }
        
        manager.save_report(report, filename='tech_report')
        
        # 搜索内容
        results = manager.search_reports(keyword='Tesla', search_content=True)
        
        assert len(results) == 1
        assert results[0]['filename'] == 'tech_report'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

