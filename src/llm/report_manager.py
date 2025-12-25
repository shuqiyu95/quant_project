"""
研究报告管理器

功能：
- 保存研究报告到本地
- 按日期组织报告
- 报告检索和加载
- 报告元数据管理
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportManager:
    """
    研究报告管理器
    
    使用示例：
        >>> manager = ReportManager(base_dir='data/reports')
        >>> manager.save_report(result, filename='tsla_q4_analysis')
        >>> reports = manager.list_reports(date='2024-12-25')
    """
    
    def __init__(self, base_dir: str = "data/reports"):
        """
        初始化报告管理器
        
        Args:
            base_dir: 报告根目录，默认为 'data/reports'
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"报告管理器初始化完成: base_dir={self.base_dir.absolute()}")
    
    def save_report(
        self,
        report_data: Dict[str, Any],
        filename: Optional[str] = None,
        date: Optional[str] = None,
        save_metadata: bool = True
    ) -> str:
        """
        保存研究报告
        
        Args:
            report_data: 报告数据（来自 GeminiDeepResearchClient.deep_research）
            filename: 文件名（不含扩展名），如果不提供则自动生成
            date: 日期字符串 (YYYY-MM-DD)，如果不提供则使用今天
            save_metadata: 是否同时保存元数据文件
            
        Returns:
            保存的文件路径
            
        Raises:
            ValueError: 如果 report_data 格式不正确
        """
        # 验证报告数据
        if not isinstance(report_data, dict):
            raise ValueError("report_data 必须是字典类型")
        
        if 'content' not in report_data:
            raise ValueError("report_data 必须包含 'content' 字段")
        
        # 确定日期
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # 创建日期目录
        date_dir = self.base_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%H%M%S')
            query_short = report_data.get('query', 'report')[:30]
            # 清理文件名中的非法字符
            query_short = "".join(c for c in query_short if c.isalnum() or c in (' ', '-', '_'))
            query_short = query_short.replace(' ', '_')
            filename = f"{timestamp}_{query_short}"
        
        # 保存报告内容
        report_path = date_dir / f"{filename}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_data['content'])
        
        logger.info(f"报告内容已保存: {report_path}")
        
        # 保存元数据
        if save_metadata:
            metadata = {
                'query': report_data.get('query', ''),
                'model': report_data.get('model', ''),
                'timestamp': report_data.get('timestamp', datetime.now().isoformat()),
                'elapsed_time': report_data.get('elapsed_time', 0),
                'metadata': report_data.get('metadata', {}),
                'content_length': len(report_data['content']),
                'has_thinking_process': bool(report_data.get('thinking_process')),
            }
            
            metadata_path = date_dir / f"{filename}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"报告元数据已保存: {metadata_path}")
            
            # 如果有思考过程，也保存下来
            if report_data.get('thinking_process'):
                thinking_path = date_dir / f"{filename}_thinking.txt"
                with open(thinking_path, 'w', encoding='utf-8') as f:
                    f.write(report_data['thinking_process'])
                logger.info(f"思考过程已保存: {thinking_path}")
        
        return str(report_path)
    
    def load_report(
        self,
        filename: str,
        date: Optional[str] = None,
        load_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        加载研究报告
        
        Args:
            filename: 文件名（不含扩展名）
            date: 日期字符串 (YYYY-MM-DD)，如果不提供则使用今天
            load_metadata: 是否同时加载元数据
            
        Returns:
            报告数据字典，包含 content 和 metadata (如果有)
            
        Raises:
            FileNotFoundError: 如果报告文件不存在
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        date_dir = self.base_dir / date
        report_path = date_dir / f"{filename}.txt"
        
        if not report_path.exists():
            raise FileNotFoundError(f"报告文件不存在: {report_path}")
        
        # 加载报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            'content': content,
            'filename': filename,
            'date': date,
            'path': str(report_path),
        }
        
        # 加载元数据
        if load_metadata:
            metadata_path = date_dir / f"{filename}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                result['metadata'] = metadata
            
            # 加载思考过程
            thinking_path = date_dir / f"{filename}_thinking.txt"
            if thinking_path.exists():
                with open(thinking_path, 'r', encoding='utf-8') as f:
                    result['thinking_process'] = f.read()
        
        logger.info(f"报告已加载: {report_path}")
        return result
    
    def list_reports(
        self,
        date: Optional[str] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        列出指定日期的所有报告
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)，如果不提供则使用今天
            include_metadata: 是否包含元数据信息
            
        Returns:
            报告列表，每个元素包含 filename, path 等信息
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        date_dir = self.base_dir / date
        
        if not date_dir.exists():
            logger.warning(f"日期目录不存在: {date_dir}")
            return []
        
        reports = []
        for report_path in sorted(date_dir.glob("*.txt")):
            if report_path.stem.endswith('_thinking'):
                continue  # 跳过思考过程文件
            
            report_info = {
                'filename': report_path.stem,
                'path': str(report_path),
                'date': date,
                'size': report_path.stat().st_size,
                'modified': datetime.fromtimestamp(
                    report_path.stat().st_mtime
                ).isoformat(),
            }
            
            # 包含元数据
            if include_metadata:
                metadata_path = report_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        report_info['metadata'] = json.load(f)
            
            reports.append(report_info)
        
        logger.info(f"找到 {len(reports)} 个报告: date={date}")
        return reports
    
    def list_dates(self) -> List[str]:
        """
        列出所有有报告的日期
        
        Returns:
            日期列表，格式为 YYYY-MM-DD
        """
        dates = []
        for date_dir in sorted(self.base_dir.iterdir()):
            if date_dir.is_dir() and date_dir.name.count('-') == 2:
                dates.append(date_dir.name)
        
        logger.info(f"找到 {len(dates)} 个日期")
        return dates
    
    def delete_report(
        self,
        filename: str,
        date: Optional[str] = None
    ) -> bool:
        """
        删除研究报告
        
        Args:
            filename: 文件名（不含扩展名）
            date: 日期字符串 (YYYY-MM-DD)，如果不提供则使用今天
            
        Returns:
            是否成功删除
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        date_dir = self.base_dir / date
        
        # 删除所有相关文件
        deleted_count = 0
        for ext in ['.txt', '.json', '_thinking.txt']:
            file_path = date_dir / f"{filename}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted_count += 1
                logger.info(f"已删除: {file_path}")
        
        if deleted_count > 0:
            logger.info(f"报告已删除: filename={filename}, date={date}")
            return True
        else:
            logger.warning(f"未找到报告: filename={filename}, date={date}")
            return False
    
    def search_reports(
        self,
        keyword: str,
        date: Optional[str] = None,
        search_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的报告
        
        Args:
            keyword: 搜索关键词
            date: 日期字符串 (YYYY-MM-DD)，如果不提供则搜索所有日期
            search_content: 是否搜索报告内容（否则只搜索元数据）
            
        Returns:
            匹配的报告列表
        """
        if date:
            dates = [date]
        else:
            dates = self.list_dates()
        
        matched_reports = []
        
        for date in dates:
            reports = self.list_reports(date=date, include_metadata=True)
            
            for report in reports:
                matched = False
                
                # 搜索元数据
                metadata = report.get('metadata', {})
                query = metadata.get('query', '')
                if keyword.lower() in query.lower():
                    matched = True
                
                # 搜索内容
                if search_content and not matched:
                    try:
                        report_data = self.load_report(
                            filename=report['filename'],
                            date=date,
                            load_metadata=False
                        )
                        if keyword.lower() in report_data['content'].lower():
                            matched = True
                    except Exception as e:
                        logger.warning(f"搜索报告内容失败: {report['filename']}, error={str(e)}")
                
                if matched:
                    matched_reports.append(report)
        
        logger.info(f"搜索完成: keyword='{keyword}', found={len(matched_reports)}")
        return matched_reports

