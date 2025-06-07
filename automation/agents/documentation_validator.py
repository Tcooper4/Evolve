import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import re
import ast
from dataclasses import dataclass
from datetime import datetime
import markdown
import docutils
import yaml
import json
import sphinx
import pylint
import mypy
import black
import isort
import flake8
import bandit
import safety
import coverage
import pytest
import doctest
import docstring_parser
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
import validators
import aiofiles
import frontmatter
import mistune
from mistune import Markdown
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import rst2html
import rst2pdf
import doc8
import restructuredtext_lint
import markdownlint
import remark
import prettier
import mccabe
import xenon
import vulture
import pyflakes
import pycodestyle

@dataclass
class ValidationIssue:
    id: str
    type: str
    severity: str
    message: str
    location: Dict[str, Any]
    context: Dict[str, Any]
    fix: Optional[str]
    created_at: datetime

@dataclass
class ValidationResult:
    id: str
    type: str
    status: str
    issues: List[ValidationIssue]
    score: float
    timestamp: str
    metadata: Dict

class DocumentationValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.validator_config = config.get('documentation', {}).get('validation', {})
        self.setup_validators()
        self.validation_stats = {
            'files_checked': 0,
            'issues_found': 0,
            'fixes_applied': 0,
            'avg_duration': 0
        }
        self.results: Dict[str, ValidationResult] = {}

    def setup_logging(self):
        """Configure logging for the documentation validator."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "validation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_validators(self):
        """Setup validation tools and configurations."""
        # Link validation
        self.link_validator = LinkValidator(self.validator_config.get('links', {}))
        
        # Content validation
        self.content_validator = ContentValidator(self.validator_config.get('content', {}))
        
        # Code validation
        self.code_validator = CodeValidator(self.validator_config.get('code', {}))
        
        # Style validation
        self.style_validator = StyleValidator(self.validator_config.get('style', {}))

    async def validate_file(
        self,
        file_path: Union[str, Path],
        validation_types: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate a documentation file."""
        try:
            start_time = datetime.now()
            self.validation_stats['files_checked'] += 1

            if isinstance(file_path, str):
                file_path = Path(file_path)

            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Determine file type
            file_type = self._get_file_type(file_path)

            # Collect validation issues
            issues = []
            
            # Validate based on file type and requested validation types
            if not validation_types or 'links' in validation_types:
                issues.extend(await self.link_validator.validate(content, file_path))
            
            if not validation_types or 'content' in validation_types:
                issues.extend(await self.content_validator.validate(content, file_path))
            
            if not validation_types or 'code' in validation_types:
                issues.extend(await self.code_validator.validate(content, file_path))
            
            if not validation_types or 'style' in validation_types:
                issues.extend(await self.style_validator.validate(content, file_path))

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Update stats
            self.validation_stats['issues_found'] += len(issues)
            self.validation_stats['avg_duration'] = (
                (self.validation_stats['avg_duration'] * (self.validation_stats['files_checked'] - 1) +
                 duration) / self.validation_stats['files_checked']
            )

            # Create validation result
            result = ValidationResult(
                id=str(file_path),
                type=file_type,
                status='pass' if self._calculate_score(issues) >= self.validator_config.get('min_score', 0.8) else 'fail',
                issues=issues,
                score=self._calculate_score(issues),
                timestamp=datetime.now().isoformat(),
                metadata={
                    'file_type': file_type,
                    'validation_types': validation_types or ['all'],
                    'timestamp': datetime.now().isoformat()
                }
            )

            self.results[str(file_path)] = result
            self.logger.info(f"Validated file: {file_path}")
            
            return result

        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            raise

    async def validate_directory(
        self,
        directory_path: Union[str, Path],
        validation_types: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """Validate all documentation files in a directory."""
        try:
            if isinstance(directory_path, str):
                directory_path = Path(directory_path)

            results = {}
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    if not file_patterns or any(
                        file_path.match(pattern) for pattern in file_patterns
                    ):
                        results[str(file_path)] = await self.validate_file(
                            file_path,
                            validation_types
                        )

            return results

        except Exception as e:
            self.logger.error(f"Directory validation error: {str(e)}")
            raise

    async def fix_issues(
        self,
        file_path: Union[str, Path],
        issues: List[ValidationIssue]
    ) -> bool:
        """Fix validation issues in a file."""
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Apply fixes
            fixed_content = content
            for issue in issues:
                if issue.fix:
                    fixed_content = self._apply_fix(fixed_content, issue)

            # Write fixed content
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(fixed_content)

            self.validation_stats['fixes_applied'] += len(issues)
            self.logger.info(f"Fixed issues in file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Issue fixing error: {str(e)}")
            return False

    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        extension = file_path.suffix.lower()
        if extension == '.md':
            return 'markdown'
        elif extension == '.rst':
            return 'restructuredtext'
        elif extension == '.py':
            return 'python'
        elif extension == '.html':
            return 'html'
        elif extension == '.txt':
            return 'text'
        else:
            return 'unknown'

    def _summarize_issues(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Summarize issues by type and severity."""
        summary = {
            'total': len(issues),
            'by_type': {},
            'by_severity': {
                'error': 0,
                'warning': 0,
                'info': 0
            }
        }

        for issue in issues:
            # Count by type
            if issue.type not in summary['by_type']:
                summary['by_type'][issue.type] = 0
            summary['by_type'][issue.type] += 1

            # Count by severity
            summary['by_severity'][issue.severity] += 1

        return summary

    def _apply_fix(self, content: str, issue: ValidationIssue) -> str:
        """Apply a fix to the content."""
        try:
            if issue.type == 'link':
                return self.link_validator.fix_link(content, issue)
            elif issue.type == 'content':
                return self.content_validator.fix_content(content, issue)
            elif issue.type == 'code':
                return self.code_validator.fix_code(content, issue)
            elif issue.type == 'style':
                return self.style_validator.fix_style(content, issue)
            else:
                return content
        except Exception as e:
            self.logger.error(f"Fix application error: {str(e)}")
            return content

    def get_stats(self) -> Dict:
        """Get validation statistics."""
        return {
            'files_checked': self.validation_stats['files_checked'],
            'issues_found': self.validation_stats['issues_found'],
            'fixes_applied': self.validation_stats['fixes_applied'],
            'avg_duration': self.validation_stats['avg_duration']
        }

    def get_validation_result(self, doc_id: str) -> Optional[ValidationResult]:
        """Get validation result for a document."""
        return self.results.get(doc_id)

    def get_validation_results(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[ValidationResult]:
        """Get validation results with optional filtering."""
        results = list(self.results.values())
        
        if doc_type:
            results = [r for r in results if r.type == doc_type]
        if status:
            results = [r for r in results if r.status == status]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        
        return results

    def clear_validation_results(self, before: Optional[datetime] = None):
        """Clear old validation results."""
        if before:
            self.results = {
                id: r for id, r in self.results.items()
                if datetime.fromisoformat(r.timestamp) > before
            }
        else:
            self.results.clear()
        
        self.logger.info(f"Cleared validation results before {before}")

    async def generate_validation_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """Generate validation report."""
        if not output_path:
            output_path = f"docs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate report content
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_docs': len(self.results),
                'passing_docs': len([r for r in self.results.values() if r.status == 'pass']),
                'failing_docs': len([r for r in self.results.values() if r.status == 'fail']),
                'average_score': sum(r.score for r in self.results.values()) / len(self.results) if self.results else 0,
                'results': [
                    {
                        'id': r.id,
                        'type': r.type,
                        'status': r.status,
                        'score': r.score,
                        'issues': r.issues
                    }
                    for r in self.results.values()
                ]
            }
            
            # Generate HTML report
            template = jinja2.Template("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Documentation Validation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                    .result { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
                    .pass { background: #e6ffe6; }
                    .fail { background: #ffe6e6; }
                    .issue { margin: 5px 0; padding: 5px; background: #fff; }
                    .error { color: red; }
                    .warning { color: orange; }
                    .info { color: blue; }
                </style>
            </head>
            <body>
                <h1>Documentation Validation Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Generated: {{ timestamp }}</p>
                    <p>Total Documents: {{ total_docs }}</p>
                    <p>Passing Documents: {{ passing_docs }}</p>
                    <p>Failing Documents: {{ failing_docs }}</p>
                    <p>Average Score: {{ "%.2f"|format(average_score) }}</p>
                </div>
                
                <h2>Results</h2>
                {% for result in results %}
                <div class="result {{ result.status }}">
                    <h3>{{ result.type }} - {{ result.id }}</h3>
                    <p>Status: {{ result.status }}</p>
                    <p>Score: {{ "%.2f"|format(result.score) }}</p>
                    
                    {% if result.issues %}
                    <h4>Issues</h4>
                    {% for issue in result.issues %}
                    <div class="issue {{ issue.severity }}">
                        <strong>{{ issue.type }} ({{ issue.severity }}):</strong>
                        {{ issue.message }}
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>
                {% endfor %}
            </body>
            </html>
            """)
            
            html = template.render(**report)
            output_path.write_text(html)
            
            self.logger.info(f"Generated validation report: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {str(e)}")
            raise 