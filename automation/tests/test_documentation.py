import pytest
import asyncio
from pathlib import Path
import json
import yaml
import markdown
from datetime import datetime
from automation.agents.documentation import DocumentationGenerator
from automation.agents.documentation_validator import DocumentationValidator
from automation.agents.documentation_search import DocumentationSearch
from automation.agents.documentation_version import DocumentationVersion

@pytest.fixture
def config():
    return {
        'documentation': {
            'validation': {
                'min_length': 100,
                'min_score': 0.8
            },
            'search': {
                'engine': 'elasticsearch',
                'elasticsearch': {
                    'hosts': ['localhost:9200']
                }
            },
            'versioning': {
                'storage': 'local',
                'local': {
                    'path': 'test_versions'
                }
            }
        }
    }

@pytest.fixture
def doc_generator(config):
    return DocumentationGenerator(config)

@pytest.fixture
def doc_validator(config):
    return DocumentationValidator(config)

@pytest.fixture
def doc_search(config):
    return DocumentationSearch(config)

@pytest.fixture
def doc_version(config):
    return DocumentationVersion(config)

@pytest.mark.asyncio
async def test_generate_documentation(doc_generator):
    """Test documentation generation."""
    # Generate API documentation
    api_doc_id = await doc_generator.generate_api_docs('test_module.py')
    assert api_doc_id is not None
    
    # Generate system documentation
    system_doc_id = await doc_generator.generate_system_docs()
    assert system_doc_id is not None
    
    # Generate code documentation
    code_doc_id = await doc_generator.generate_code_docs('test_file.py')
    assert code_doc_id is not None

@pytest.mark.asyncio
async def test_validate_documentation(doc_validator):
    """Test documentation validation."""
    # Test API documentation validation
    api_result = await doc_validator.validate_documentation(
        doc_id='test_api',
        doc_type='api',
        content='# API Documentation\n\n## Endpoints\n\n### GET /users\nGet user list',
        format='markdown'
    )
    assert api_result.status == 'pass'
    assert api_result.score >= 0.8
    
    # Test system documentation validation
    system_result = await doc_validator.validate_documentation(
        doc_id='test_system',
        doc_type='system',
        content='# System Documentation\n\n## Architecture\n\n### Components\n\n#### API Server\nHandles API requests',
        format='markdown'
    )
    assert system_result.status == 'pass'
    assert system_result.score >= 0.8
    
    # Test code documentation validation
    code_result = await doc_validator.validate_documentation(
        doc_id='test_code',
        doc_type='code',
        content='# Code Documentation\n\n## Classes\n\n### User\nUser management class',
        format='markdown'
    )
    assert code_result.status == 'pass'
    assert code_result.score >= 0.8

@pytest.mark.asyncio
async def test_search_documentation(doc_search):
    """Test documentation search."""
    # Index test documents
    await doc_search.index_documentation(
        doc_id='test1',
        doc_type='api',
        title='Test API',
        content='API documentation for testing'
    )
    
    await doc_search.index_documentation(
        doc_id='test2',
        doc_type='system',
        title='Test System',
        content='System documentation for testing'
    )
    
    # Test search
    results = await doc_search.search_documentation('test')
    assert len(results) > 0
    
    # Test type filtering
    api_results = await doc_search.search_documentation('test', doc_type='api')
    assert all(r.type == 'api' for r in api_results)
    
    # Test suggestions
    suggestions = await doc_search.get_suggestions('test')
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_version_documentation(doc_version):
    """Test documentation versioning."""
    # Create initial version
    version1 = await doc_version.create_version(
        doc_id='test_doc',
        content='Initial version',
        author='test_user'
    )
    assert version1.version == '1.0.0'
    
    # Create new version
    version2 = await doc_version.create_version(
        doc_id='test_doc',
        content='Updated version',
        author='test_user'
    )
    assert version2.version == '1.0.1'
    
    # Get version
    retrieved_version = await doc_version.get_version('test_doc', '1.0.0')
    assert retrieved_version.content == 'Initial version'
    
    # Get latest version
    latest_version = doc_version.get_latest_version('test_doc')
    assert latest_version.version == '1.0.1'
    
    # Get version history
    history = doc_version.get_version_history('test_doc')
    assert len(history) == 2
    
    # Compare versions
    comparison = await doc_version.compare_versions(
        'test_doc',
        '1.0.0',
        '1.0.1'
    )
    assert len(comparison['changes']) > 0
    
    # Rollback version
    rollback_version = await doc_version.rollback_version('test_doc', '1.0.0')
    assert rollback_version.version == '1.0.2'
    
    # Cleanup versions
    await doc_version.cleanup_versions('test_doc', keep_versions=1)
    history = doc_version.get_version_history('test_doc')
    assert len(history) == 1

@pytest.mark.asyncio
async def test_documentation_export(doc_generator):
    """Test documentation export."""
    # Generate test documentation
    doc_id = await doc_generator.generate_api_docs('test_module.py')
    
    # Export to HTML
    html_path = await doc_generator.export_documentation(
        format='html',
        output_path='test_export'
    )
    assert Path(html_path).exists()
    
    # Export to PDF
    pdf_path = await doc_generator.export_documentation(
        format='pdf',
        output_path='test_export'
    )
    assert Path(pdf_path).exists()
    
    # Export to Markdown
    md_path = await doc_generator.export_documentation(
        format='markdown',
        output_path='test_export'
    )
    assert Path(md_path).exists()

@pytest.mark.asyncio
async def test_documentation_validation_rules(doc_validator):
    """Test documentation validation rules."""
    # Test minimum length
    result = await doc_validator.validate_documentation(
        doc_id='test_short',
        doc_type='api',
        content='Short content',
        format='markdown'
    )
    assert result.status == 'fail'
    assert 'length' in [issue['type'] for issue in result.issues]
    
    # Test missing sections
    result = await doc_validator.validate_documentation(
        doc_id='test_no_sections',
        doc_type='api',
        content='Content without sections',
        format='markdown'
    )
    assert result.status == 'fail'
    assert 'structure' in [issue['type'] for issue in result.issues]
    
    # Test missing code blocks
    result = await doc_validator.validate_documentation(
        doc_id='test_no_code',
        doc_type='api',
        content='# API Documentation\n\nNo code blocks here',
        format='markdown'
    )
    assert result.status == 'fail'
    assert 'content' in [issue['type'] for issue in result.issues]

@pytest.mark.asyncio
async def test_documentation_search_features(doc_search):
    """Test advanced documentation search features."""
    # Index test documents with metadata
    await doc_search.index_documentation(
        doc_id='test1',
        doc_type='api',
        title='User API',
        content='User management API documentation',
        metadata={'tags': ['users', 'api']}
    )
    
    await doc_search.index_documentation(
        doc_id='test2',
        doc_type='system',
        title='Auth System',
        content='Authentication system documentation',
        metadata={'tags': ['auth', 'system']}
    )
    
    # Test related documents
    related = await doc_search.get_related_documents('test1')
    assert len(related) > 0
    
    # Test metadata search
    results = await doc_search.search_documentation('users')
    assert any('users' in r.metadata.get('tags', []) for r in results)

@pytest.mark.asyncio
async def test_documentation_version_features(doc_version):
    """Test advanced documentation versioning features."""
    # Create versions with metadata
    version1 = await doc_version.create_version(
        doc_id='test_doc',
        content='Initial version',
        author='test_user',
        metadata={'status': 'draft'}
    )
    
    version2 = await doc_version.create_version(
        doc_id='test_doc',
        content='Updated version',
        author='test_user',
        metadata={'status': 'review'}
    )
    
    # Test metadata comparison
    comparison = await doc_version.compare_versions(
        'test_doc',
        '1.0.0',
        '1.0.1'
    )
    assert 'status' in comparison['metadata_changes']['modified']
    
    # Test version cleanup with metadata
    await doc_version.cleanup_versions('test_doc', keep_versions=1)
    history = doc_version.get_version_history('test_doc')
    assert len(history) == 1
    assert history[0].metadata['status'] == 'review'

def test_documentation_templates():
    """Test documentation templates."""
    # Test API template
    with open('automation/templates/api.md.j2') as f:
        api_template = f.read()
    assert 'API Documentation' in api_template
    assert 'Endpoints' in api_template
    
    # Test system template
    with open('automation/templates/system.md.j2') as f:
        system_template = f.read()
    assert 'System Documentation' in system_template
    assert 'Architecture' in system_template
    
    # Test code template
    with open('automation/templates/code.md.j2') as f:
        code_template = f.read()
    assert 'Code Documentation' in code_template
    assert 'Classes' in code_template
    
    # Test HTML export template
    with open('automation/templates/html_export.html.j2') as f:
        html_template = f.read()
    assert '<!DOCTYPE html>' in html_template
    assert '<title>' in html_template

def test_documentation_configuration():
    """Test documentation configuration."""
    # Test validation configuration
    assert 'validation' in config['documentation']
    assert 'min_length' in config['documentation']['validation']
    assert 'min_score' in config['documentation']['validation']
    
    # Test search configuration
    assert 'search' in config['documentation']
    assert 'engine' in config['documentation']['search']
    
    # Test versioning configuration
    assert 'versioning' in config['documentation']
    assert 'storage' in config['documentation']['versioning'] 