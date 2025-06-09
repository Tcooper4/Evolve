import pytest
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import os
import json
from unittest.mock import Mock, patch
import markdown

from automation.notifications.notification_service import (
    Notification,
    NotificationType,
    NotificationPriority,
    NotificationChannel
)
from automation.notifications.templates import (
    EmailTemplate,
    SlackTemplate,
    WebhookTemplate
)

@pytest.fixture
def jinja_env():
    """Create a Jinja2 environment for testing templates."""
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    return Environment(loader=FileSystemLoader(template_dir))

@pytest.fixture
def sample_notification():
    """Create a sample notification for testing templates."""
    return Notification(
        id="test_notification",
        title="Test Notification",
        message="This is a test notification",
        type=NotificationType.INFO,
        priority=NotificationPriority.MEDIUM,
        channel=NotificationChannel.EMAIL,
        recipient="test@example.com",
        metadata={
            "source": "test",
            "timestamp": datetime.utcnow().isoformat()
        },
        created_at=datetime.utcnow(),
        sent_at=datetime.utcnow(),
        status="sent",
        retry_count=0,
        max_retries=3
    )

def test_email_template(jinja_env, sample_notification):
    """Test the email notification template."""
    template = jinja_env.get_template("email/notification.html")
    rendered = template.render(
        title=sample_notification.title,
        message=sample_notification.message,
        type=sample_notification.type,
        priority=sample_notification.priority,
        metadata=sample_notification.metadata,
        sent_at=sample_notification.sent_at.isoformat()
    )
    
    # Verify template rendering
    assert sample_notification.title in rendered
    assert sample_notification.message in rendered
    assert sample_notification.type in rendered
    assert sample_notification.priority in rendered
    assert "Automation System" in rendered
    
    # Verify HTML structure
    assert "<!DOCTYPE html>" in rendered
    assert "<html>" in rendered
    assert "<head>" in rendered
    assert "<body>" in rendered
    assert "<div class=\"container\">" in rendered
    assert "<div class=\"notification\">" in rendered

def test_slack_template(jinja_env, sample_notification):
    """Test the Slack notification template."""
    template = jinja_env.get_template("slack/notification.json")
    rendered = template.render(
        recipient=sample_notification.recipient,
        title=sample_notification.title,
        message=sample_notification.message,
        type=sample_notification.type,
        priority=sample_notification.priority,
        color="#2196F3",
        created_at_timestamp=int(sample_notification.created_at.timestamp()),
        metadata=sample_notification.metadata
    )
    
    # Parse rendered JSON
    data = json.loads(rendered)
    
    # Verify template rendering
    assert data["channel"] == sample_notification.recipient
    assert len(data["attachments"]) == 1
    attachment = data["attachments"][0]
    assert attachment["title"] == sample_notification.title
    assert attachment["text"] == sample_notification.message
    assert attachment["color"] == "#2196F3"
    
    # Verify fields
    fields = attachment["fields"]
    assert any(f["title"] == "Priority" and f["value"] == sample_notification.priority for f in fields)
    assert any(f["title"] == "Type" and f["value"] == sample_notification.type for f in fields)

def test_webhook_template(jinja_env, sample_notification):
    """Test the webhook notification template."""
    template = jinja_env.get_template("webhook/notification.json")
    rendered = template.render(
        id=sample_notification.id,
        title=sample_notification.title,
        message=sample_notification.message,
        type=sample_notification.type,
        priority=sample_notification.priority,
        channel=sample_notification.channel,
        recipient=sample_notification.recipient,
        created_at=sample_notification.created_at.isoformat(),
        sent_at=sample_notification.sent_at.isoformat(),
        metadata=sample_notification.metadata,
        status=sample_notification.status,
        retry_count=sample_notification.retry_count,
        max_retries=sample_notification.max_retries
    )
    
    # Parse rendered JSON
    data = json.loads(rendered)
    
    # Verify template rendering
    assert data["id"] == sample_notification.id
    assert data["title"] == sample_notification.title
    assert data["message"] == sample_notification.message
    assert data["type"] == sample_notification.type
    assert data["priority"] == sample_notification.priority
    assert data["channel"] == sample_notification.channel
    assert data["recipient"] == sample_notification.recipient
    assert data["status"] == sample_notification.status
    assert data["retry_count"] == sample_notification.retry_count
    assert data["max_retries"] == sample_notification.max_retries
    assert data["metadata"] == sample_notification.metadata

def test_email_template_with_different_types(jinja_env, sample_notification):
    """Test the email template with different notification types."""
    types = [
        (NotificationType.INFO, "#e3f2fd"),
        (NotificationType.WARNING, "#fff3e0"),
        (NotificationType.ERROR, "#ffebee"),
        (NotificationType.SUCCESS, "#e8f5e9"),
        (NotificationType.ALERT, "#fce4ec")
    ]
    
    for type_, color in types:
        sample_notification.type = type_
        template = jinja_env.get_template("email/notification.html")
        rendered = template.render(
            title=sample_notification.title,
            message=sample_notification.message,
            type=sample_notification.type,
            priority=sample_notification.priority,
            metadata=sample_notification.metadata,
            sent_at=sample_notification.sent_at.isoformat()
        )
        
        # Verify type-specific styling
        assert f"class=\"notification {type_}\"" in rendered

def test_email_template_with_different_priorities(jinja_env, sample_notification):
    """Test the email template with different notification priorities."""
    priorities = [
        (NotificationPriority.LOW, "priority-low"),
        (NotificationPriority.MEDIUM, "priority-medium"),
        (NotificationPriority.HIGH, "priority-high"),
        (NotificationPriority.CRITICAL, "priority-critical")
    ]
    
    for priority, class_name in priorities:
        sample_notification.priority = priority
        template = jinja_env.get_template("email/notification.html")
        rendered = template.render(
            title=sample_notification.title,
            message=sample_notification.message,
            type=sample_notification.type,
            priority=sample_notification.priority,
            metadata=sample_notification.metadata,
            sent_at=sample_notification.sent_at.isoformat()
        )
        
        # Verify priority-specific styling
        assert f"class=\"priority {class_name}\"" in rendered 

@pytest.fixture
def template_test_dir(tmp_path):
    """Create a temporary directory for template testing."""
    return tmp_path

@pytest.fixture
def email_template_dir(template_test_dir):
    """Create email template directory."""
    email_dir = template_test_dir / "email"
    email_dir.mkdir()
    return email_dir

@pytest.fixture
def slack_template_dir(template_test_dir):
    """Create Slack template directory."""
    slack_dir = template_test_dir / "slack"
    slack_dir.mkdir()
    return slack_dir

@pytest.fixture
def webhook_template_dir(template_test_dir):
    """Create webhook template directory."""
    webhook_dir = template_test_dir / "webhook"
    webhook_dir.mkdir()
    return webhook_dir

@pytest.fixture
def email_template(email_template_dir):
    """Create email template."""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .notification { padding: 20px; border: 1px solid #ddd; }
            .title { font-size: 24px; color: #333; }
            .message { font-size: 16px; color: #666; }
            .metadata { font-size: 14px; color: #999; }
            .footer { font-size: 12px; color: #999; }
        </style>
    </head>
    <body>
        <div class="notification">
            <h1 class="title">{{ title }}</h1>
            <p class="message">{{ message }}</p>
            <div class="metadata">
                <p>Type: {{ type }}</p>
                <p>Priority: {{ priority }}</p>
                {% if metadata %}
                <p>Metadata: {{ metadata | tojson }}</p>
                {% endif %}
            </div>
            <div class="footer">
                <p>Sent at: {{ timestamp }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    template_path = email_template_dir / "notification.html"
    with open(template_path, "w") as f:
        f.write(template)
    
    return template_path

@pytest.fixture
def slack_template(slack_template_dir):
    """Create Slack template."""
    template = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "{{ title }}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "{{ message }}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*Type:*\n{{ type }}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Priority:*\n{{ priority }}"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Sent at: {{ timestamp }}"
                    }
                ]
            }
        ]
    }
    
    template_path = slack_template_dir / "notification.json"
    with open(template_path, "w") as f:
        json.dump(template, f)
    
    return template_path

@pytest.fixture
def webhook_template(webhook_template_dir):
    """Create webhook template."""
    template = {
        "id": "{{ id }}",
        "title": "{{ title }}",
        "message": "{{ message }}",
        "type": "{{ type }}",
        "priority": "{{ priority }}",
        "metadata": {{ metadata | tojson }},
        "timestamp": "{{ timestamp }}"
    }
    
    template_path = webhook_template_dir / "notification.json"
    with open(template_path, "w") as f:
        json.dump(template, f)
    
    return template_path

def test_email_template_rendering(email_template):
    """Test rendering of email template."""
    template = EmailTemplate(email_template)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Render template
    rendered = template.render(data)
    
    # Verify rendered content
    assert data["title"] in rendered
    assert data["message"] in rendered
    assert data["type"] in rendered
    assert data["priority"] in rendered
    assert data["metadata"]["key"] in rendered
    assert data["timestamp"] in rendered
    
    # Verify HTML structure
    assert "<!DOCTYPE html>" in rendered
    assert "<html>" in rendered
    assert "<head>" in rendered
    assert "<body>" in rendered
    assert "<style>" in rendered
    assert "</style>" in rendered
    assert "</head>" in rendered
    assert "</body>" in rendered
    assert "</html>" in rendered

def test_slack_template_rendering(slack_template):
    """Test rendering of Slack template."""
    template = SlackTemplate(slack_template)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Render template
    rendered = template.render(data)
    
    # Parse rendered JSON
    rendered_json = json.loads(rendered)
    
    # Verify rendered content
    assert rendered_json["blocks"][0]["text"]["text"] == data["title"]
    assert rendered_json["blocks"][1]["text"]["text"] == data["message"]
    assert rendered_json["blocks"][2]["fields"][0]["text"] == f"*Type:*\n{data['type']}"
    assert rendered_json["blocks"][2]["fields"][1]["text"] == f"*Priority:*\n{data['priority']}"
    assert rendered_json["blocks"][3]["elements"][0]["text"] == f"Sent at: {data['timestamp']}"

def test_webhook_template_rendering(webhook_template):
    """Test rendering of webhook template."""
    template = WebhookTemplate(webhook_template)
    
    # Create test data
    data = {
        "id": "test-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Render template
    rendered = template.render(data)
    
    # Parse rendered JSON
    rendered_json = json.loads(rendered)
    
    # Verify rendered content
    assert rendered_json["id"] == data["id"]
    assert rendered_json["title"] == data["title"]
    assert rendered_json["message"] == data["message"]
    assert rendered_json["type"] == data["type"]
    assert rendered_json["priority"] == data["priority"]
    assert rendered_json["metadata"] == data["metadata"]
    assert rendered_json["timestamp"] == data["timestamp"]

def test_template_missing_variables(email_template):
    """Test handling of missing template variables."""
    template = EmailTemplate(email_template)
    
    # Create test data with missing variables
    data = {
        "title": "Test Notification",
        "message": "This is a test notification"
    }
    
    # Render template
    rendered = template.render(data)
    
    # Verify rendered content
    assert data["title"] in rendered
    assert data["message"] in rendered
    assert "Type:" in rendered
    assert "Priority:" in rendered
    assert "Metadata:" in rendered
    assert "Sent at:" in rendered

def test_template_invalid_json(slack_template):
    """Test handling of invalid JSON template."""
    template = SlackTemplate(slack_template)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Corrupt template file
    with open(slack_template, "w") as f:
        f.write("invalid json")
    
    # Verify template loading fails
    with pytest.raises(json.JSONDecodeError):
        template.render(data)

def test_template_custom_filters(email_template):
    """Test custom template filters."""
    template = EmailTemplate(email_template)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Add custom filter
    template.env.filters["uppercase"] = lambda x: x.upper()
    
    # Modify template to use custom filter
    with open(email_template, "r") as f:
        template_content = f.read()
    
    template_content = template_content.replace("{{ title }}", "{{ title | uppercase }}")
    
    with open(email_template, "w") as f:
        f.write(template_content)
    
    # Render template
    rendered = template.render(data)
    
    # Verify custom filter
    assert data["title"].upper() in rendered

def test_template_inheritance(email_template_dir):
    """Test template inheritance."""
    # Create base template
    base_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .notification { padding: 20px; border: 1px solid #ddd; }
            .title { font-size: 24px; color: #333; }
            .message { font-size: 16px; color: #666; }
            .metadata { font-size: 14px; color: #999; }
            .footer { font-size: 12px; color: #999; }
        </style>
    </head>
    <body>
        {% block content %}{% endblock %}
    </body>
    </html>
    """
    
    base_path = email_template_dir / "base.html"
    with open(base_path, "w") as f:
        f.write(base_template)
    
    # Create child template
    child_template = """
    {% extends "base.html" %}
    {% block content %}
    <div class="notification">
        <h1 class="title">{{ title }}</h1>
        <p class="message">{{ message }}</p>
        <div class="metadata">
            <p>Type: {{ type }}</p>
            <p>Priority: {{ priority }}</p>
            {% if metadata %}
            <p>Metadata: {{ metadata | tojson }}</p>
            {% endif %}
        </div>
        <div class="footer">
            <p>Sent at: {{ timestamp }}</p>
        </div>
    </div>
    {% endblock %}
    """
    
    child_path = email_template_dir / "notification.html"
    with open(child_path, "w") as f:
        f.write(child_template)
    
    # Create template
    template = EmailTemplate(child_path)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Render template
    rendered = template.render(data)
    
    # Verify rendered content
    assert data["title"] in rendered
    assert data["message"] in rendered
    assert data["type"] in rendered
    assert data["priority"] in rendered
    assert data["metadata"]["key"] in rendered
    assert data["timestamp"] in rendered
    
    # Verify HTML structure
    assert "<!DOCTYPE html>" in rendered
    assert "<html>" in rendered
    assert "<head>" in rendered
    assert "<body>" in rendered
    assert "<style>" in rendered
    assert "</style>" in rendered
    assert "</head>" in rendered
    assert "</body>" in rendered
    assert "</html>" in rendered

def test_template_macros(email_template):
    """Test template macros."""
    template = EmailTemplate(email_template)
    
    # Create test data
    data = {
        "title": "Test Notification",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "value"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Add macro
    template.env.globals["format_timestamp"] = lambda x: datetime.fromisoformat(x).strftime("%Y-%m-%d %H:%M:%S")
    
    # Modify template to use macro
    with open(email_template, "r") as f:
        template_content = f.read()
    
    template_content = template_content.replace("{{ timestamp }}", "{{ format_timestamp(timestamp) }}")
    
    with open(email_template, "w") as f:
        f.write(template_content)
    
    # Render template
    rendered = template.render(data)
    
    # Verify macro
    formatted_timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    assert formatted_timestamp in rendered

def test_template_security(email_template):
    """Test template security."""
    template = EmailTemplate(email_template)
    
    # Create test data with potentially dangerous content
    data = {
        "title": "<script>alert('xss')</script>",
        "message": "This is a test notification",
        "type": NotificationType.INFO,
        "priority": NotificationPriority.MEDIUM,
        "metadata": {
            "key": "<script>alert('xss')</script>"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Render template
    rendered = template.render(data)
    
    # Verify XSS prevention
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "alert('xss')" not in rendered
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in rendered 