import os
import pytest
from unittest.mock import MagicMock, patch
from squeeze.report.notifier import LineNotifier, EmailNotifier

@pytest.fixture
def mock_line_env(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test_token")
    monkeypatch.setenv("LINE_USER_ID", "test_user")

@pytest.fixture
def mock_email_env(monkeypatch):
    monkeypatch.setenv("SMTP_SERVER", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "sender@test.com")
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    monkeypatch.setenv("SMTP_RECIPIENT", "user1@test.com, user2@test.com")

def test_line_notifier_init_from_env(mock_line_env):
    notifier = LineNotifier()
    assert notifier.access_token == "test_token"
    assert notifier.user_id == "test_user"

def test_line_notifier_init_explicit():
    notifier = LineNotifier(access_token="explicit_token", user_id="explicit_user")
    assert notifier.access_token == "explicit_token"
    assert notifier.user_id == "explicit_user"

@patch('squeeze.report.notifier.MessagingApi')
@patch('squeeze.report.notifier.ApiClient')
@patch('squeeze.report.notifier.Configuration')
def test_send_line_summary_success(mock_config, mock_api_client, mock_messaging_api, mock_line_env):
    # Setup mocks
    mock_instance = mock_messaging_api.return_value
    
    notifier = LineNotifier()
    result = notifier.send_summary("Test message")
    
    assert result is True
    mock_messaging_api.assert_called_once()
    mock_instance.push_message.assert_called_once()

def test_send_line_summary_missing_config():
    with patch.dict(os.environ, {}, clear=True):
        notifier = LineNotifier()
        result = notifier.send_summary("Test message")
        assert result is False

def test_send_line_summary_empty_message(mock_line_env):
    notifier = LineNotifier()
    result = notifier.send_summary("")
    assert result is False

def test_email_notifier_init_from_env(mock_email_env):
    notifier = EmailNotifier()
    assert notifier.smtp_server == "smtp.test.com"
    assert notifier.smtp_port == 587
    assert notifier.username == "sender@test.com"
    assert notifier.password == "password"
    assert notifier.recipient_str == "user1@test.com, user2@test.com"

def test_email_notifier_get_recipient_list(mock_email_env):
    notifier = EmailNotifier()
    recipients = notifier._get_recipient_list()
    assert recipients == ["user1@test.com", "user2@test.com"]

@patch('smtplib.SMTP')
def test_send_email_success(mock_smtp, mock_email_env):
    notifier = EmailNotifier()
    result = notifier.send_email("Subject", "Body")
    
    assert result is True
    mock_smtp.assert_called_once_with("smtp.test.com", 587)
    
    instance = mock_smtp.return_value
    instance.starttls.assert_called_once()
    instance.login.assert_called_once_with("sender@test.com", "password")
    instance.sendmail.assert_called_once()
    instance.quit.assert_called_once()
    
    # Check if To header is set to sender (Bcc effect)
    args, kwargs = instance.sendmail.call_args
    # args[0] is from, args[1] is to_list, args[2] is msg_string
    assert args[0] == "sender@test.com"
    assert args[1] == ["user1@test.com", "user2@test.com"]
    assert 'To: sender@test.com' in args[2]
    assert 'user1@test.com' not in args[2].split('\nSubject:')[0] # user1 should not be in headers before subject
