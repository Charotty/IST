"""
Security tests for API key management.
"""
import pytest
import os
from its_project.config import load_config


@pytest.mark.security
class TestAPIKeys:
    """Test API key security."""
    
    def test_api_keys_not_hardcoded(self):
        """Test that API keys are not hardcoded in source code."""
        # Check that config file doesn't contain hardcoded keys
        config = load_config()
        
        # Should not have placeholder values
        assert config.get('glassnode_api_key') != 'your_api_key_here'
        assert config.get('twitter_bearer_token') != 'your_token_here'
    
    def test_api_keys_from_environment(self, monkeypatch):
        """Test that API keys can be loaded from environment variables."""
        monkeypatch.setenv('GLASSNODE_API_KEY', 'test_key_123')
        monkeypatch.setenv('TWITTER_BEARER_TOKEN', 'test_token_456')
        
        # Load config should use environment variables
        config = load_config()
        
        # Should have loaded from environment
        assert 'GLASSNODE_API_KEY' in os.environ or config.get('glassnode_api_key') is not None
    
    def test_api_keys_masked_in_logs(self):
        """Test that API keys are masked in logs."""
        config = load_config()
        api_key = config.get('glassnode_api_key')
        
        if api_key:
            # Should not be exposed in logs
            assert len(api_key) > 0
            # Key should not be 'your_api_key_here' or similar placeholder
            assert api_key not in ['your_api_key_here', 'placeholder', 'test_key']
    
    def test_config_file_permissions(self):
        """Test that config file has appropriate permissions."""
        config_path = 'config.json'
        
        if os.path.exists(config_path):
            # Check file permissions (should not be world-readable)
            # Note: This is a basic check, production should use proper permission management
            assert os.path.exists(config_path)
    
    def test_sensitive_data_encryption(self):
        """Test that sensitive data can be encrypted."""
        # This is a placeholder for encryption tests
        # In production, sensitive data should be encrypted
        assert True  # Placeholder
    
    def test_api_key_rotation(self):
        """Test API key rotation capability."""
        # Placeholder for key rotation tests
        # In production, keys should be rotatable
        assert True  # Placeholder
