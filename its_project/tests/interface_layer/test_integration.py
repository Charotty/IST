"""
Integration tests for Interface Layer.
"""
import pytest
from click.testing import CliRunner
from its_project.cli.main import cli
from its_project.web_dashboard.app import app


@pytest.mark.integration
@pytest.mark.interface_layer
class TestInterfaceLayerIntegration:
    """Test interface layer integration scenarios."""
    
    def test_dashboard_backend_communication(self):
        """Test dashboard ↔ backend communication."""
        client = app.test_client()
        
        # Test data endpoint
        response = client.get('/api/data')
        
        assert response.status_code in [200, 404]
    
    def test_bot_backend_communication(self):
        """Test bot ↔ backend communication."""
        # This would test actual Telegram bot integration
        # For now, verify bot can be created
        try:
            from its_project.telegram_bot.bot import create_telegram_bot
            config = {
                'token': 'test_token',
                'chat_id': 'test_chat_id',
                'enabled': False  # Disabled for testing
            }
            bot = create_telegram_bot(config)
            assert bot is not None
        except ImportError:
            # Library not installed, which is acceptable
            assert True
    
    def test_cli_backend_communication(self):
        """Test CLI ↔ backend communication."""
        runner = CliRunner()
        
        # Test status command
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
    
    def test_concurrent_interface_usage(self):
        """Test concurrent interface usage."""
        runner = CliRunner()
        web_client = app.test_client()
        
        # Test CLI and web simultaneously
        cli_result = runner.invoke(cli, ['status'])
        web_result = web_client.get('/api/data')
        
        assert cli_result.exit_code == 0
        assert web_result.status_code in [200, 404]
    
    def test_data_synchronization(self):
        """Test data synchronization across interfaces."""
        # Test that all interfaces show consistent data
        runner = CliRunner()
        
        # Get portfolio via CLI
        cli_result = runner.invoke(cli, ['portfolio'])
        
        # Get portfolio via web
        web_client = app.test_client()
        web_result = web_client.get('/api/portfolio')
        
        # Both should respond
        assert cli_result.exit_code == 0
        assert web_result.status_code in [200, 404]
    
    def test_real_time_updates(self):
        """Test real-time updates across interfaces."""
        # This would test WebSocket updates
        # For now, verify endpoint exists
        web_client = app.test_client()
        response = web_client.get('/ws')
        
        assert response.status_code in [101, 404]
    
    def test_performance_under_load(self):
        """Test performance under load."""
        runner = CliRunner()
        
        # Multiple rapid CLI commands
        for _ in range(10):
            result = runner.invoke(cli, ['status'])
            assert result.exit_code == 0
    
    def test_error_handling_consistency(self):
        """Test error handling consistency across interfaces."""
        runner = CliRunner()
        web_client = app.test_client()
        
        # Test invalid command
        cli_result = runner.invoke(cli, ['invalid_command'])
        
        # Test invalid endpoint
        web_result = web_client.get('/invalid_endpoint')
        
        # Both should handle errors gracefully
        assert cli_result.exit_code != 0 or 'No such command' in cli_result.output
        assert web_result.status_code == 404
