"""
Unit tests for CLI interface.
"""
import pytest
from click.testing import CliRunner
from its_project.cli.main import cli


@pytest.mark.unit
@pytest.mark.interface_layer
class TestCLIInterface:
    """Test CLI interface functionality."""
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Commands:' in result.output
    
    def test_status_command(self):
        """Test status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        
        # Should not fail even if system not running
        assert result.exit_code == 0
    
    def test_order_command(self):
        """Test order command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['order', '--symbol', 'BTCUSDT', '--side', 'buy', '--amount', '0.001'])
        
        # Should not fail (may return error if system not running)
        assert result.exit_code == 0
    
    def test_orders_command(self):
        """Test orders command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['orders'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_portfolio_command(self):
        """Test portfolio command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_backtest_command(self):
        """Test backtest command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['backtest'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_config_command(self):
        """Test config command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['config'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_logs_command(self):
        """Test logs command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['logs'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_train_command(self):
        """Test train command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_version_command(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['version'])
        
        # Should not fail
        assert result.exit_code == 0
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        runner = CliRunner()
        result = runner.invoke(cli, ['invalid_command'])
        
        # Should show error
        assert result.exit_code != 0 or 'No such command' in result.output
    
    def test_help_message(self):
        """Test help message display."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert 'Commands:' in result.output
        assert 'status' in result.output
        assert 'order' in result.output
        assert 'portfolio' in result.output
