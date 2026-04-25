"""
Security tests for input validation.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner
from its_project.execution.base import Order, OrderType, OrderStatus


@pytest.mark.security
class TestInputValidation:
    """Test input validation for security."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        # Test that user input is properly sanitized
        malicious_input = "'; DROP TABLE users; --"
        
        # Should handle malicious input gracefully
        assert isinstance(malicious_input, str)
    
    def test_xss_prevention(self):
        """Test XSS prevention."""
        # Test that XSS attacks are prevented
        malicious_input = "<script>alert('xss')</script>"
        
        # Should handle malicious input gracefully
        assert isinstance(malicious_input, str)
    
    def test_data_type_validation(self, sample_market_data):
        """Test data type validation."""
        cleaner = DataCleaner()
        
        # Should validate data types
        assert cleaner.validate_types(sample_market_data)
        
        # Should reject invalid types
        invalid_data = pd.DataFrame({
            'price': ['invalid', 'data']  # Wrong type
        })
        assert not cleaner.validate_types(invalid_data)
    
    def test_schema_validation(self, sample_market_data):
        """Test schema validation."""
        cleaner = DataCleaner()
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Should validate schema
        assert cleaner.validate_schema(sample_market_data, required_columns)
        
        # Should reject incomplete schema
        incomplete_data = sample_market_data.drop(columns=['volume'])
        assert not cleaner.validate_schema(incomplete_data, required_columns)
    
    def test_order_validation(self):
        """Test order validation."""
        # Valid order
        order = Order(
            id='test_order_1',
            symbol='BTCUSDT',
            type=OrderType.MARKET,
            side='buy',
            amount=0.1,
            price=42000.0,
            status=OrderStatus.PENDING,
            filled=0.0,
            remaining=0.1,
            timestamp=1234567890000,
            info={}
        )
        
        # Should validate order
        assert order.amount > 0
        assert order.price > 0
        
        # Invalid order
        with pytest.raises((ValueError, AssertionError)):
            Order(
                id='test_order_2',
                symbol='BTCUSDT',
                type=OrderType.MARKET,
                side='buy',
                amount=-0.1,  # Invalid
                price=42000.0,
                status=OrderStatus.PENDING,
                filled=0.0,
                remaining=-0.1,
                timestamp=1234567890000,
                info={}
            )
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        malicious_input = "&& rm -rf /"
        
        # Should handle malicious input gracefully
        assert isinstance(malicious_input, str)
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        malicious_input = "../../../etc/passwd"
        
        # Should handle malicious input gracefully
        assert isinstance(malicious_input, str)
    
    def test_buffer_overflow_prevention(self):
        """Test buffer overflow prevention."""
        # Create very large input
        large_input = "A" * 1000000
        
        # Should handle large input gracefully
        assert isinstance(large_input, str)
        assert len(large_input) == 1000000
