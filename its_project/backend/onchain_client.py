"""
On-chain Data Client for Cryptocurrency Markets
Integrates with Glassnode API for blockchain metrics
"""

import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OnChainClient:
    """
    Client for fetching on-chain metrics from Glassnode API.
    
    Provides access to:
    - Net Exchange Flow
    - Whale Activity
    - Active Addresses
    - Transaction Volume
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize on-chain client.
        
        Args:
            api_key: Glassnode API key (optional for free tier)
        """
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get_net_exchange_flow(self, asset: str = "BTC", 
                              since: Optional[datetime] = None,
                              until: Optional[datetime] = None) -> List[Dict]:
        """
        Get net exchange flow (inflow - outflow).
        
        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            since: Start date
            until: End date
            
        Returns:
            List of {timestamp, value} dictionaries
        """
        endpoint = f"{self.base_url}/exchanges/net_flow_volume_mean"
        params = {
            "a": asset,
            "i": "24h",  # 24 hour intervals
            "s": int(since.timestamp()) if since else None,
            "u": int(until.timestamp()) if until else None
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching net exchange flow: {e}")
            return []
    
    def get_whale_activity(self, asset: str = "BTC",
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None) -> List[Dict]:
        """
        Get whale activity (large transactions > 100 BTC).
        
        Args:
            asset: Asset symbol
            since: Start date
            until: End date
            
        Returns:
            List of whale transaction data
        """
        # Glassnode endpoint for large transactions
        endpoint = f"{self.base_url}/transactions/transfers_volume_sum"
        params = {
            "a": asset,
            "i": "24h",
            "s": int(since.timestamp()) if since else None,
            "u": int(until.timestamp()) if until else None
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching whale activity: {e}")
            return []
    
    def get_active_addresses(self, asset: str = "BTC",
                           since: Optional[datetime] = None,
                           until: Optional[datetime] = None) -> List[Dict]:
        """
        Get number of active addresses.
        
        Args:
            asset: Asset symbol
            since: Start date
            until: End date
            
        Returns:
            List of {timestamp, value} dictionaries
        """
        endpoint = f"{self.base_url}/addresses/active_count"
        params = {
            "a": asset,
            "i": "24h",
            "s": int(since.timestamp()) if since else None,
            "u": int(until.timestamp()) if until else None
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching active addresses: {e}")
            return []
    
    def get_transaction_volume(self, asset: str = "BTC",
                             since: Optional[datetime] = None,
                             until: Optional[datetime] = None) -> List[Dict]:
        """
        Get transaction volume.
        
        Args:
            asset: Asset symbol
            since: Start date
            until: End date
            
        Returns:
            List of {timestamp, value} dictionaries
        """
        endpoint = f"{self.base_url}/transactions/volume_sum"
        params = {
            "a": asset,
            "i": "24h",
            "s": int(since.timestamp()) if since else None,
            "u": int(until.timestamp()) if until else None
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching transaction volume: {e}")
            return []
    
    def get_mvrv_zscore(self, asset: str = "BTC",
                       since: Optional[datetime] = None,
                       until: Optional[datetime] = None) -> List[Dict]:
        """
        Get MVRV Z-Score (Market Value to Realized Value).
        
        Args:
            asset: Asset symbol
            since: Start date
            until: End date
            
        Returns:
            List of {timestamp, value} dictionaries
        """
        endpoint = f"{self.base_url}/market/mvrv_z_score"
        params = {
            "a": asset,
            "i": "24h",
            "s": int(since.timestamp()) if since else None,
            "u": int(until.timestamp()) if until else None
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching MVRV Z-Score: {e}")
            return []
    
    def get_all_metrics(self, asset: str = "BTC", days: int = 30) -> Dict[str, List[Dict]]:
        """
        Get all on-chain metrics for a given asset.
        
        Args:
            asset: Asset symbol
            days: Number of days to fetch
            
        Returns:
            Dictionary with all metrics
        """
        since = datetime.now() - timedelta(days=days)
        until = datetime.now()
        
        return {
            "net_exchange_flow": self.get_net_exchange_flow(asset, since, until),
            "whale_activity": self.get_whale_activity(asset, since, until),
            "active_addresses": self.get_active_addresses(asset, since, until),
            "transaction_volume": self.get_transaction_volume(asset, since, until),
            "mvrv_zscore": self.get_mvrv_zscore(asset, since, until)
        }
