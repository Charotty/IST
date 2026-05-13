#!/usr/bin/env python3
"""
Test script to check imports and basic functionality
"""

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    import yaml
    print("✅ yaml imported successfully")
except ImportError as e:
    print("❌ yaml import failed:", e)

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print("❌ pandas import failed:", e)

try:
    import asyncio
    print("✅ asyncio imported successfully")
except ImportError as e:
    print("❌ asyncio import failed:", e)

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported successfully")
except ImportError as e:
    print("❌ dotenv import failed:", e)

# Test importing our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_layer.data_manager import DataManager
    print("✅ DataManager imported successfully")
except ImportError as e:
    print("❌ DataManager import failed:", e)

try:
    from data_layer.connectors.okx_connector import OKXConnector
    print("✅ OKXConnector imported successfully")
except ImportError as e:
    print("❌ OKXConnector import failed:", e)

print("\nAll basic tests completed!")
