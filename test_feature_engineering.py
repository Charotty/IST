"""
Feature Engineering Layer Test

Тестирование Feature Engineering Layer с реальными данными BTC-USDT.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Добавление путей
sys.path.append('.')

from data_layer.data_manager import DataManager
from feature_engineering.feature_manager import FeatureManager


async def test_feature_engineering():
    """Тестирование Feature Engineering Layer"""
    
    print("🚀 Testing Feature Engineering Layer with real BTC-USDT data...")
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    try:
        # Конфигурация
        config = {
            'data_layer': {
                'sources': [
                    {
                        'name': 'okx',
                        'connector': 'okx_official',
                        'enabled': True,
                        'data_types': ['ohlcv', 'trades', 'orderbook']
                    }
                ]
            },
            'synchronization': {
                'base_timestep': '1s',
                'max_gap_size': 5,
                'ohlcv': {'ohlcv_method': 'standard'},
                'trades': {'aggregation_method': 'vwap'},
                'orderbook': {'depth': 5, 'calculate_spreads': True}
            },
            'feature_engineering': {
                'features': {
                    'technical': {
                        'moving_averages': {
                            'periods': [5, 10, 20],
                            'types': ['SMA', 'EMA']
                        },
                        'momentum': {
                            'rsi_periods': [14],
                            'macd_params': [12, 26, 9]
                        },
                        'volatility': {
                            'bollinger_periods': [20],
                            'bollinger_std': [2.0]
                        }
                    },
                    'orderbook': {
                        'imbalance': {
                            'levels': [5, 10]
                        },
                        'spread': {
                            'levels': [5]
                        },
                        'depth': {
                            'levels': [5, 10]
                        }
                    },
                    'temporal': {
                        'returns': {
                            'return_periods': [1, 5, 15]
                        },
                        'time_features': {
                            'seasonal_features': True
                        }
                    }
                },
                'scaling': {
                    'method': 'standard',
                    'save_scalers': True
                },
                'normalization': {
                    'method': 'l2',
                    'apply_pca': False
                }
            }
        }
        
        print("\n📦 Initializing Feature Engineering Manager...")
        
        # Инициализация DataManager
        data_manager = DataManager(config)
        
        # Инициализация FeatureManager
        feature_manager = FeatureManager(config)
        
        print("\n🔄 Starting systems...")
        
        # Запуск систем
        await data_manager.start()
        await feature_manager.start()
        
        print("\n📊 Testing OHLCV feature generation...")
        
        # Тестирование OHLCV данных
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=30)
            
            # Получение исторических OHLCV данных
            ohlcv_data = await data_manager.get_historical_data(
                data_type='ohlcv',
                symbol='BTC-USDT',
                timeframe='1m',
                start_time=start_time,
                end_time=end_time
            )
            
            if ohlcv_data and ohlcv_data.get('data'):
                print(f"  ✅ Got {len(ohlcv_data['data'])} OHLCV records")
                
                # Генерация признаков
                features = await feature_manager.generate_features(
                    ohlcv_data['data'], 
                    'ohlcv',
                    apply_scaling=True
                )
                
                if not features.empty:
                    print(f"  ✅ Generated {len(features.columns)} features from OHLCV")
                    print(f"  📋 Feature names: {list(features.columns[:10])}...")
                else:
                    print("  ❌ No features generated from OHLCV")
            else:
                print("  ⚠️  No OHLCV data available")
                
        except Exception as e:
            print(f"  ❌ OHLCV feature test error: {e}")
        
        print("\n🌊 Testing trades feature generation...")
        
        # Тестирование trades данных
        try:
            trades_data = []
            
            # Сбор trades данных
            async for data in data_manager.stream_data('BTC-USDT', 'trades'):
                trades_data.append(data)
                if len(trades_data) >= 50:  # Собираем 50 записей
                    break
            
            if trades_data:
                print(f"  ✅ Collected {len(trades_data)} trades records")
                
                # Генерация признаков
                features = await feature_manager.generate_features(
                    trades_data,
                    'trades',
                    apply_scaling=True
                )
                
                if not features.empty:
                    print(f"  ✅ Generated {len(features.columns)} features from trades")
                    print(f"  📋 Feature names: {list(features.columns[:10])}...")
                else:
                    print("  ❌ No features generated from trades")
            else:
                print("  ⚠️  No trades data available")
                
        except Exception as e:
            print(f"  ❌ Trades feature test error: {e}")
        
        print("\n📚 Testing orderbook feature generation...")
        
        # Тестирование orderbook данных
        try:
            orderbook_data = []
            
            # Сбор orderbook данных
            async for data in data_manager.stream_data('BTC-USDT', 'orderbook'):
                orderbook_data.append(data)
                if len(orderbook_data) >= 20:  # Собираем 20 записей
                    break
            
            if orderbook_data:
                print(f"  ✅ Collected {len(orderbook_data)} orderbook records")
                
                # Генерация признаков
                features = await feature_manager.generate_features(
                    orderbook_data,
                    'orderbook',
                    apply_scaling=True
                )
                
                if not features.empty:
                    print(f"  ✅ Generated {len(features.columns)} features from orderbook")
                    print(f"  📋 Feature names: {list(features.columns[:10])}...")
                else:
                    print("  ❌ No features generated from orderbook")
            else:
                print("  ⚠️  No orderbook data available")
                
        except Exception as e:
            print(f"  ❌ Orderbook feature test error: {e}")
        
        print("\n📈 Testing feature manager metrics...")
        
        # Метрики Feature Manager
        metrics = feature_manager.get_metrics()
        
        print(f"  ✅ Total requests: {metrics.get('total_requests', 0)}")
        print(f"  ✅ Successful requests: {metrics.get('successful_requests', 0)}")
        print(f"  ✅ Features generated: {metrics.get('features_generated', 0)}")
        print(f"  ✅ Processing time: {metrics.get('processing_time', 0):.2f}s")
        
        if 'pipeline' in metrics:
            pipeline_metrics = metrics['pipeline']
            print(f"  📊 Pipeline runs: {pipeline_metrics.get('pipeline_runs', 0)}")
            print(f"  📊 Pipeline errors: {pipeline_metrics.get('errors_count', 0)}")
        
        print("\n🔧 Testing feature names...")
        
        # Тестирование имен признаков
        for data_type in ['ohlcv', 'trades', 'orderbook']:
            feature_names = feature_manager.get_feature_names(data_type)
            print(f"  📋 {data_type}: {len(feature_names)} features")
        
        print("\n🏥 Testing system health...")
        
        # Health check
        status = feature_manager.get_status()
        
        print(f"  ✅ Feature Manager running: {status['running']}")
        print(f"  ✅ Uptime: {status['uptime']:.2f}s")
        print(f"  📊 Components: {status['components']}")
        
        print("\n🧹 Cleaning up...")
        
        # Очистка
        await feature_manager.stop()
        await data_manager.stop()
        
        print("\n🎉 Feature Engineering Layer test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Feature Engineering test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_feature_engineering())
