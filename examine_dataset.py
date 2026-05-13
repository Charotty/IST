#!/usr/bin/env python3
"""
Скрипт для изучения сгенерированного датасета

Показывает структуру и содержимое данных, загруженных через API
"""

import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def examine_parquet_file(file_path):
    """Изучение одного parquet файла"""
    try:
        df = pd.read_parquet(file_path)
        
        print(f"\n📄 File: {os.path.basename(file_path)}")
        print(f"   Shape: {df.shape}")
        print(f"   Size: {os.path.getsize(file_path) / 1024:.2f} KB")
        print(f"   Columns: {list(df.columns)}")
        
        if not df.empty:
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            print(f"   Data types:\n{df.dtypes}")
            
            # Показываем первые несколько строк
            print(f"   Sample data:")
            print(df.head(3).to_string())
            
            # Базовая статистика для числовых колонок
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"   Basic statistics:")
                print(df[numeric_cols].describe().to_string())
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def main():
    """Главная функция"""
    print("=" * 80)
    print("DATASET EXAMINATION TOOL")
    print("=" * 80)
    
    data_dir = Path("./data/parquet")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    print(f"📁 Examining data in: {data_dir.absolute()}")
    
    # Изучаем OHLCV данные
    ohlcv_dir = data_dir / "ohlcv"
    if ohlcv_dir.exists():
        print(f"\n" + "=" * 60)
        print("OHLCV DATA")
        print("=" * 60)
        
        # Берем несколько примеров разных символов и таймфреймов
        sample_files = []
        for symbol in ['BTC-USDT', 'ETH-USDT']:
            for timeframe in ['1m', '1h', '1d']:
                file_path = ohlcv_dir / f"{symbol}_{timeframe}.parquet"
                if file_path.exists():
                    sample_files.append(file_path)
                    if len(sample_files) >= 3:  # Ограничиваем количество примеров
                        break
            if len(sample_files) >= 3:
                break
        
        for file_path in sample_files:
            examine_parquet_file(file_path)
    
    # Изучаем данные о сделках
    trades_dir = data_dir / "trades"
    if trades_dir.exists():
        print(f"\n" + "=" * 60)
        print("TRADES DATA")
        print("=" * 60)
        
        for file_path in trades_dir.glob("*.parquet"):
            examine_parquet_file(file_path)
            break  # Показываем только один файл для примера
    
    # Изучаем данные orderbook
    orderbook_dir = data_dir / "orderbook"
    if orderbook_dir.exists():
        print(f"\n" + "=" * 60)
        print("ORDERBOOK DATA")
        print("=" * 60)
        
        for file_path in orderbook_dir.glob("*.parquet"):
            examine_parquet_file(file_path)
            break  # Показываем только один файл для примера
    
    # Статистика по всем файлам
    print(f"\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    total_files = 0
    total_size = 0
    file_types = {}
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.parquet'):
                total_files += 1
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                # Определяем тип данных
                if 'ohlcv' in root:
                    file_type = 'ohlcv'
                elif 'trades' in root:
                    file_type = 'trades'
                elif 'orderbook' in root:
                    file_type = 'orderbook'
                else:
                    file_type = 'other'
                
                if file_type not in file_types:
                    file_types[file_type] = {'count': 0, 'size': 0}
                
                file_types[file_type]['count'] += 1
                file_types[file_type]['size'] += file_size
    
    print(f"📊 Total files: {total_files}")
    print(f"💾 Total size: {total_size / (1024*1024):.2f} MB")
    
    print(f"\n📈 Files by type:")
    for file_type, stats in file_types.items():
        print(f"   {file_type}: {stats['count']} files, {stats['size'] / (1024*1024):.2f} MB")
    
    # Показываем структуру директорий
    print(f"\n📂 Directory structure:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/: {len(files)} files")
    
    print(f"\n✅ Dataset examination completed!")
    print(f"💡 You can now use this data for analysis and model training")

if __name__ == "__main__":
    main()
