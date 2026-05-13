"""
Version Manager

Управление версиями моделей.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
# import semver  # Заменено на встроенную функцию сравнения версий


class VersionManager:
    """Менеджер версий моделей"""
    
    def __init__(self):
        """Инициализация менеджера версий"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Паттерны версий
        self.version_patterns = {
            'semantic': r'^\d+\.\d+\.\d+$',  # 1.0.0
            'patch': r'^\d+\.\d+\.\d+-\w+$',  # 1.0.0-alpha
            'build': r'^\d+\.\d+\.\d+\.\d+$'  # 1.0.0.1
        }
    
    def parse_version(self, version_str: str) -> Dict[str, Any]:
        """
        Парсинг версии
        
        Args:
            version_str: Строка версии
            
        Returns:
            Dict: Компоненты версии
        """
        try:
            # Удаление префикса 'v' если есть
            version_str = version_str.lstrip('v')
            
            # Разделение на компоненты
            parts = version_str.split('.')
            
            if len(parts) >= 3:
                major = int(parts[0])
                minor = int(parts[1])
                
                # Обработка patch версии
                patch_part = parts[2]
                if '-' in patch_part:
                    patch, suffix = patch_part.split('-', 1)
                    patch = int(patch)
                    return {
                        'major': major,
                        'minor': minor,
                        'patch': patch,
                        'suffix': suffix,
                        'type': 'semantic',
                        'full': f"{major}.{minor}.{patch}-{suffix}"
                    }
                else:
                    patch = int(patch_part)
                    return {
                        'major': major,
                        'minor': minor,
                        'patch': patch,
                        'type': 'semantic',
                        'full': f"{major}.{minor}.{patch}"
                    }
            
            return {'raw': version_str, 'type': 'unknown'}
            
        except Exception as e:
            self.logger.error(f"Error parsing version {version_str}: {e}")
            return {'raw': version_str, 'type': 'error'}
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Сравнение версий
        
        Args:
            version1: Первая версия
            version2: Вторая версия
            
        Returns:
            int: -1 если version1 < version2, 0 если равны, 1 если version1 > version2
        """
        try:
            v1 = self.parse_version(version1)
            v2 = self.parse_version(version2)
            
            if v1['type'] == 'semantic' and v2['type'] == 'semantic':
                # Сравнение семантических версий
                if v1['major'] != v2['major']:
                    return v1['major'] - v2['major']
                if v1['minor'] != v2['minor']:
                    return v1['minor'] - v2['minor']
                if v1['patch'] != v2['patch']:
                    return v1['patch'] - v2['patch']
                return 0
            else:
                # Простое строковое сравнение
                if version1 < version2:
                    return -1
                elif version1 > version2:
                    return 1
                else:
                    return 0
                    
        except Exception as e:
            self.logger.error(f"Error comparing versions: {e}")
            return 0
    
    def increment_version(self, version: str, increment_type: str = 'patch') -> str:
        """
        Инкремент версии
        
        Args:
            version: Текущая версия
            increment_type: Тип инкремента (major, minor, patch)
            
        Returns:
            str: Новая версия
        """
        try:
            parsed = self.parse_version(version)
            
            if parsed['type'] != 'semantic':
                raise ValueError(f"Cannot increment non-semantic version: {version}")
            
            major = parsed['major']
            minor = parsed['minor']
            patch = parsed['patch']
            suffix = parsed.get('suffix', '')
            
            if increment_type == 'major':
                major += 1
                minor = 0
                patch = 0
                suffix = ''
            elif increment_type == 'minor':
                minor += 1
                patch = 0
                suffix = ''
            elif increment_type == 'patch':
                patch += 1
                suffix = ''
            else:
                raise ValueError(f"Invalid increment type: {increment_type}")
            
            new_version = f"{major}.{minor}.{patch}"
            if suffix:
                new_version += f"-{suffix}"
            
            return new_version
            
        except Exception as e:
            self.logger.error(f"Error incrementing version: {e}")
            return version
    
    def validate_version(self, version: str) -> bool:
        """
        Валидация версии
        
        Args:
            version: Версия для валидации
            
        Returns:
            bool: True если версия валидна
        """
        try:
            parsed = self.parse_version(version)
            return parsed['type'] != 'error'
        except Exception:
            return False
    
    def get_version_type(self, version: str) -> str:
        """
        Получение типа версии
        
        Args:
            version: Версия
            
        Returns:
            str: Тип версии
        """
        parsed = self.parse_version(version)
        return parsed.get('type', 'unknown')
    
    def is_compatible(self, required_version: str, current_version: str) -> bool:
        """
        Проверка совместимости версий
        
        Args:
            required_version: Требуемая версия
            current_version: Текущая версия
            
        Returns:
            bool: True если версии совместимы
        """
        try:
            # Простое правило: текущая версия должна быть >= требуемой
            return self.compare_versions(current_version, required_version) >= 0
        except Exception as e:
            self.logger.error(f"Error checking version compatibility: {e}")
            return False
    
    def get_latest_version(self, versions: List[str]) -> str:
        """
        Получение последней версии из списка
        
        Args:
            versions: Список версий
            
        Returns:
            str: Последняя версия
        """
        try:
            # Фильтрация валидных версий
            valid_versions = [v for v in versions if self.validate_version(v)]
            
            if not valid_versions:
                return "0.0.0"
            
            # Сортировка версий
            sorted_versions = sorted(valid_versions, key=self._version_key, reverse=True)
            
            return sorted_versions[0] if sorted_versions else "0.0.0"
            
        except Exception as e:
            self.logger.error(f"Error getting latest version: {e}")
            return "0.0.0"
    
    def _version_key(self, version: str) -> Tuple:
        """
        Ключ для сортировки версий
        
        Args:
            version: Версия
            
        Returns:
            Tuple: Ключ сортировки
        """
        try:
            parsed = self.parse_version(version)
            
            if parsed['type'] == 'semantic':
                return (parsed['major'], parsed['minor'], parsed['patch'])
            else:
                return (0, 0, 0)
        except Exception:
            return (0, 0, 0)
    
    def format_version_info(self, version: str) -> str:
        """
        Форматирование информации о версии
        
        Args:
            version: Версия
            
        Returns:
            str: Отформатированная информация
        """
        try:
            parsed = self.parse_version(version)
            
            if parsed['type'] == 'semantic':
                suffix = f"-{parsed['suffix']}" if parsed.get('suffix') else ''
                return f"v{parsed['major']}.{parsed['minor']}.{parsed['patch']}{suffix}"
            else:
                return f"v{version}"
                
        except Exception as e:
            self.logger.error(f"Error formatting version info: {e}")
            return f"v{version}"
    
    def get_version_history(self, versions: List[str]) -> List[Dict[str, Any]]:
        """
        Получение истории версий
        
        Args:
            versions: Список версий
            
        Returns:
            List: История версий с метаданными
        """
        try:
            version_history = []
            
            for version in versions:
                parsed = self.parse_version(version)
                
                version_info = {
                    'version': version,
                    'formatted': self.format_version_info(version),
                    'parsed': parsed,
                    'type': parsed['type'],
                    'created_at': None,  # Будет заполнено извне
                    'changes': []  # Будет заполнено извне
                }
                
                version_history.append(version_info)
            
            # Сортировка по версии
            version_history.sort(key=lambda x: self._version_key(x['version']), reverse=True)
            
            return version_history
            
        except Exception as e:
            self.logger.error(f"Error getting version history: {e}")
            return []
    
    def suggest_next_version(self, current_version: str, change_type: str = 'patch') -> str:
        """
        Предложение следующей версии
        
        Args:
            current_version: Текущая версия
            change_type: Тип изменений (major, minor, patch)
            
        Returns:
            str: Предложенная версия
        """
        try:
            if change_type == 'major':
                return self.increment_version(current_version, 'major')
            elif change_type == 'minor':
                return self.increment_version(current_version, 'minor')
            elif change_type == 'patch':
                return self.increment_version(current_version, 'patch')
            else:
                return self.increment_version(current_version, 'patch')
                
        except Exception as e:
            self.logger.error(f"Error suggesting next version: {e}")
            return current_version
    
    def get_version_range(self, min_version: str, max_version: str) -> List[str]:
        """
        Получение диапазона версий
        
        Args:
            min_version: Минимальная версия
            max_version: Максимальная версия
            
        Returns:
            List: Версии в диапазоне
        """
        try:
            # Генерация версий в диапазоне
            min_parsed = self.parse_version(min_version)
            max_parsed = self.parse_version(max_version)
            
            if min_parsed['type'] != 'semantic' or max_parsed['type'] != 'semantic':
                return [min_version, max_version]
            
            versions = []
            
            # Генерация всех версий в диапазоне
            for major in range(min_parsed['major'], max_parsed['major'] + 1):
                for minor in range(min_parsed['minor'], max_parsed['minor'] + 1):
                    for patch in range(min_parsed['patch'], max_parsed['patch'] + 1):
                        version = f"{major}.{minor}.{patch}"
                        
                        # Проверка границ
                        if (major == min_parsed['major'] and minor == min_parsed['minor'] and patch >= min_parsed['patch']) or \
                           (major == min_parsed['major'] and minor > min_parsed['minor']) or \
                           (major > min_parsed['major']):
                            versions.append(version)
                        
                        if (major == max_parsed['major'] and minor == max_parsed['minor'] and patch >= max_parsed['patch']) or \
                           (major == max_parsed['major'] and minor > max_parsed['minor']) or \
                           (major > max_parsed['major']):
                            break
                    if minor == max_parsed['minor'] and patch >= max_parsed['patch']:
                        break
                if major == max_parsed['major']:
                    break
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Error getting version range: {e}")
            return []
