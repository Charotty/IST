# GUI Layer

## Назначение

Графический интерфейс пользователя для мониторинга и управления торговой системой в реальном времени.

## Основные компоненты

### Live Chart

График цены в реальном времени:
- Кандалы OHLCV
- Технические индикаторы
- Торговые сигналы
- Уровни стоп-лосс и тейк-профит

### Signals Panel

Панель торговых сигналов:
- Текущие сигналы BUY/SELL/HOLD
- Вероятности и уверенность
- История сигналов
- Фильтрация по времени

### Orders Table

Таблица ордеров:
- Активные ордера
- Исполненные сделки
- Статусы ордеров
- Детали исполнения

### Metrics Dashboard

Аналитическая панель:
- Текущий PnL
- Доходность
- Просадка
- Sharpe Ratio
- Win Rate

### Model Monitor

Мониторинг моделей:
- Статус моделей
- Производительность
- Точность предсказаний
- Вес ансамбля

### Logs

Системные события:
- Торговые операции
- Ошибки и предупреждения
- Системные статусы
- Фильтрация по уровню

## Структура модуля

```
gui/
├── __init__.py
├── main_window/
│   ├── __init__.py
│   ├── main_window.py         # Главное окно
│   ├── menu_bar.py           # Меню приложения
│   ├── status_bar.py         # Статусная строка
│   └── toolbar.py           # Панель инструментов
├── charts/
│   ├── __init__.py
│   ├── price_chart.py       # График цены
│   ├── volume_chart.py       # График объема
│   ├── indicator_chart.py   # График индикаторов
│   └── signal_chart.py      # График сигналов
├── panels/
│   ├── __init__.py
│   ├── signals_panel.py     # Панель сигналов
│   ├── orders_panel.py      # Панель ордеров
│   ├── positions_panel.py   # Панель позиций
│   └── metrics_panel.py    # Панель метрик
├── dialogs/
│   ├── __init__.py
│   ├── settings_dialog.py   # Диалог настроек
│   ├── strategy_dialog.py   # Диалог стратегии
│   ├── risk_dialog.py       # Диалог рисков
│   └── about_dialog.py     # Диалог о программе
├── widgets/
│   ├── __init__.py
│   ├── custom_widgets.py    # Кастомные виджеты
│   ├── table_widgets.py     # Табличные виджеты
│   ├── chart_widgets.py     # Графические виджеты
│   └── control_widgets.py   # Элементы управления
├── data/
│   ├── __init__.py
│   ├── data_manager.py      # Управление данными GUI
│   ├── update_manager.py    # Обновление данных
│   └── cache_manager.py    # Кэширование данных
├── themes/
│   ├── __init__.py
│   ├── dark_theme.py       # Темная тема
│   ├── light_theme.py      # Светлая тема
│   └── custom_theme.py     # Кастомная тема
└── app.py                 # Главное приложение
```

## Ключевые компоненты

### MainWindow

Главное окно приложения:
- Компоновка интерфейса
- Управление меню и панелями
- Обработка событий
- Координация обновлений

### PriceChart

Интерактивный график цены:
- Реальное время обновления
- Масштабирование и панорамирование
- Наложение индикаторов
- Торговые сигналы

### SignalsPanel

Панель управления сигналами:
- Отображение текущих сигналов
- Фильтрация и сортировка
- Экспорт сигналов
- Исторические данные

### MetricsDashboard

Панель метрик:
- Real-time обновление
- Графическое представление
- Сравнение с бенчмарками
- Экспорт отчетов

## PyQt6 Архитектура

### Основные принципы

- **MVC Pattern**: разделение логики и представления
- **Signal-Slot**: асинхронная коммуникация
- **Thread Safety**: безопасная работа с потоками
- **Resource Management**: эффективное использование ресурсов

### Пример компонента

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, pyqtSignal

class PriceChart(QWidget):
    price_updated = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.price_label = QLabel("Price: --")
        layout.addWidget(self.price_label)
        self.setLayout(layout)
    
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_price)
        self.timer.start(1000)  # 1 second
    
    def update_price(self):
        # Получение цены от data manager
        price = self.get_current_price()
        self.price_label.setText(f"Price: {price:.2f}")
        self.price_updated.emit(price)
```

## Реальное время обновления

### Data Manager

```python
class GUIDataManager:
    def __init__(self):
        self.subscribers = {}
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.broadcast_updates)
        self.update_timer.start(100)  # 100ms
    
    def subscribe(self, widget, data_type):
        if data_type not in self.subscribers:
            self.subscribers[data_type] = []
        self.subscribers[data_type].append(widget)
    
    def broadcast_updates(self):
        for data_type, widgets in self.subscribers.items():
            data = self.get_data(data_type)
            for widget in widgets:
                widget.update_data(data)
```

### Thread Safety

```python
from PyQt6.QtCore import QThread, pyqtSignal

class DataUpdater(QThread):
    data_updated = pyqtSignal(dict)
    
    def run(self):
        while True:
            # Получение данных из системы
            data = self.fetch_system_data()
            
            # Безопасная передача в GUI поток
            self.data_updated.emit(data)
            
            self.msleep(100)  # 100ms delay
```

## Визуализация данных

### Matplotlib Integration

```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TradingChart(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 6))
        super().__init__(self.figure)
        
        self.ax = self.figure.add_subplot(111)
        self.setup_chart()
    
    def setup_chart(self):
        self.ax.set_title("Price Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)
    
    def update_chart(self, data):
        self.ax.clear()
        self.ax.plot(data['timestamp'], data['price'], 'b-')
        self.ax.plot(data['timestamp'], data['signals'], 'ro', markersize=3)
        self.draw()
```

### Custom Widgets

```python
from PyQt6.QtWidgets import QProgressBar, QLabel, QVBoxLayout, QWidget

class MetricWidget(QWidget):
    def __init__(self, title, value, unit="%"):
        super().__init__()
        self.setup_ui(title, value, unit)
    
    def setup_ui(self, title, value, unit):
        layout = QVBoxLayout()
        
        self.title_label = QLabel(title)
        self.value_label = QLabel(f"{value}{unit}")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(int(value))
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
    def update_value(self, value):
        self.value_label.setText(f"{value}%")
        self.progress_bar.setValue(int(value))
```

## Технологии

- **PyQt6** - GUI фреймворк
- **Matplotlib** - графики и визуализация
- **NumPy** - численные расчеты
- **Pandas** - обработка данных
- **PyQtGraph** - high-performance графики

## Конфигурация

```yaml
gui:
  window:
    title: "Intelligent Trading System"
    width: 1200
    height: 800
    resizable: true
    
  charts:
    update_interval: 1000  # milliseconds
    max_points: 1000
    theme: "dark"  # dark, light, custom
    
  panels:
    auto_refresh: true
    refresh_interval: 5000  # milliseconds
    
  data:
    cache_size: 10000
    update_timeout: 30  # seconds
    
  themes:
    default: "dark"
    custom_colors:
      background: "#2b2b2b"
      foreground: "#ffffff"
      accent: "#007acc"
```

## Пользовательский опыт

### Интерактивность

- **Drag & Drop**: для файлов конфигурации
- **Keyboard Shortcuts**: быстрые команды
- **Context Menus**: контекстные меню
- **Tooltips**: подсказки и помощь

### Производительность

- **Lazy Loading**: загрузка данных по требованию
- **Virtual Scrolling**: для больших таблиц
- **Caching**: кэширование данных
- **Async Updates**: асинхронные обновления

### Доступность

- **High Contrast**: режим высокой контрастности
- **Font Scaling**: масштабирование шрифтов
- **Keyboard Navigation**: навигация клавиатурой
- **Screen Reader**: поддержка скринридеров

## Интеграция

GUI Layer получает данные от:
- **Data Layer** - рыночные данные
- **Models Layer** - предсказания и сигналы
- **Execution Layer** - информация об ордерах
- **Risk Management** - риск-метрики

И передает команды в:
- **Decision Layer** - ручные торговые команды
- **Execution Layer** - управление ордерами
- **Risk Management** - настройка рисков

## Требования к реализации

1. **Производительность** - плавные обновления в реальном времени
2. **Отзывчивость** - быстрая реакция на действия пользователя
3. **Надежность** - обработка ошибок и восстановление
4. **Масштабируемость** - поддержка больших объемов данных
5. **Кастомизация** - гибкая настройка интерфейса

## Тестирование

- Unit тесты для компонентов GUI
- Integration тесты для потока данных
- Performance тесты для обновлений
- Usability тесты для пользовательского опыта
