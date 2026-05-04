# Mission Control UI

**Версия**: 2.0 (8-шаговый мастер обучения)  
**Последнее обновление**: 2024

## Обзор

Mission Control UI — это единая операторская панель для управления полным циклом обучения нейронных сетей: от загрузки данных до деплоя модели в production.

### Ключевые возможности

- 📦 **Загрузка датасетов** без ручного редактирования файловой структуры
- ✏️ **Редактирование YAML-конфигов** и шаблонов моделей прямо в браузере
- 🚀 **Запуск train/eval** через `control-api` с отслеживанием lifecycle jobs
- 📊 **Просмотр TensorBoard, MLflow**, рекомендаций системы и сравнение запусков
- 🏷️ **Теги, рейтинг и заметки** к экспериментам
- 🔧 **Регистрация моделей в TorchServe** и выполнение инференса
- 📈 **Контроль нагрузки** по контейнерам и процессам
- 🎨 **Минималистичный дизайн** с адаптивной вёрсткой

## Где доступен

| Параметр | Значение |
|----------|----------|
| URL | `http://localhost:${UI_HOST_PORT}` |
| Порт по умолчанию | 18090 |
| Сервис Docker | `uav-ui` |
| Технология | Nginx + static frontend + reverse proxy |

## 8 шагов обучения нейронной сети

### Шаг 1: Data 📊

**Назначение**: Загрузка и подготовка датасетов

**Функционал**:
- Upload архивов (`zip`, `tar`, `tgz`, `gz`)
- Регистрация существующих каталогов без копирования
- Поиск по имени, пути и тегам
- Скачивание dataset bundle через UI
- Автоматическая конвертация в COCO-формат

**Результат**: Готовый к обучению датасет в `data/processed/`

---

### Шаг 2: Architecture 🏗️

**Назначение**: Конструктор модели

**Функционал**:
- Библиотека базовых архитектур (ViT, DETR, YOLOs)
- Конструктор кастомных detector heads
- Выбор слоёв из каталога (conv, pool, attention, FC)
- Автоподстановка параметров и рекомендации по dataset profile
- Предпросмотр архитектуры в реальном времени
- Синхронизация code/YAML editors

**Результат**: Конфигурация модели в `configs/experiments/`

---

### Шаг 3: Configuration ⚙️

**Назначение**: Параметры обучения

**Функционал**:
- Выбор optimizer (Adam, SGD, AdamW)
- Настройка scheduler (Cosine, Step, ReduceLROnPlateau)
- Аугментации данных (Mosaic, MixUp, RandomFlip)
- Гиперпараметры (batch size, learning rate, epochs)
- Валидация конфига перед запуском

**Результат**: Полный YAML-конфиг эксперимента

---

### Шаг 4: Training 🚀

**Назначение**: Запуск и мониторинг обучения

**Функционал**:
- Запуск train/eval через control-api
- Отслеживание lifecycle jobs (pending, running, completed, failed)
- Live-логи в реальном времени
- Прогресс-бары по эпохам
- Экстренная остановка эксперимента
- Автоматическое сохранение чекпоинтов

**Результат**: Обученная модель в `runs/<experiment>/`

---

### Шаг 5: Evaluation 📈

**Назначение**: Анализ результатов

**Функционал**:
- Метрики на валидации/тесте (mAP50, mAP75, precision, recall)
- Confusion matrix
- PR-кривые
- Визуализация предсказаний на изображениях
- Сравнение с baseline
- Экспорт отчётов (CSV, LaTeX)

**Результат**: Отчёт о метриках в `reports/`

---

### Шаг 6: Deploy 🌐

**Назначение**: Экспорт и serving модели

**Функционал**:
- Экспорт модели в TorchScript / ONNX
- Регистрация модели в TorchServe
- Inference probe (проверка работоспособности)
- Настройка gateway ко всем web-сервисам
- Версионирование моделей

**Результат**: Модель доступна через REST API

---

### Шаг 7: Metrics 📊

**Назначение**: Live-мониторинг системы

**Функционал**:
- Интеграция с Prometheus + Grafana
- Дашборды: System, Training, Resources
- KPI-карточки: mAP50, latency, FPS, service health
- Host CPU/memory usage
- Firing alerts из Alertmanager
- История метрик экспериментов

**Результат**: Полная наблюдаемость системы

---

### Шаг 8: Resources 💾

**Назначение**: Управление ресурсами

**Функционал**:
- Потребление GPU/CPU/RAM по контейнерам
- Потребление по процессам
- Генерация `docker-compose.override.yml`
- Квоты и лимиты ресурсов
- Рекомендации по оптимизации

**Результат**: Оптимизированная конфигурация ресурсов

---

## Навигация и UX

### Режимы работы

- **Последовательный режим**: автоматический переход между шагами после выполнения действий
- **Ручное управление**: кнопки Back/Next для навигации
- **Прямой доступ**: клики по шагам в боковой панели
- **Сохранение прогресса**: состояние сохраняется в localStorage браузера

### Темы и настройки

| Параметр | Опции | Описание |
|----------|-------|----------|
| Тема | `Flight`, `Horizon`, `Paper`, `Signal` | Цветовая схема интерфейса |
| Плотность | `Compact`, `Comfort` | Размер элементов UI |
| Анимации | `On`, `Off` | Движения и переходы |

**Настройки сохраняются в `localStorage`** и применяются при следующем посещении.

---

## Техническая архитектура

### Frontend

| Файл | Назначение |
|------|------------|
| `ui/index.html` | Разметка приложения, 8 вкладок |
| `ui/styles.css` | Стили, темы, адаптивная вёрстка |
| `ui/app.js` | Логика навигации, формы, API calls |

### Backend

| Компонент | Назначение |
|-----------|------------|
| `docker/ui/nginx.conf` | Reverse proxy до web-сервисов стека |
| `src/uav_vit/control/app.py` | FastAPI control layer |
| `src/uav_vit/control/workspace.py` | Управление workspace, jobs, configs, datasets |
| `src/uav_vit/control/mlops.py` | Bridges к MLflow, TensorBoard и TorchServe |

### Reverse proxy endpoints

Через единый UI host доступны:

```
/api/control          → Control Plane API
/api/mlflow           → MLflow tracking
/api/grafana          → Grafana dashboards
/api/prometheus       → Prometheus metrics
/api/tensorboard      → TensorBoard
/api/alertmanager     → Alertmanager
/api/pushgateway      → Pushgateway
/api/minio            → MinIO API
/api/minio-console    → MinIO Console
/api/torchserve       → TorchServe inference
/api/torchserve-mgmt  → TorchServe management
/api/torchserve-metrics → TorchServe metrics
/api/cadvisor         → cAdvisor metrics
/api/node-exporter    → Node Exporter
/api/process-exporter → Process Exporter
/api/postgres-exporter → Postgres Exporter
```

---

## Запуск

### Базовый режим

UI, control plane, tracking, monitoring, storage:

```bash
docker compose up -d --build
```

### Полный режим

С обучением и инференсом:

```bash
docker compose --profile training --profile inference up -d --build
```

### Проверка работоспособности

1. Откройте `http://localhost:18090`
2. Убедитесь, что все 8 шагов доступны
3. Проверьте статус сервисов в разделе **Metrics**

---

## Troubleshooting

### Проблема: UI не открывается

**Решение:**
```bash
docker compose ps uav-ui
docker compose logs uav-ui
```

### Проблема: Не работают proxy endpoints

**Решение:**
1. Проверьте, что целевые сервисы запущены
2. Проверьте логи nginx: `docker compose logs uav-ui | grep proxy`

### Проблема: Сбилась тема интерфейса

**Решение:**
Очистите localStorage браузера или откройте DevTools Console:
```javascript
localStorage.clear()
location.reload()
```

---

**Документ обновлён**: 2024  
**Версия UI**: 2.0 (8-шаговый мастер)
