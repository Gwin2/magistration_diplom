# Neural Network Architecture Builder UI

Адаптивный drag-and-drop конструктор архитектур нейронных сетей с подсказками и рекомендациями.

## Обзор

Модуль `uav_vit.ui` предоставляет интерактивный интерфейс для построения архитектур нейронных сетей с помощью перетаскивания слоёв. Включает валидацию совместимости слоёв и рекомендации по эффективности.

## Компоненты

### 1. builder.py - Ядро системы

#### LayerType
Перечисление поддерживаемых типов слоёв:
- `CONV2D` - 2D свёрточный слой
- `MAXPOOL2D` - Max pooling
- `AVGPOOL2D` - Average pooling
- `BATCHNORM` - Batch normalization
- `DROPOUT` - Регуляризация dropout
- `FLATTEN` - Выпрямление тензора
- `LINEAR` - Полносвязный слой
- `VIT_BLOCK` - Блок Vision Transformer
- `EMBEDDING` - Patch embedding для ViT
- `RESIDUAL` - Residual block
- `LAYER_NORM` - Layer normalization
- `RELU` - Activation ReLU
- `SOFTMAX` - Activation Softmax

#### ActivationType
Функции активации:
- `RELU`, `GELU`, `SIGMOID`, `TANH`, `SOFTMAX`, `LEAKY_RELU`

#### LayerNode
Представляет узел в графе архитектуры:
```python
from uav_vit.ui.builder import LayerNode, LayerType, ActivationType

node = LayerNode(
    id="conv1",
    layer_type=LayerType.CONV2D,
    params={"in_channels": 3, "out_channels": 64, "kernel_size": 3},
    position=0,
    activation=ActivationType.RELU
)
```

#### ArchitectureValidator
Валидирует архитектуру на:
- Совместимость последовательных слоёв
- Позиционные рекомендации (early/middle/late)
- Глобальные ограничения архитектуры
- Эффективность (с подсказками)

```python
from uav_vit.ui.builder import ArchitectureValidator, LayerNode, LayerType

validator = ArchitectureValidator()
layers = [
    LayerNode("l1", LayerType.CONV2D, {"in_channels": 3, "out_channels": 32}, 0),
    LayerNode("l2", LayerType.MAXPOOL2D, {"kernel_size": 2}, 1),
]

is_valid, errors, warnings, recommendations = validator.validate_layer_sequence(layers)
```

#### NetworkBuilder
Конструктор для построения архитектур:

```python
from uav_vit.ui.builder import NetworkBuilder, LayerType

builder = NetworkBuilder()

# Добавление слоёв
builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 32})
builder.add_layer(LayerType.BATCHNORM, {"num_features": 32})
builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2})

# Перемещение слоя
builder.move_layer("layer_1", new_position=2)

# Удаление слоя
builder.remove_layer("layer_2")

# Валидация
is_valid, errors, warnings, recs = builder.validate()

# Экспорт конфигурации
config = builder.export_to_config()
```

### 2. app.py - Интерактивный UI (Gradio)

Запуск интерактивного интерфейса:

```bash
python -m uav_vit.ui.app --port 7860
```

Или программно:

```python
from uav_vit.ui.app import ArchitectureBuilderUI

ui = ArchitectureBuilderUI()
ui.launch(port=7860, share=False)
```

#### Возможности UI:
- 🎨 Палитра слоёв с описанием
- 🏛️ Визуализация архитектуры
- ✅ Валидация в реальном времени
- 💡 Рекомендации по эффективности
- 📖 Примеры архитектур (CNN, ViT, ResNet)
- 🔧 Управление слоями (добавление, удаление, перемещение)

## Примеры использования

### Построение простой CNN

```python
from uav_vit.ui.builder import NetworkBuilder, LayerType

builder = NetworkBuilder()

# Conv block 1
builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 32, "kernel_size": 3})
builder.add_layer(LayerType.BATCHNORM, {"num_features": 32})
builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2, "stride": 2})

# Conv block 2
builder.add_layer(LayerType.CONV2D, {"in_channels": 32, "out_channels": 64, "kernel_size": 3})
builder.add_layer(LayerType.BATCHNORM, {"num_features": 64})
builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2, "stride": 2})

# Classifier head
builder.add_layer(LayerType.FLATTEN, {})
builder.add_layer(LayerType.DROPOUT, {"p": 0.5})
builder.add_layer(LayerType.LINEAR, {"in_features": 64 * 7 * 7, "out_features": 128})
builder.add_layer(LayerType.LINEAR, {"in_features": 128, "out_features": 10})

summary = builder.get_architecture_summary()
print(f"Valid: {summary['valid']}")
print(f"Layers: {summary['layer_count']}")
```

### Построение ViT-подобной архитектуры

```python
from uav_vit.ui.builder import NetworkBuilder, LayerType

builder = NetworkBuilder()

builder.add_layer(LayerType.EMBEDDING, {"img_size": 224, "patch_size": 16, "dim": 768})
builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
builder.add_layer(LayerType.LINEAR, {"in_features": 768, "out_features": 10})

config = builder.export_to_config()
```

## Правила совместимости слоёв

### Conv2D
- **Предшественники**: Conv2D, BatchNorm, ReLU, Embedding
- **Преемники**: Conv2D, BatchNorm, MaxPool2D, AvgPool2D, Dropout, ReLU, Linear
- **Позиция**: Ранние слои
- **Советы**: Используйте маленькие ядра (3x3), добавляйте BatchNorm

### MaxPool2D
- **Предшественники**: Conv2D, ReLU, BatchNorm
- **Преемники**: Conv2D, MaxPool2D, Flatten
- **Позиция**: Ранние слои
- **Советы**: Используйте после функций активации

### Linear
- **Предшественники**: Flatten, Linear, Dropout
- **Преемники**: ReLU, Dropout, Linear, Softmax
- **Позиция**: Поздние слои
- **Советы**: Уменьшайте размер постепенно

### ViTBlock
- **Предшественники**: ViTBlock, Embedding
- **Преемники**: ViTBlock, LayerNorm, Linear
- **Позиция**: Средние слои
- **Советы**: Используйте 6-12 блоков для глубокой экстракции признаков

## Тестирование

```bash
pytest tests/ui/test_builder.py -v
```

## Зависимости

- Python 3.8+
- Gradio (для UI): `pip install gradio`

## Архитектурные рекомендации

### Хорошие практики:
1. Начинайте с Conv2D или Embedding
2. Используйте BatchNorm после Conv2D
3. Применяйте Pooling после активации
4. Завершайте Linear или классификационным слоем
5. Добавляйте Dropout перед Linear слоями

### Избегайте:
1. Linear слой без предварительного Flatten
2. MaxPool2D после Flatten
3. Слишком много слоев одного типа подряд
4. Размещения поздних слоёв (Linear, Dropout) в начале

## Расширение

Для добавления новых типов слоёв:

1. Добавьте тип в `LayerType` enum
2. Определите правила в `LAYER_RULES`
3. Добавьте параметры по умолчанию в `_get_default_params`
4. Обновите тесты
