#!/usr/bin/env python3
"""
UI приложение для конструктора архитектур нейронных сетей.
Запуск: python -m uav_vit.ui.app --port 7860
"""

import argparse
import json

try:
    import gradio as gr
except ImportError as err:
    raise ImportError("Gradio не установлен. Установите его: pip install gradio") from err

from uav_vit.ui.builder import (
    EFFICIENCY_TIPS,
    LAYER_INFO,
    POSITION_ADVICE,
    build_model_from_layers,
    get_architecture_examples,
    get_compatibility_issues,
    validate_layer_sequence,
)


def format_layer_info(layer_type: str) -> str:
    """Форматирует информацию о слое для отображения."""
    info = LAYER_INFO.get(layer_type, {})
    advice = POSITION_ADVICE.get(layer_type, {})
    tips = EFFICIENCY_TIPS.get(layer_type, [])

    lines = [
        f"**{info.get('name', layer_type)}**",
        f"{info.get('description', 'Нет описания')}",
        "",
        "**Параметры:**",
    ]
    for param, details in info.get("params", {}).items():
        lines.append(f"- `{param}`: {details}")

    lines.extend(["", "**Рекомендации по позиции:**"])
    for position, desc in advice.items():
        lines.append(f"- {position}: {desc}")

    if tips:
        lines.extend(["", "**Советы по эффективности:**"])
        for tip in tips:
            lines.append(f"- {tip}")

    return "\n".join(lines)


def update_layer_info(selected_layer: str) -> str:
    """Обновляет информацию о выбранном слое."""
    if not selected_layer:
        return "Выберите слой из палитры для просмотра информации."
    return format_layer_info(selected_layer)


def add_layer_to_architecture(
    architecture_json: str, layer_type: str, params_str: str
) -> tuple[str, str]:
    """Добавляет слой в архитектуру."""
    try:
        architecture = json.loads(architecture_json) if architecture_json else []
    except json.JSONDecodeError:
        architecture = []

    params = {}
    if params_str.strip():
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            error_msg = "❌ Ошибка: Неверный формат JSON в параметрах"
            return json.dumps(architecture, ensure_ascii=False), error_msg

    layer_config = {"type": layer_type, "params": params}

    issues = get_compatibility_issues(architecture + [layer_config])
    if issues:
        warning = "⚠️ Предупреждение:\n" + "\n".join([f"- {issue}" for issue in issues])
        architecture.append(layer_config)
        return json.dumps(architecture, ensure_ascii=False), warning

    architecture.append(layer_config)
    return json.dumps(architecture, ensure_ascii=False), "✅ Слой добавлен успешно"


def remove_layer(architecture_json: str, index: int) -> tuple[str, str]:
    """Удаляет слой из архитектуры по индексу."""
    try:
        architecture = json.loads(architecture_json)
    except json.JSONDecodeError:
        return architecture_json, "❌ Ошибка: Неверный формат архитектуры"

    if not 0 <= index < len(architecture):
        return architecture_json, "❌ Ошибка: Неверный индекс"

    architecture.pop(index)
    return json.dumps(architecture, ensure_ascii=False), "✅ Слой удалён"


def move_layer(architecture_json: str, index: int, direction: str) -> tuple[str, str]:
    """Перемещает слой вверх или вниз."""
    try:
        architecture = json.loads(architecture_json)
    except json.JSONDecodeError:
        return architecture_json, "❌ Ошибка: Неверный формат архитектуры"

    if not 0 <= index < len(architecture):
        return architecture_json, "❌ Ошибка: Неверный индекс"

    if direction == "up" and index > 0:
        architecture[index], architecture[index - 1] = architecture[index - 1], architecture[index]
    elif direction == "down" and index < len(architecture) - 1:
        architecture[index], architecture[index + 1] = architecture[index + 1], architecture[index]
    else:
        return architecture_json, "⚠️ Нельзя переместить дальше"

    return json.dumps(architecture, ensure_ascii=False), "✅ Слой перемещён"


def validate_architecture(architecture_json: str) -> str:
    """Проверяет корректность архитектуры."""
    try:
        architecture = json.loads(architecture_json)
    except json.JSONDecodeError:
        return "❌ Ошибка: Неверный формат JSON"

    is_valid, issues = validate_layer_sequence(architecture)
    if is_valid:
        return "✅ Архитектура корректна! Все слои совместимы."
    else:
        return "❌ Проблемы с архитектурой:\n" + "\n".join([f"- {issue}" for issue in issues])


def load_example(example_name: str) -> str:
    """Загружает пример архитектуры."""
    examples = get_architecture_examples()
    if example_name in examples:
        return json.dumps(examples[example_name], ensure_ascii=False, indent=2)
    return ""


def build_and_summary(architecture_json: str) -> str:
    """Строит модель и возвращает сводку."""
    try:
        architecture = json.loads(architecture_json)
    except json.JSONDecodeError:
        return "❌ Ошибка: Неверный формат JSON"

    if not architecture:
        return "⚠️ Архитектура пуста"

    try:
        model = build_model_from_layers(architecture)
        total_params = sum(p.numel() for p in model.parameters())
        layers_info = []
        for i, layer in enumerate(architecture):
            layers_info.append(f"{i + 1}. {layer['type']}")

        result_msg = (
            "✅ Модель построена успешно!\n\n**Структура:**\n"
            + "\n".join(layers_info)
            + f"\n\n**Всего параметров:** {total_params:,}"
        )
        return result_msg
    except Exception as e:
        return f"❌ Ошибка при построении модели: {str(e)}"


def create_ui() -> gr.Blocks:
    """Создаёт Gradio интерфейс."""

    with gr.Blocks(title="Конструктор Нейронных Сетей", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🧠 Конструктор Архитектур Нейронных Сетей

        Перетаскивайте слои, настраивайте параметры и получайте рекомендации в реальном времени.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📦 Палитра Слоёв")

                layer_options = list(LAYER_INFO.keys())
                layer_selector = gr.Dropdown(
                    choices=layer_options,
                    label="Выберите тип слоя",
                    value=layer_options[0] if layer_options else None,
                )

                layer_info_display = gr.Markdown("Выберите слой для просмотра информации.")

                params_input = gr.Textbox(
                    label="Параметры слоя (JSON)",
                    placeholder='{"channels": 64, "kernel_size": 3}',
                    lines=3,
                )

                add_btn = gr.Button("➕ Добавить слой", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 🏗️ Архитектура Модели")

                architecture_json = gr.Textbox(
                    label="Конфигурация архитектуры (JSON)", lines=10, placeholder="[]"
                )

                with gr.Row():
                    layer_index = gr.Number(label="Индекс слоя", precision=0, value=0)
                    move_up_btn = gr.Button("⬆️ Вверх")
                    move_down_btn = gr.Button("⬇️ Вниз")
                    remove_btn = gr.Button("🗑️ Удалить")

                validate_btn = gr.Button("✅ Проверить архитектуру", variant="secondary")
                build_btn = gr.Button("🚀 Построить модель", variant="primary")

                status_display = gr.Textbox(label="Статус", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Примеры Архитектур")
                example_names = list(get_architecture_examples().keys())
                example_selector = gr.Dropdown(
                    choices=example_names, label="Загрузить пример", value=None
                )
                load_example_btn = gr.Button("Загрузить")

        # Обработчики событий
        layer_selector.change(
            fn=update_layer_info, inputs=[layer_selector], outputs=[layer_info_display]
        )

        add_btn.click(
            fn=add_layer_to_architecture,
            inputs=[architecture_json, layer_selector, params_input],
            outputs=[architecture_json, status_display],
        )

        remove_btn.click(
            fn=remove_layer,
            inputs=[architecture_json, layer_index],
            outputs=[architecture_json, status_display],
        )

        move_up_btn.click(
            fn=lambda arch, idx: move_layer(arch, idx, "up"),
            inputs=[architecture_json, layer_index],
            outputs=[architecture_json, status_display],
        )

        move_down_btn.click(
            fn=lambda arch, idx: move_layer(arch, idx, "down"),
            inputs=[architecture_json, layer_index],
            outputs=[architecture_json, status_display],
        )

        validate_btn.click(
            fn=validate_architecture, inputs=[architecture_json], outputs=[status_display]
        )

        build_btn.click(fn=build_and_summary, inputs=[architecture_json], outputs=[status_display])

        load_example_btn.click(
            fn=load_example, inputs=[example_selector], outputs=[architecture_json]
        )

        # Инициализация информации о первом слое
        app.load(fn=update_layer_info, inputs=[layer_selector], outputs=[layer_info_display])

    return app


def main():
    parser = argparse.ArgumentParser(description="UI конструктора нейронных сетей")
    parser.add_argument("--port", type=int, default=7860, help="Порт для запуска")
    parser.add_argument("--share", action="store_true", help="Создать публичную ссылку")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Имя сервера")
    args = parser.parse_args()

    app = create_ui()
    print(f"🚀 Запуск UI на порту {args.port}...")
    app.launch(server_port=args.port, server_name=args.server_name, share=args.share)


if __name__ == "__main__":
    main()
