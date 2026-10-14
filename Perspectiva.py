# -*- coding: utf-8 -*-
"""
Perspectiva.py — точка входа.
Запуск: python Perspectiva.py image.jpg
"""
import sys
import os

# Фикс кодировки консоли Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from gui import interactive_grid

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Укажите путь к изображению: python Perspectiva.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("Файл не найден: " + str(image_path))
        sys.exit(1)

    result = interactive_grid(
        image_path,
        rows=0, cols=0,
        output_path="result_with_grid.jpg"
    )

    if result is not None:
        print("Готово! Результат сохранён.")
    else:
        print("Операция отменена.")
