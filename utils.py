# -*- coding: utf-8 -*-
"""
utils.py — вспомогательные функции.
"""
import sys
import ctypes


def set_unicode_title(win_key, title):
    """Устанавливает Unicode-заголовок для консольного окна."""
    try:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass
