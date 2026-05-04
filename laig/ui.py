"""
Общие UI-компоненты для всех страниц.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
from .algorithms import Step


def theorem_card(name: str, statement: str, idea: str = "", proof: str = "",
                 pitfalls: str = "", deps: str = "") -> None:
    """
    Карточка теоремы с раскрывающимися блоками.

    Дисциплина: формулировка всегда видна, доказательство и подводные камни
    спрятаны под expander, чтобы можно было сначала попробовать вспомнить.
    """
    st.markdown(f"#### {name}")
    st.markdown(statement)

    if idea:
        with st.expander("💡 Идея доказательства", expanded=False):
            st.markdown(idea)

    if deps:
        with st.expander("🔗 Зависит от", expanded=False):
            st.markdown(deps)

    if proof:
        with st.expander("📜 Полное доказательство", expanded=False):
            st.markdown(proof)

    if pitfalls:
        with st.expander("⚠️ Подводные камни", expanded=False):
            st.markdown(pitfalls)


def definition_card(num: int, name: str, content: str) -> None:
    """Краткая карточка определения."""
    with st.container(border=True):
        st.markdown(f"**{num}. {name}**")
        st.markdown(content)


def show_steps(steps: list[Step]) -> None:
    """
    Отображает список шагов алгоритма с слайдером — можно листать вперёд/назад
    или сразу прыгнуть на финал.
    """
    if not steps:
        st.info("Нет шагов для отображения.")
        return

    n = len(steps)
    if n == 1:
        idx = 0
    else:
        idx = st.slider("Шаг", 1, n, 1) - 1

    step = steps[idx]
    st.markdown(f"**{step.title}**")
    st.markdown(step.description)

    if step.formulas:
        for f in step.formulas:
            st.markdown(f"`{f}`")

    if step.matrices:
        cols = st.columns(min(len(step.matrices), 3))
        for i, (name, mat) in enumerate(step.matrices.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{name}**")
                st.dataframe(_format_matrix(mat), hide_index=True)


def matrix_input(label: str, default: np.ndarray, key: str,
                 max_rows: int = 8, max_cols: int = 8) -> np.ndarray:
    """
    Ввод матрицы через текстовое поле. Каждая строка — одна строка матрицы,
    числа через пробел или запятую.
    """
    default_str = "\n".join(
        " ".join(str(_round(x)) for x in row) for row in default
    )

    txt = st.text_area(label, value=default_str, key=key, height=150,
                       help="Каждая строка — одна строка матрицы. Числа через пробел.")

    # Парсинг
    try:
        rows = []
        for line in txt.strip().split("\n"):
            line = line.replace(",", " ")
            row = [float(x) for x in line.split() if x.strip()]
            if row:
                rows.append(row)
        if not rows:
            st.warning("Пустой ввод.")
            return default
        # Проверка одинаковой длины
        n_cols = len(rows[0])
        if any(len(r) != n_cols for r in rows):
            st.error(f"Не все строки одинаковой длины ({n_cols} столбцов в первой).")
            return default
        if len(rows) > max_rows or n_cols > max_cols:
            st.error(f"Размер ограничен: ≤{max_rows}×{max_cols}.")
            return default
        return np.array(rows, dtype=float)
    except ValueError as e:
        st.error(f"Не удалось распарсить: {e}")
        return default


def vector_input(label: str, default: np.ndarray, key: str,
                 max_dim: int = 10) -> np.ndarray:
    """Ввод вектора через одну строку."""
    default_str = " ".join(str(_round(x)) for x in default.flat)
    txt = st.text_input(label, value=default_str, key=key)
    try:
        txt = txt.replace(",", " ")
        vec = np.array([float(x) for x in txt.split() if x.strip()], dtype=float)
        if len(vec) == 0:
            return default
        if len(vec) > max_dim:
            st.error(f"Размерность ограничена: ≤{max_dim}.")
            return default
        return vec
    except ValueError as e:
        st.error(f"Не удалось распарсить: {e}")
        return default


def _round(x: float) -> float:
    """Аккуратное округление для отображения."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, 4)


def _format_matrix(mat: np.ndarray) -> list[list]:
    """Превращает numpy-матрицу в список списков с округлением для st.dataframe."""
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    return [[_round(x) for x in row] for row in mat]
