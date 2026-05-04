import numpy as np
import streamlit as st

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_4 import DEFS, THMS
from laig.ui import show_steps, matrix_input, vector_input
from laig.algorithms import gram_schmidt, least_squares
from laig.viz import projection_2d, projection_3d_to_plane

register(DEFS)
register(THMS)


st.title("4. Евклидовы пространства")

tab_def, tab_thm, tab_gs, tab_proj, tab_lsq = st.tabs([
    "Определения", "Теоремы", "Грам-Шмидт", "Проекции (визуализация)", "МНК",
])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_4")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_4")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_gs:
    st.markdown(
        "Введи произвольную систему линейно независимых векторов в $\\mathbb{R}^n$ — "
        "увидишь весь процесс ортогонализации Грама-Шмидта пошагово."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        default_vecs = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        V = matrix_input(
            "Векторы (по строкам, через пробел)",
            default_vecs, key="gs_vecs",
        )
        normalize = st.checkbox("Нормировать (получить ОНБ)", value=True)

    with col2:
        if V is not None and len(V) > 0:
            try:
                result, steps = gram_schmidt(V, normalize=normalize)
                show_steps(steps)
            except Exception as e:
                st.error(f"Ошибка: {e}")

with tab_proj:
    st.markdown("Визуализация ортогональной проекции на подпространство.")

    mode = st.radio("Размерность", ["2D: проекция на прямую", "3D: проекция на плоскость"],
                    horizontal=True)

    if mode == "2D: проекция на прямую":
        col1, col2 = st.columns([1, 2])
        with col1:
            v = vector_input("Вектор $v$", np.array([2.0, 3.0]), key="proj_v_2d")
            u = vector_input("Направление прямой $u$", np.array([3.0, 1.0]), key="proj_u_2d")
            if len(v) == 2 and len(u) == 2:
                v = v[:2]
                u = u[:2]
                coef = (v @ u) / (u @ u) if (u @ u) > 1e-12 else 0
                proj = coef * u
                perp = v - proj
                st.markdown(f"**$\\operatorname{{pr}}_U(v) = {coef:.3f} \\cdot u = {tuple(np.round(proj, 3))}$**")
                st.markdown(f"**$v - \\operatorname{{pr}} = {tuple(np.round(perp, 3))}$**")
                st.markdown(f"**$\\|v - \\operatorname{{pr}}\\| = \\rho(v, U) = {np.linalg.norm(perp):.3f}$**")
        with col2:
            if len(v) == 2 and len(u) == 2:
                fig = projection_2d(v, u)
                st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            v = vector_input("Вектор $v$", np.array([1.0, 2.0, 3.0]), key="proj_v_3d")
            u1 = vector_input("Базис плоскости $u_1$", np.array([1.0, 0.0, 0.0]), key="proj_u1_3d")
            u2 = vector_input("Базис плоскости $u_2$", np.array([0.0, 1.0, 0.0]), key="proj_u2_3d")
        with col2:
            if len(v) == 3 and len(u1) == 3 and len(u2) == 3:
                fig = projection_3d_to_plane(v, u1, u2)
                st.plotly_chart(fig, use_container_width=True)

with tab_lsq:
    st.markdown(
        "Метод наименьших квадратов: ищем $x$, минимизирующий $\\|Ax - b\\|^2$ для "
        "(возможно несовместной) системы $Ax = b$."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Пример: подгонка прямой $y = ax + b$**")
        st.markdown("Точки $(x_i, y_i) \\to A = [[x_i, 1]],\\ b = [y_i]$.")

        default_A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
        default_b = np.array([1.1, 1.9, 3.2, 3.8])

        A = matrix_input("Матрица $A$", default_A, key="lsq_A")
        b = vector_input("Правая часть $b$", default_b, key="lsq_b")

    with col2:
        if A is not None and b is not None and A.shape[0] == len(b):
            try:
                x, steps = least_squares(A, b)
                show_steps(steps)
            except Exception as e:
                st.error(f"Ошибка: {e}")
        else:
            st.warning(f"Размеры не согласованы: $A \\in \\mathbb{{R}}^{{{A.shape[0]} \\times {A.shape[1]}}}$, $b \\in \\mathbb{{R}}^{{{len(b)}}}$.")
