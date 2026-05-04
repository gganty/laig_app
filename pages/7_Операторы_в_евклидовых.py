import numpy as np
import streamlit as st

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_7 import DEFS, THMS
from laig.ui import show_steps, matrix_input
from laig.algorithms import svd_steps
from laig.viz import operator_action_2d, rotation_2d

register(DEFS)
register(THMS)


st.title("7. Линейные отображения и операторы в евклидовых пространствах")

tab_def, tab_thm, tab_svd, tab_rot, tab_action = st.tabs([
    "Определения", "Теоремы", "SVD пошагово", "Поворот в 2D", "Действие на круг",
])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_7")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_7")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_svd:
    st.markdown(
        "Введи произвольную (не обязательно квадратную) матрицу $A$ — увидишь построение "
        "сингулярного разложения $A = U \\Sigma V^T$ через спектр $A^T A$ пошагово."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        default_A = np.array([[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0]])
        A = matrix_input("Матрица $A$", default_A, key="svd_A")
        if A is not None:
            st.markdown(f"**Размер:** ${A.shape[0]} \\times {A.shape[1]}$")
            st.markdown(f"**Ранг:** {np.linalg.matrix_rank(A)}")

    with col2:
        if A is not None and A.size > 0:
            try:
                U, Sigma, V, steps = svd_steps(A)
                show_steps(steps)
            except Exception as e:
                st.error(f"Ошибка: {e}")

with tab_rot:
    st.markdown(
        "Ортогональный оператор в $\\mathbb{R}^2$ — это либо поворот, либо отражение. "
        "Здесь — поворот на угол $\\theta$. Единичный круг отображается на себя "
        "(сохранение длин), но базис движется."
    )

    theta = st.slider("Угол $\\theta$ (градусы)", -180, 180, 30, 1)
    fig = rotation_2d(theta)
    st.plotly_chart(fig, use_container_width=True)

    theta_rad = np.radians(theta)
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)],
    ])
    st.markdown(f"**Матрица:** $R_\\theta = \\begin{{pmatrix}} {R[0,0]:.3f} & {R[0,1]:.3f} \\\\ {R[1,0]:.3f} & {R[1,1]:.3f} \\end{{pmatrix}}$")
    st.markdown(f"**$\\det R_\\theta = {np.linalg.det(R):.3f}$** (для собственного движения = $+1$)")

with tab_action:
    st.markdown(
        "Геометрический смысл SVD: единичный круг отображается оператором $A$ в эллипс. "
        "Полуоси эллипса — сингулярные числа $\\sigma_1, \\sigma_2$. Если есть вещественные собственные "
        "векторы, они показаны пунктиром."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Введи матрицу $A$ ($2 \\times 2$):**")
        a11 = st.slider("$a_{11}$", -3.0, 3.0, 2.0, 0.1)
        a12 = st.slider("$a_{12}$", -3.0, 3.0, 1.0, 0.1)
        a21 = st.slider("$a_{21}$", -3.0, 3.0, 0.0, 0.1)
        a22 = st.slider("$a_{22}$", -3.0, 3.0, 1.0, 0.1)
        A = np.array([[a11, a12], [a21, a22]])

        st.markdown(f"**$\\det A = {np.linalg.det(A):.3f}$**")

        s = np.linalg.svd(A, compute_uv=False)
        st.markdown(f"**$\\sigma_1 = {s[0]:.3f},\\ \\sigma_2 = {s[1]:.3f}$**")
        st.markdown(f"**$\\operatorname{{rk}} A = {np.linalg.matrix_rank(A)}$**")

        eigvals = np.linalg.eigvals(A)
        if all(abs(e.imag) < 1e-9 for e in eigvals):
            st.markdown(f"**Собственные значения:** ${eigvals[0].real:.3f},\\ {eigvals[1].real:.3f}$")
        else:
            st.markdown(f"**Собственные значения:** комплексные, ${eigvals[0]:.3f},\\ {eigvals[1]:.3f}$")

        if np.allclose(A, A.T):
            st.info("$A$ симметрична -> самосопряжённый оператор.")
        if np.allclose(A.T @ A, np.eye(2)):
            st.success("$A$ ортогональна ($A^T A = I$).")

    with col2:
        fig = operator_action_2d(A)
        st.plotly_chart(fig, use_container_width=True)
