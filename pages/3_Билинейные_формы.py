import numpy as np
import streamlit as st

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_3 import DEFS, THMS
from laig.ui import show_steps, matrix_input
from laig.algorithms import jacobi_method
from laig.viz import quadric_section_2d

register(DEFS)
register(THMS)


st.title("3. Билинейные и квадратичные формы")

tab_def, tab_thm, tab_jac, tab_quad = st.tabs([
    "Определения", "Теоремы", "Метод Якоби", "Линии уровня квадрики",
])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_3")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_3")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_jac:
    st.markdown(
        "Ввод симметричной матрицы $B$. Вывод — построение диагонального вида "
        "методом Якоби пошагово (см. [теорему о методе Якоби](#теоремы))."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        default_B = np.array([
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ])
        B = matrix_input("Матрица $B$ (должна быть симметричной)", default_B, key="jac_B")

        if B is not None and B.shape[0] == B.shape[1]:
            st.markdown("**Угловые миноры:**")
            for k in range(1, B.shape[0] + 1):
                d = np.linalg.det(B[:k, :k])
                st.markdown(f"- $\\Delta_{{{k}}} = {d:.4g}$")

    with col2:
        if B is not None and B.shape[0] == B.shape[1]:
            try:
                D, C, steps = jacobi_method(B)
                show_steps(steps)

                if D is not None:
                    n = D.shape[0]
                    diag = [D[i, i] for i in range(n)]
                    p = sum(1 for x in diag if x > 1e-9)
                    r = sum(1 for x in diag if x < -1e-9)
                    z = sum(1 for x in diag if abs(x) <= 1e-9)
                    st.success(
                        f"**Сигнатура: ($p={p}$, $r={r}$" +
                        (f", нулевых $={z}$" if z > 0 else "") + ")**"
                    )
                    if z == 0:
                        if p == n:
                            st.success("Форма **положительно определена**.")
                        elif r == n:
                            st.success("Форма **отрицательно определена**.")
                        else:
                            st.info("Форма **неопределённая**.")
                    else:
                        st.info("Форма вырождена (есть нулевые на диагонали).")

            except Exception as e:
                st.error(f"Ошибка: {e}")
        else:
            st.warning("Матрица должна быть квадратной.")

with tab_quad:
    st.markdown(
        "Линии уровня квадратичной формы $q(x, y) = b_{11}x^2 + 2b_{12}xy + b_{22}y^2$ на $\\mathbb{R}^2$. "
        "Главные оси (пунктир) соответствуют собственным векторам ассоциированного "
        "самосопряжённого оператора (см. теорему о главных осях, раздел 7)."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        b11 = st.slider("$b_{11}$", -3.0, 3.0, 1.0, 0.25)
        b12 = st.slider("$b_{12}$", -3.0, 3.0, 0.5, 0.25)
        b22 = st.slider("$b_{22}$", -3.0, 3.0, 1.0, 0.25)
        B = np.array([[b11, b12], [b12, b22]])

        eigvals = np.linalg.eigvalsh(B)
        st.markdown(f"**Собственные значения:** $\\lambda_1 = {eigvals[0]:.3f},\\ \\lambda_2 = {eigvals[1]:.3f}$")

        if eigvals[0] > 1e-9 and eigvals[1] > 1e-9:
            st.success("Форма положительно определена → **эллипс**.")
        elif eigvals[0] < -1e-9 and eigvals[1] < -1e-9:
            st.success("Форма отрицательно определена → линии уровня — эллипсы (для отрицательных значений).")
        elif eigvals[0] * eigvals[1] < -1e-9:
            st.info("Форма неопределённая → **гипербола**.")
        else:
            st.warning("Форма вырождена ($\\det = 0$) → **параболическая** или константа.")

    with col2:
        fig = quadric_section_2d(B)
        st.plotly_chart(fig, use_container_width=True)
