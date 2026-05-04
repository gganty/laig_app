import numpy as np
import streamlit as st
import pandas as pd

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_2 import DEFS, THMS
from laig.ui import matrix_input

register(DEFS)
register(THMS)


def _qr_pivoting(A):
    try:
        from scipy.linalg import qr
        Q, R, P = qr(A, pivoting=True)
        return Q, R, P
    except ImportError:
        return np.eye(A.shape[0]), A, np.arange(A.shape[1])


st.title("2. Линейные отображения")

tab_def, tab_thm, tab_kernel = st.tabs(["Определения", "Теоремы", "Ядро/образ/ранг"])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_2")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_2")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_kernel:
    st.markdown(
        "Ввод — матрица линейного отображения. Вывод — ранг, размерности ядра и образа, "
        "явные базисы $\\ker \\varphi$ и $\\operatorname{Im} \\varphi$."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        default_A = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [1.0, 1.0, 1.0],
        ])
        A = matrix_input("Матрица $A$ (отображения $V \\to W$)", default_A, key="kerim_A")

        if A is not None:
            m, n = A.shape
            r = int(np.linalg.matrix_rank(A))
            st.markdown(f"**Размер:** $m \\times n = {m} \\times {n}$ ($\\varphi \\colon \\mathbb{{R}}^{{{n}}} \\to \\mathbb{{R}}^{{{m}}}$)")
            st.markdown(f"**$\\operatorname{{rk}} A = \\dim \\operatorname{{Im}} \\varphi = {r}$**")
            st.markdown(f"**$\\dim \\ker \\varphi = n - \\operatorname{{rk}} A = {n - r}$**")
            st.markdown(f"**Проверка:** $\\dim V = \\dim \\ker + \\dim \\operatorname{{Im}} = {n - r} + {r} = {n}$ ✅")

    with col2:
        if A is not None:
            m, n = A.shape

            U, sigma, Vt = np.linalg.svd(A)
            tol = max(m, n) * np.max(sigma) * 1e-10 if len(sigma) > 0 else 0
            ker_dim = int(np.sum(sigma <= tol)) + (n - len(sigma))
            if ker_dim > 0:
                ker_basis = Vt[-ker_dim:].T
                st.markdown("**Базис ядра (по столбцам):**")
                st.dataframe(
                    pd.DataFrame(np.round(ker_basis, 4)),
                    hide_index=True,
                )
                check = A @ ker_basis
                st.caption(f"Проверка: $\\|A \\cdot (\\text{{базис ядра}})\\| = {np.linalg.norm(check):.2e} \\approx 0$")
            else:
                st.markdown("**Ядро тривиально:** $\\ker \\varphi = \\{0\\}$ ($\\varphi$ инъективно).")

            r = int(np.linalg.matrix_rank(A))
            if r > 0:
                Q, R, P = _qr_pivoting(A)
                pivot_cols = P[:r]
                im_basis = A[:, sorted(pivot_cols)]
                st.markdown("**Базис образа (выбраны лин. независимые столбцы $A$):**")
                st.dataframe(
                    pd.DataFrame(np.round(im_basis, 4)),
                    hide_index=True,
                )

            st.markdown("---")
            if ker_dim == 0:
                st.success("$\\varphi$ **инъективно** ($\\ker \\varphi = \\{0\\}$).")
            else:
                st.info("$\\varphi$ **не инъективно** ($\\ker \\varphi \\ne \\{0\\}$).")

            if r == m:
                st.success("$\\varphi$ **сюръективно** ($\\operatorname{Im} \\varphi = W$).")
            else:
                st.info("$\\varphi$ **не сюръективно**.")
