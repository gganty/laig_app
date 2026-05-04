import numpy as np
import streamlit as st

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_5 import DEFS, THMS
from laig.ui import vector_input
from laig.viz import cross_product_3d

register(DEFS)
register(THMS)


st.title("5. Аналитическая геометрия и линейные многообразия")

tab_def, tab_thm, tab_cross, tab_dist = st.tabs([
    "Определения", "Теоремы", "Векторное произведение", "Расстояния",
])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_5")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_5")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_cross:
    st.markdown(
        "Векторное произведение $[u, v]$ и параллелограмм на $u, v$ в $\\mathbb{R}^3$. "
        "Длина $\\|[u, v]\\|$ — площадь параллелограмма."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        u = vector_input("$u$", np.array([1.0, 2.0, 0.0]), key="cross_u")
        v = vector_input("$v$", np.array([0.0, 1.0, 2.0]), key="cross_v")

        if len(u) == 3 and len(v) == 3:
            w = np.cross(u, v)
            st.markdown(f"**$[u, v] = ({w[0]:.3f},\\ {w[1]:.3f},\\ {w[2]:.3f})$**")
            st.markdown(f"**$\\|[u, v]\\| = {np.linalg.norm(w):.4f}$** (площадь параллелограмма)")
            st.markdown(f"**$\\langle [u, v], u \\rangle = {w @ u:.6f}$** (должно быть 0)")
            st.markdown(f"**$\\langle [u, v], v \\rangle = {w @ v:.6f}$** (должно быть 0)")

            if np.linalg.norm(w) < 1e-9:
                st.warning("$u, v$ **коллинеарны** (или один из них нулевой).")

    with col2:
        if len(u) == 3 and len(v) == 3:
            fig = cross_product_3d(u, v)
            st.plotly_chart(fig, use_container_width=True)

with tab_dist:
    st.markdown("Калькулятор расстояний в $\\mathbb{R}^3$.")

    mode = st.radio("Тип задачи", [
        "Точка -> прямая",
        "Точка -> плоскость",
        "Прямая -> прямая (скрещивающиеся)",
    ])

    if mode == "Точка -> прямая":
        col1, col2 = st.columns(2)
        with col1:
            M0 = vector_input("Точка $M_0$", np.array([1.0, 1.0, 1.0]), key="d1_m0")
            M1 = vector_input("Точка прямой $M_1$", np.array([0.0, 0.0, 0.0]), key="d1_m1")
            a = vector_input("Направляющий вектор $\\vec{a}$", np.array([1.0, 0.0, 0.0]), key="d1_a")
        with col2:
            if len(M0) == 3 and len(M1) == 3 and len(a) == 3:
                cross = np.cross(M0 - M1, a)
                norm_a = np.linalg.norm(a)
                if norm_a < 1e-9:
                    st.error("Направляющий вектор не должен быть нулевым.")
                else:
                    rho = np.linalg.norm(cross) / norm_a
                    st.markdown(f"**$[M_0 - M_1,\\ \\vec{{a}}] = {tuple(np.round(cross, 4))}$**")
                    st.markdown(f"**$\\|[M_0 - M_1,\\ \\vec{{a}}]\\| = {np.linalg.norm(cross):.4f}$**")
                    st.markdown(f"**$\\|\\vec{{a}}\\| = {norm_a:.4f}$**")
                    st.success(f"**$\\rho(M_0, \\ell) = {rho:.4f}$**")

    elif mode == "Точка -> плоскость":
        col1, col2 = st.columns(2)
        with col1:
            M0 = vector_input("Точка $M_0$", np.array([1.0, 1.0, 1.0]), key="d2_m0")
            st.markdown("Уравнение плоскости $Ax + By + Cz + D = 0$:")
            normal = vector_input("Нормаль $(A, B, C)$", np.array([1.0, 1.0, 1.0]), key="d2_n")
            D = st.number_input("Константа $D$", value=-3.0)
        with col2:
            if len(M0) == 3 and len(normal) == 3:
                num = abs(normal @ M0 + D)
                den = np.linalg.norm(normal)
                if den < 1e-9:
                    st.error("Нормаль не должна быть нулевой.")
                else:
                    rho = num / den
                    st.markdown(f"**$Ax_0 + By_0 + Cz_0 + D = {normal @ M0 + D:.4f}$**")
                    st.markdown(f"**$\\sqrt{{A^2 + B^2 + C^2}} = {den:.4f}$**")
                    st.success(f"**$\\rho(M_0, \\Pi) = {rho:.4f}$**")

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Прямая 1:** $P_1 + t \\vec{a}_1$")
            M1 = vector_input("$M_1$", np.array([0.0, 0.0, 0.0]), key="d3_m1")
            a1 = vector_input("$\\vec{a}_1$", np.array([1.0, 0.0, 0.0]), key="d3_a1")
            st.markdown("**Прямая 2:** $P_2 + s \\vec{a}_2$")
            M2 = vector_input("$M_2$", np.array([0.0, 0.0, 1.0]), key="d3_m2")
            a2 = vector_input("$\\vec{a}_2$", np.array([0.0, 1.0, 0.0]), key="d3_a2")
        with col2:
            if all(len(x) == 3 for x in [M1, a1, M2, a2]):
                cross_a = np.cross(a1, a2)
                if np.linalg.norm(cross_a) < 1e-9:
                    st.warning(
                        "Прямые параллельны или совпадают ($\\vec{a}_1, \\vec{a}_2$ коллинеарны). "
                        "Используй формулу для параллельных прямых."
                    )
                else:
                    mixed = np.linalg.det(np.array([M2 - M1, a1, a2]))
                    den = np.linalg.norm(cross_a)
                    rho = abs(mixed) / den

                    if abs(mixed) < 1e-9:
                        st.info("Прямые **пересекаются** ($(M_2 - M_1, \\vec{a}_1, \\vec{a}_2) = 0$).")
                    else:
                        st.markdown(f"**$(M_2 - M_1, \\vec{{a}}_1, \\vec{{a}}_2) = {mixed:.4f}$**")
                        st.markdown(f"**$\\|[\\vec{{a}}_1, \\vec{{a}}_2]\\| = {den:.4f}$**")
                        st.success(f"**$\\rho(\\ell_1, \\ell_2) = {rho:.4f}$**")
                        st.caption("Прямые скрещиваются (смешанное произведение ненулевое).")
