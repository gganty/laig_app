import numpy as np
import streamlit as st
import pandas as pd

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_6 import DEFS, THMS
from laig.ui import matrix_input
from laig.algorithms import operator_spectrum
from laig.viz import operator_action_2d

register(DEFS)
register(THMS)


st.title("6. Линейные операторы")

tab_def, tab_thm, tab_spec = st.tabs(["Определения", "Теоремы", "Спектральный анализ"])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_6")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_6")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_spec:
    st.markdown(
        "Введи матрицу оператора. Получишь характеристический многочлен, "
        "собственные значения, кратности, диагонализуемость."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        default_A = np.array([
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        A = matrix_input("Матрица оператора $A$", default_A, key="op_A")

        if A is not None and A.shape[0] == A.shape[1]:
            st.markdown(f"**Размер:** ${A.shape[0]} \\times {A.shape[0]}$")
            st.markdown(f"**$\\det A = {np.linalg.det(A):.4g}$**")
            st.markdown(f"**$\\operatorname{{tr}} A = {np.trace(A):.4g}$**")

    with col2:
        if A is not None and A.shape[0] == A.shape[1]:
            try:
                spec = operator_spectrum(A)

                coefs = spec["char_poly_coeffs"]
                n = len(coefs) - 1
                terms = []
                for i, c in enumerate(coefs):
                    power = n - i
                    if abs(c) < 1e-9:
                        continue
                    sign = "+" if c >= 0 else "−"
                    abs_c = abs(c)
                    if power == 0:
                        terms.append(f"{sign} {abs_c:.4g}")
                    elif power == 1:
                        if abs(abs_c - 1) < 1e-9:
                            terms.append(f"{sign} t")
                        else:
                            terms.append(f"{sign} {abs_c:.4g} \\cdot t")
                    else:
                        if abs(abs_c - 1) < 1e-9:
                            terms.append(f"{sign} t^{{{power}}}")
                        else:
                            terms.append(f"{sign} {abs_c:.4g} \\cdot t^{{{power}}}")
                poly_str = " ".join(terms).lstrip("+ ").lstrip()
                st.markdown(f"**$\\chi_\\varphi(t) = {poly_str}$**")

                st.markdown("**Собственные значения:**")
                rows = []
                for e in spec["eigenvalues"]:
                    if e["is_real"]:
                        rows.append({
                            "λ": f"{e['value']:.4g}",
                            "m_a": e["algebraic_mult"],
                            "m_g": e["geometric_mult"],
                            "Состояние": "✅" if e["algebraic_mult"] == e["geometric_mult"] else "⚠️ m_g < m_a",
                        })
                    else:
                        rows.append({
                            "λ": f"{e['value'].real:.4g} ± {abs(e['value'].imag):.4g}i",
                            "m_a": e["algebraic_mult"],
                            "m_g": "— (комплексное)",
                            "Состояние": "❌ Над R нет соб. вектора",
                        })
                st.dataframe(pd.DataFrame(rows), hide_index=True)

                if spec["is_diagonalizable"]:
                    st.success("**Оператор диагонализуем** (над $\\mathbb{R}$).")
                else:
                    st.warning("**Оператор не диагонализуем** (над $\\mathbb{R}$).")
                for r in spec["reasons"]:
                    st.caption(f"— {r}")

            except Exception as e:
                st.error(f"Ошибка: {e}")

    if A is not None and A.shape == (2, 2):
        st.markdown("---")
        st.markdown("### Геометрия в 2D")
        st.markdown(
            "Действие оператора на единичный круг. Если есть вещественные собственные "
            "векторы — они показаны пунктиром."
        )
        fig = operator_action_2d(A)
        st.plotly_chart(fig, use_container_width=True)
