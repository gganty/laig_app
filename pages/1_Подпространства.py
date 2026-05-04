import streamlit as st
import numpy as np
import plotly.graph_objects as go

from laig.content import register, render_definition, render_theorem, filter_items, get_registry
from laig.content.section_1 import DEFS, THMS
from laig.ui import vector_input

register(DEFS)
register(THMS)


st.title("1. Подпространства")

tab_def, tab_thm, tab_demo = st.tabs(["Определения", "Теоремы", "Сумма подпространств в R³"])

with tab_def:
    query = st.text_input("🔍 Поиск", "", key="def_search_1")
    items = filter_items(DEFS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_definition(item, get_registry())

with tab_thm:
    query = st.text_input("🔍 Поиск", "", key="thm_search_1")
    items = filter_items(THMS, query)
    if not items:
        st.info("По запросу ничего не найдено.")
    for item in items:
        render_theorem(item, get_registry())
        st.markdown("---")

with tab_demo:
    st.markdown(
        "Визуализация суммы и пересечения двух подпространств в $\\mathbb{R}^3$. "
        "Подпространства задаются как линейные оболочки указанных векторов."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Подпространство $U_1$:**")
        u1 = vector_input("u₁", np.array([1.0, 0.0, 0.0]), key="sum_u1")
        u2 = vector_input("u₂ (для плоскости в $U_1$)", np.array([0.0, 1.0, 0.0]), key="sum_u2")

        st.markdown("**Подпространство $U_2$:**")
        v1 = vector_input("v₁", np.array([0.0, 1.0, 1.0]), key="sum_v1")
        v2 = vector_input("v₂", np.array([1.0, 0.0, 1.0]), key="sum_v2")

        U1_basis = np.array([u1, u2]).T
        U2_basis = np.array([v1, v2]).T
        dim_U1 = np.linalg.matrix_rank(U1_basis)
        dim_U2 = np.linalg.matrix_rank(U2_basis)

        sum_basis = np.hstack([U1_basis, U2_basis])
        dim_sum = np.linalg.matrix_rank(sum_basis)
        dim_intersect = dim_U1 + dim_U2 - dim_sum

        st.markdown(f"**dim $U_1$ = {dim_U1}, dim $U_2$ = {dim_U2}**")
        st.markdown(f"**dim($U_1 + U_2$) = {dim_sum}**")
        st.markdown(f"**dim($U_1 \\cap U_2$) = {dim_intersect}** (по формуле размерности суммы)")

    with col2:
        fig = go.Figure()

        s_range = np.linspace(-2, 2, 8)
        t_range = np.linspace(-2, 2, 8)
        S, T = np.meshgrid(s_range, t_range)

        if dim_U1 == 2:
            X1 = S * u1[0] + T * u2[0]
            Y1 = S * u1[1] + T * u2[1]
            Z1 = S * u1[2] + T * u2[2]
            fig.add_trace(go.Surface(
                x=X1, y=Y1, z=Z1, opacity=0.3, showscale=False,
                colorscale=[[0, "#534AB7"], [1, "#534AB7"]],
                name="U₁",
            ))
        elif dim_U1 == 1:
            t = np.linspace(-2, 2, 2)
            d = u1 / np.linalg.norm(u1) if np.linalg.norm(u1) > 0 else u1
            fig.add_trace(go.Scatter3d(
                x=t * d[0], y=t * d[1], z=t * d[2],
                mode="lines", line=dict(color="#534AB7", width=6), name="U₁",
            ))

        if dim_U2 == 2:
            X2 = S * v1[0] + T * v2[0]
            Y2 = S * v1[1] + T * v2[1]
            Z2 = S * v1[2] + T * v2[2]
            fig.add_trace(go.Surface(
                x=X2, y=Y2, z=Z2, opacity=0.3, showscale=False,
                colorscale=[[0, "#1D9E75"], [1, "#1D9E75"]],
                name="U₂",
            ))
        elif dim_U2 == 1:
            t = np.linspace(-2, 2, 2)
            d = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
            fig.add_trace(go.Scatter3d(
                x=t * d[0], y=t * d[1], z=t * d[2],
                mode="lines", line=dict(color="#1D9E75", width=6), name="U₂",
            ))

        fig.update_layout(
            title="$U_1$ (фиолетовое) и $U_2$ (зелёное) в R³",
            scene=dict(aspectmode="cube",
                       xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-3, 3]), zaxis=dict(range=[-3, 3])),
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
