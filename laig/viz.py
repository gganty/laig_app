"""
Геометрические визуализации в 2D и 3D через Plotly.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def _arrow_2d(fig: go.Figure, start: np.ndarray, end: np.ndarray,
              color: str, name: str, width: int = 3) -> None:
    """Рисует стрелку из start в end в 2D."""
    fig.add_trace(go.Scatter(
        x=[start[0], end[0]], y=[start[1], end[1]],
        mode="lines+markers",
        line=dict(color=color, width=width),
        marker=dict(size=[1, 12], symbol=["circle", "triangle-up"],
                    color=color, angle=[0, _angle(end - start) - 90]),
        name=name,
        showlegend=True,
    ))


def _arrow_3d(fig: go.Figure, start: np.ndarray, end: np.ndarray,
              color: str, name: str) -> None:
    """Стрелка в 3D через линию + конус на конце."""
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines",
        line=dict(color=color, width=6),
        name=name,
        showlegend=True,
    ))
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        d = direction / norm
        fig.add_trace(go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[d[0]], v=[d[1]], w=[d[2]],
            sizemode="absolute", sizeref=norm * 0.15,
            colorscale=[[0, color], [1, color]], showscale=False,
            anchor="tip", showlegend=False,
        ))


def _angle(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def projection_2d(v: np.ndarray, u: np.ndarray) -> go.Figure:
    """
    Визуализация проекции вектора v на одномерное подпространство ⟨u⟩.
    """
    v = np.asarray(v, dtype=float).flatten()[:2]
    u = np.asarray(u, dtype=float).flatten()[:2]

    if np.linalg.norm(u) < 1e-12:
        u = np.array([1.0, 0.0])

    coef = (v @ u) / (u @ u)
    proj = coef * u
    perp = v - proj

    fig = go.Figure()

    # Линия подпространства ⟨u⟩
    t = np.linspace(-3, 3, 2)
    fig.add_trace(go.Scatter(
        x=t * u[0], y=t * u[1],
        mode="lines", line=dict(color="rgba(150,150,150,0.5)", dash="dash", width=1),
        name="⟨u⟩", showlegend=True,
    ))

    _arrow_2d(fig, np.zeros(2), v, "#534AB7", "v")
    _arrow_2d(fig, np.zeros(2), u, "#888780", "u")
    _arrow_2d(fig, np.zeros(2), proj, "#1D9E75", "pr_U(v)")
    _arrow_2d(fig, proj, v, "#D85A30", "v − pr_U(v)", width=2)

    fig.update_layout(
        title="Ортогональная проекция",
        xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor="rgba(0,0,0,0.1)", zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", zerolinecolor="rgba(0,0,0,0.3)"),
        showlegend=True,
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
    )
    return fig


def projection_3d_to_plane(v: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> go.Figure:
    """
    Проекция вектора v на двумерное подпространство ⟨u1, u2⟩.
    """
    v = np.asarray(v, dtype=float).flatten()[:3]
    u1 = np.asarray(u1, dtype=float).flatten()[:3]
    u2 = np.asarray(u2, dtype=float).flatten()[:3]

    # Грам-Шмидт для проекции
    e1 = u1 / np.linalg.norm(u1)
    u2_perp = u2 - (u2 @ e1) * e1
    if np.linalg.norm(u2_perp) < 1e-9:
        e2 = np.zeros(3)
    else:
        e2 = u2_perp / np.linalg.norm(u2_perp)

    proj = (v @ e1) * e1 + (v @ e2) * e2
    perp = v - proj

    fig = go.Figure()

    # Плоскость
    s_range = np.linspace(-2, 2, 10)
    t_range = np.linspace(-2, 2, 10)
    S, T = np.meshgrid(s_range, t_range)
    X = S * u1[0] + T * u2[0]
    Y = S * u1[1] + T * u2[1]
    Z = S * u1[2] + T * u2[2]
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.3, showscale=False,
        colorscale=[[0, "#AFA9EC"], [1, "#AFA9EC"]],
        name="⟨u₁, u₂⟩",
    ))

    _arrow_3d(fig, np.zeros(3), v, "#534AB7", "v")
    _arrow_3d(fig, np.zeros(3), u1, "#888780", "u₁")
    _arrow_3d(fig, np.zeros(3), u2, "#888780", "u₂")
    _arrow_3d(fig, np.zeros(3), proj, "#1D9E75", "pr_U(v)")
    _arrow_3d(fig, proj, v, "#D85A30", "v − pr")

    fig.update_layout(
        title="Проекция на плоскость в R³",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="data",
        ),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def operator_action_2d(A: np.ndarray) -> go.Figure:
    """
    Действие линейного оператора 2×2 на единичный круг и стандартный базис.
    Показывает, как круг превращается в эллипс (визуальное представление SVD).
    """
    A = np.asarray(A, dtype=float)[:2, :2]

    # Точки круга
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    image = A @ circle

    # Стандартные базисные векторы
    e1, e2 = np.array([1, 0]), np.array([0, 1])
    Ae1, Ae2 = A @ e1, A @ e2

    fig = go.Figure()
    # Исходный круг
    fig.add_trace(go.Scatter(
        x=circle[0], y=circle[1], mode="lines",
        line=dict(color="#888780", dash="dash", width=2),
        name="Единичный круг",
    ))
    # Образ
    fig.add_trace(go.Scatter(
        x=image[0], y=image[1], mode="lines",
        line=dict(color="#534AB7", width=3),
        fill="toself", fillcolor="rgba(83, 74, 183, 0.1)",
        name="Образ",
    ))
    _arrow_2d(fig, np.zeros(2), e1, "#888780", "e₁")
    _arrow_2d(fig, np.zeros(2), e2, "#888780", "e₂")
    _arrow_2d(fig, np.zeros(2), Ae1, "#0F6E56", "Ae₁")
    _arrow_2d(fig, np.zeros(2), Ae2, "#993C1D", "Ae₂")

    # Если есть вещественные собственные векторы — показываем их
    eigvals, eigvecs = np.linalg.eig(A)
    for i, lam in enumerate(eigvals):
        if abs(lam.imag) < 1e-9:
            v = eigvecs[:, i].real
            v = v / np.linalg.norm(v)
            t = np.linspace(-2, 2, 2)
            fig.add_trace(go.Scatter(
                x=t * v[0], y=t * v[1],
                mode="lines",
                line=dict(color="#EF9F27", width=1.5, dash="dot"),
                name=f"⟨v⟩, λ={lam.real:.2f}",
            ))

    fig.update_layout(
        title="Действие оператора A на единичный круг",
        xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        showlegend=True,
        height=500,
        plot_bgcolor="white",
    )
    return fig


def cross_product_3d(u: np.ndarray, v: np.ndarray) -> go.Figure:
    """Визуализация векторного произведения [u, v]."""
    u = np.asarray(u, dtype=float).flatten()[:3]
    v = np.asarray(v, dtype=float).flatten()[:3]
    w = np.cross(u, v)

    fig = go.Figure()

    _arrow_3d(fig, np.zeros(3), u, "#534AB7", "u")
    _arrow_3d(fig, np.zeros(3), v, "#1D9E75", "v")
    _arrow_3d(fig, np.zeros(3), w, "#D85A30", "[u, v]")

    # Параллелограмм на u, v
    para_x = [0, u[0], u[0] + v[0], v[0], 0]
    para_y = [0, u[1], u[1] + v[1], v[1], 0]
    para_z = [0, u[2], u[2] + v[2], v[2], 0]
    fig.add_trace(go.Scatter3d(
        x=para_x, y=para_y, z=para_z,
        mode="lines", line=dict(color="rgba(83,74,183,0.5)", width=2),
        name="параллелограмм",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"[u, v]; ‖[u,v]‖ = площадь параллелограмма = {np.linalg.norm(w):.3f}",
        scene=dict(aspectmode="data"),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def quadric_section_2d(B: np.ndarray) -> go.Figure:
    """
    Линии уровня квадратичной формы q(x) = x^T B x в R².
    Визуализирует тип формы (эллипс / гипербола / параболическая).
    """
    B = np.asarray(B, dtype=float)[:2, :2]
    B = (B + B.T) / 2  # симметризуем на всякий случай

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Q = B[0, 0] * X * X + 2 * B[0, 1] * X * Y + B[1, 1] * Y * Y

    fig = go.Figure(data=go.Contour(
        x=x, y=y, z=Q,
        contours=dict(start=-5, end=5, size=0.5, showlabels=True),
        colorscale="RdBu_r",
        zmid=0,
    ))

    # Главные оси (собственные векторы B)
    eigvals, eigvecs = np.linalg.eigh(B)
    for i, lam in enumerate(eigvals):
        v = eigvecs[:, i]
        t = np.linspace(-3, 3, 2)
        fig.add_trace(go.Scatter(
            x=t * v[0], y=t * v[1],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name=f"гл. ось λ={lam:.2f}",
        ))

    fig.update_layout(
        title="Линии уровня q(x) = xᵀ B x; пунктир — главные оси",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(),
        height=500,
        plot_bgcolor="white",
    )
    return fig


def rotation_2d(theta_deg: float) -> go.Figure:
    """Визуализация поворота на угол theta."""
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return operator_action_2d(R)
