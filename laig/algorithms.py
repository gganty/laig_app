from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """Один шаг алгоритма для отображения в UI."""
    title: str
    description: str
    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    formulas: list[str] = field(default_factory=list)


def gram_schmidt(vectors: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, list[Step]]:
    """
    Метод ортогонализации Грама-Шмидта пошагово.

    vectors: массив (k, n) — k векторов в R^n.
    Возвращает (ортогональная/ОНС, шаги).
    """
    vectors = np.asarray(vectors, dtype=float)
    k = vectors.shape[0]
    steps: list[Step] = []

    u = np.zeros_like(vectors)
    u[0] = vectors[0].copy()

    steps.append(Step(
        title="Шаг 1: u₁ = v₁",
        description=f"Берём первый вектор как есть: u₁ = v₁ = {_vec_str(u[0])}.",
        matrices={"u₁": u[0:1]},
    ))

    for j in range(1, k):
        u_j = vectors[j].copy()
        formulas = [f"u_{j+1} = v_{j+1}"]
        proj_terms = []
        for i in range(j):
            denom = float(u[i] @ u[i])
            if denom < 1e-12:
                continue
            coef = float(vectors[j] @ u[i]) / denom
            u_j = u_j - coef * u[i]
            formulas.append(
                f" − ({_num(vectors[j] @ u[i])}/{_num(denom)})·u_{i+1}"
            )
            proj_terms.append(
                f"⟨v_{j+1}, u_{i+1}⟩ = {_num(vectors[j] @ u[i])}, "
                f"⟨u_{i+1}, u_{i+1}⟩ = {_num(denom)}"
            )
        u[j] = u_j

        steps.append(Step(
            title=f"Шаг {j+1}: вычитаем проекции на u₁,…,u_{j}",
            description=(
                "Из v_{j+1} вычитаем все проекции на уже найденные u_i.\n\n"
                + "\n\n".join(proj_terms)
                + f"\n\nРезультат: u_{j+1} = {_vec_str(u_j)}"
            ),
            matrices={f"u_{j+1}": u[j:j+1]},
            formulas=["".join(formulas)],
        ))

    if normalize:
        e = np.zeros_like(u)
        for i in range(k):
            norm = np.linalg.norm(u[i])
            if norm > 1e-12:
                e[i] = u[i] / norm
        steps.append(Step(
            title="Финал: нормировка",
            description="Делим каждый u_i на его длину, получаем ОНС e_i = u_i / ‖u_i‖.",
            matrices={"e (ОНБ, по строкам)": e},
        ))
        return e, steps
    else:
        return u, steps


def jacobi_method(B: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Step]]:
    """
    Метод Якоби для симметричной матрицы B.

    Если все угловые миноры ненулевые, строит:
      - диагональную матрицу D = diag(Δ_k / Δ_{k-1})
      - верхнетреугольную C с единицами на диагонали такую, что C^T B C = D.

    Возвращает (D, C, steps).
    Если какой-то Δ_k = 0, возвращает (None, None, steps с описанием проблемы).
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    steps: list[Step] = []

    if not np.allclose(B, B.T, atol=1e-9):
        steps.append(Step(
            title="Ошибка: матрица не симметрична",
            description=f"Метод Якоби применяется только к симметричным матрицам. "
                        f"Сейчас B − B^T = ненулевая.",
            matrices={"B": B, "B^T": B.T, "B − B^T": B - B.T},
        ))
        return None, None, steps

    # Угловые миноры
    deltas = [1.0]  # Δ_0 := 1
    for k in range(1, n + 1):
        deltas.append(float(np.linalg.det(B[:k, :k])))

    minor_lines = "\n".join(
        f"Δ_{k} = det B_{k} = {_num(deltas[k])}" for k in range(n + 1)
    )
    steps.append(Step(
        title="Шаг 0: вычисляем угловые миноры",
        description=f"Δ_0 := 1 по соглашению. Затем:\n\n{minor_lines}",
        matrices={"B": B},
    ))

    if any(abs(d) < 1e-12 for d in deltas):
        steps.append(Step(
            title="Метод Якоби неприменим",
            description=(
                "Один из угловых миноров обратился в ноль. "
                "Метод Якоби в чистом виде требует Δ_k ≠ 0 для всех k = 1, …, n. "
                "Можно попытаться переставить базисные векторы или использовать "
                "симметричный алгоритм Гаусса."
            ),
        ))
        return None, None, steps

    C = np.eye(n)
    for k in range(1, n):
        rhs = -B[:k, k]
        sub = B[:k, :k]
        alpha = np.linalg.solve(sub, rhs)
        C[:k, k] = alpha
        steps.append(Step(
            title=f"Шаг {k}: строим e'_{k+1}",
            description=(
                f"Ищем e'_{k+1} = e_{k+1} + α_1·e_1 + … + α_{k}·e_{k}, "
                f"требуя b(e'_{k+1}, e_j) = 0 для j = 1, …, {k}.\n\n"
                f"Это система с матрицей B_{k} (верхний {k}×{k}-блок), "
                f"det B_{k} = {_num(deltas[k])} ≠ 0.\n\n"
                f"Решение: α = {_vec_str(alpha)}"
            ),
            matrices={f"B_{k}": sub, "rhs": rhs.reshape(-1, 1), "α": alpha.reshape(-1, 1)},
        ))

    D = C.T @ B @ C
    D_clean = np.zeros_like(D)
    for i in range(n):
        D_clean[i, i] = D[i, i]

    diag_formula = ", ".join(
        f"β_{k+1} = Δ_{k+1}/Δ_{k} = {_num(deltas[k+1])}/{_num(deltas[k])} = {_num(deltas[k+1]/deltas[k])}"
        for k in range(n)
    )
    steps.append(Step(
        title="Финал: D = C^T B C",
        description=(
            f"Получили диагональную матрицу с β_k = Δ_k/Δ_{{k-1}}:\n\n{diag_formula}"
        ),
        matrices={"C (матрица перехода)": C, "D = C^T B C": D_clean},
    ))

    return D_clean, C, steps


def svd_steps(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Step]]:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    steps: list[Step] = []

    AtA = A.T @ A
    steps.append(Step(
        title="Шаг 1: оператор φ*φ",
        description=(
            "В стандартном ОНБ матрица сопряжённого оператора — это A^T. "
            "Значит, оператор φ*φ имеет матрицу A^T A. Эта матрица симметрична "
            "и неотрицательно определена (⟨A^T A v, v⟩ = ‖Av‖² ≥ 0)."
        ),
        matrices={"A": A, "A^T A": AtA},
    ))

    eigvals, eigvecs = np.linalg.eigh(AtA)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.where(eigvals < 1e-12, 0.0, eigvals)

    eig_lines = ", ".join(f"μ_{i+1} = {_num(v)}" for i, v in enumerate(eigvals))
    steps.append(Step(
        title="Шаг 2: спектр A^T A",
        description=(
            f"По спектральной теореме A^T A имеет ОНБ из собственных векторов "
            f"с неотрицательными собственными значениями:\n\n{eig_lines}\n\n"
            f"Векторы в столбцах матрицы V."
        ),
        matrices={"V (собственные векторы по столбцам)": eigvecs,
                  "Λ (диагональ собственных значений)": np.diag(eigvals)},
    ))

    sigmas = np.sqrt(eigvals)
    r = int(np.sum(sigmas > 1e-9))
    sigma_lines = ", ".join(f"σ_{i+1} = √μ_{i+1} = {_num(s)}" for i, s in enumerate(sigmas))
    steps.append(Step(
        title="Шаг 3: сингулярные числа σᵢ = √μᵢ",
        description=(
            f"{sigma_lines}\n\nРанг A равен числу ненулевых σᵢ: rk A = {r}."
        ),
        matrices={"σ": sigmas.reshape(-1, 1)},
    ))

    V = eigvecs
    U = np.zeros((m, m))
    for i in range(r):
        U[:, i] = A @ V[:, i] / sigmas[i]

    if r < m:
        Q, _ = np.linalg.qr(np.column_stack([U[:, :r], np.eye(m)]))
        for i in range(r, m):
            candidate = np.zeros(m)
            candidate[i % m] = 1.0
            for j in range(m):
                candidate = np.zeros(m)
                candidate[j] = 1.0
                for k in range(i):
                    candidate = candidate - (candidate @ U[:, k]) * U[:, k]
                if np.linalg.norm(candidate) > 1e-9:
                    candidate = candidate / np.linalg.norm(candidate)
                    U[:, i] = candidate
                    break

    steps.append(Step(
        title=f"Шаг 4: левые сингулярные векторы fᵢ = φ(eᵢ) / σᵢ",
        description=(
            f"Для i = 1, …, {r}: f_i = A·v_i / σ_i. "
            f"Эти векторы автоматически ортонормированы:\n\n"
            f"⟨f_i, f_j⟩ = ⟨A v_i, A v_j⟩ / (σ_i σ_j) = ⟨A^T A v_i, v_j⟩ / (σ_i σ_j) "
            f"= μ_i ⟨v_i, v_j⟩ / (σ_i σ_j) = δ_ij.\n\n"
            f"Дополняем до ОНБ R^{m} (если r < m), чтобы U была квадратной ортогональной."
        ),
        matrices={"U (по столбцам)": U},
    ))

    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = sigmas[i] if i < len(sigmas) else 0

    A_reconstructed = U @ Sigma @ V.T
    err = np.linalg.norm(A - A_reconstructed)

    steps.append(Step(
        title="Финал: A = U Σ V^T",
        description=(
            f"Собираем разложение. Проверка:\n\n"
            f"‖A − UΣV^T‖_F = {_num(err)} ≈ 0 (численная погрешность)."
        ),
        matrices={"U": U, "Σ": Sigma, "V^T": V.T, "U·Σ·V^T (проверка)": A_reconstructed},
    ))

    return U, Sigma, V, steps


def operator_spectrum(A: np.ndarray) -> dict[str, Any]:
    """
    Спектральный анализ оператора с матрицей A.

    Возвращает словарь с:
      - char_poly_coeffs: коэффициенты характеристического многочлена
      - eigenvalues: список (значение, алг. кратность, геом. кратность,
        собственные векторы)
      - is_diagonalizable: bool
      - reasons: list[str] — почему да/нет
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    result: dict[str, Any] = {}

    char_coefs = np.poly(A)
    result["char_poly_coeffs"] = char_coefs

    roots = np.roots(char_coefs)

    eigenvalues = []
    used = [False] * len(roots)
    for i, r in enumerate(roots):
        if used[i]:
            continue
        algebraic_mult = 1
        used[i] = True
        for j in range(i + 1, len(roots)):
            if not used[j] and abs(roots[j] - r) < 1e-6:
                algebraic_mult += 1
                used[j] = True

        if abs(r.imag) < 1e-9:
            r_real = r.real
            ker_matrix = A - r_real * np.eye(n)
            rank = np.linalg.matrix_rank(ker_matrix, tol=1e-8)
            geometric_mult = n - rank
            _, _, vh = np.linalg.svd(ker_matrix)
            null_space = vh[rank:].T
            eigenvalues.append({
                "value": r_real,
                "is_real": True,
                "algebraic_mult": algebraic_mult,
                "geometric_mult": geometric_mult,
                "eigenvectors": null_space,
            })
        else:
            eigenvalues.append({
                "value": complex(r),
                "is_real": False,
                "algebraic_mult": algebraic_mult,
                "geometric_mult": 0,
                "eigenvectors": None,
            })

    result["eigenvalues"] = eigenvalues

    all_real = all(e["is_real"] for e in eigenvalues)
    if not all_real:
        result["is_diagonalizable"] = False
        result["reasons"] = [
            "Над R: характеристический многочлен имеет комплексные корни.",
            "Над C: проверка кратностей нужна отдельно.",
        ]
    else:
        kratnosti_ok = all(e["algebraic_mult"] == e["geometric_mult"] for e in eigenvalues)
        result["is_diagonalizable"] = kratnosti_ok
        if kratnosti_ok:
            result["reasons"] = [
                "Все корни вещественные.",
                "Для каждого собственного значения геом. кратность = алг. кратности.",
            ]
        else:
            result["reasons"] = [
                "Все корни вещественные.",
                "Но для некоторых λ геом. кратность < алг. кратности.",
            ]

    return result


def least_squares(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, list[Step]]:
    """
    Метод наименьших квадратов: решить A x ≈ b в смысле min ‖Ax − b‖².

    Возвращает (x, steps).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    steps: list[Step] = []

    AtA = A.T @ A
    Atb = A.T @ b

    steps.append(Step(
        title="Шаг 1: нормальное уравнение A^T A x = A^T b",
        description=(
            "Псевдорешение минимизирует ‖Ax − b‖². Условие минимума: "
            "b − Ax ⊥ Im A, что эквивалентно A^T(b − Ax) = 0, "
            "или A^T A x = A^T b."
        ),
        matrices={"A": A, "b": b.reshape(-1, 1), "A^T A": AtA, "A^T b": Atb.reshape(-1, 1)},
    ))

    det_AtA = np.linalg.det(AtA)
    rank_A = np.linalg.matrix_rank(A)

    if rank_A < A.shape[1]:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        steps.append(Step(
            title="Шаг 2: столбцы A линейно зависимы",
            description=(
                "rk A < число столбцов, значит, A^T A вырождена и псевдорешений "
                "бесконечно много. Среди них выделяется нормальное псевдорешение "
                "минимальной нормы — оно вычисляется через псевдообратную матрицу A⁺."
            ),
            matrices={"x (норм. псевдорешение)": x.reshape(-1, 1)},
        ))
    else:
        x = np.linalg.solve(AtA, Atb)
        steps.append(Step(
            title="Шаг 2: решаем нормальное уравнение",
            description=(
                f"Столбцы A линейно независимы (rk A = {rank_A} = число столбцов), "
                f"значит, A^T A обратима (det = {_num(det_AtA)}).\n\n"
                f"x = (A^T A)⁻¹ A^T b"
            ),
            matrices={"x = (A^T A)⁻¹ A^T b": x.reshape(-1, 1)},
        ))

    residual = b - A @ x
    steps.append(Step(
        title="Шаг 3: проверка",
        description=(
            f"Невязка r = b − Ax. Её норма ‖r‖ = {_num(np.linalg.norm(residual))}.\n\n"
            f"Должно быть r ⊥ Im A, т.\u202fе. A^T r ≈ 0."
        ),
        matrices={"r = b − Ax": residual.reshape(-1, 1), "A^T r": (A.T @ residual).reshape(-1, 1)},
    ))

    return x, steps


def _num(x: float, precision: int = 4) -> str:
    """Аккуратный вывод числа: целое если близко к целому, иначе с заданной точностью."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{precision}g}"


def _vec_str(v: np.ndarray) -> str:
    """Вектор в виде ‘(1, 2, 3)’."""
    return "(" + ", ".join(_num(x) for x in v.flat) + ")"
