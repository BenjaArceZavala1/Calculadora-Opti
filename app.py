"""
Proyecto Final — Métodos de Optimización
Aplicación web para encontrar mínimos de funciones mediante:
  - Método del Gradiente (Steepest Descent)
  - Método del Gradiente Conjugado (Fletcher-Reeves)
  - Método de Newton
Todos con condiciones de Wolfe (Armijo + Curvatura).
"""

import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd
import traceback
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimización — Métodos Numéricos",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .block-container { padding-top: 1.5rem; }
    .stAlert { border-radius: 8px; }
    code { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("📉 Calculadora de Mínimos")
st.caption("Gradiente · Gradiente Conjugado · Newton — con condiciones de Wolfe")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Panel lateral — Entradas del usuario
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros")

    n_vars = st.number_input(
        "Número de variables (n)", min_value=1, max_value=20, value=2, step=1
    )
    n_vars = int(n_vars)

    method = st.selectbox(
        "Método de optimización",
        [
            "Gradiente (Steepest Descent)",
            "Gradiente Conjugado (Fletcher-Reeves)",
            "Método de Newton",
        ],
    )

    default_fn = (
        "(x1 - 2)**2 + (x2 - 3)**2" if n_vars >= 2 else "(x1 - 2)**2"
    )
    func_input = st.text_area(
        "Función objetivo f(x)",
        value=default_fn,
        height=90,
        help="Usa x1, x2, ..., xn. Ej: 100*(x2-x1**2)**2 + (1-x1)**2",
    )

    x0_input = st.text_input(
        "Punto de partida x₀ (separado por comas)",
        value=", ".join(["0"] * n_vars),
        help="Ejemplo para n=2: 0, 0",
    )

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        max_iter = int(
            st.number_input("Iter. máx.", min_value=10, max_value=100_000, value=1000, step=100)
        )
    with col_r:
        tol_exp = st.number_input("Tol. (10ⁿ)", min_value=-15, max_value=-1, value=-6, step=1)
    tol = 10.0 ** tol_exp
    st.caption(f"Tolerancia: `{tol:.0e}`")

    st.divider()
    st.subheader("Parámetros de Wolfe")
    c1 = st.number_input(
        "c₁ — Armijo",
        min_value=1e-10, max_value=0.49, value=1e-4, format="%.2e",
        help="Condición de disminución suficiente. Rango válido: 0 < c₁ < c₂ < 1. Típico: 1e-4",
    )
    c2 = st.number_input(
        "c₂ — Curvatura",
        min_value=0.01, max_value=0.9999, value=0.9, format="%.4f",
        help="Condición de curvatura. Típico: 0.9 para GD/Newton, 0.1 para GC",
    )
    alpha_init = st.number_input(
        "α inicial (paso máximo)",
        min_value=1e-6, max_value=100.0, value=1.0, format="%.4f",
    )

    st.divider()
    compare_mode = st.checkbox(
        "🔀 Comparar los 3 métodos",
        value=False,
        help="Ejecuta los 3 métodos con los mismos parámetros y compara su convergencia",
    )

    run_btn = st.button("▶ Ejecutar", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Parseo simbólico de la función
# ─────────────────────────────────────────────────────────────────────────────

def build_functions(func_str: str, n: int):
    """
    Construye funciones numéricas para f, ∇f y ∇²f a partir de una
    expresión simbólica en cadena de texto.
    """
    # Crear símbolos x1, x2, ..., xn
    syms = sp.symbols(" ".join(f"x{i+1}" for i in range(n)))
    if n == 1:
        syms = (syms,)

    # Contexto local para sympify
    local_dict = {f"x{i+1}": syms[i] for i in range(n)}
    local_dict.update({
        "e": sp.E, "pi": sp.pi,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "abs": sp.Abs,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    })

    expr = sp.sympify(func_str, locals=local_dict)

    # Gradiente simbólico
    grad_exprs = [sp.diff(expr, v) for v in syms]

    # Hessiana simbólica
    hess_exprs = [
        [sp.diff(grad_exprs[i], syms[j]) for j in range(n)]
        for i in range(n)
    ]

    # Lambdify para evaluación numérica rápida
    f_lam = sp.lambdify(syms, expr, modules="numpy")
    grad_lams = [sp.lambdify(syms, ge, modules="numpy") for ge in grad_exprs]
    hess_lams = [
        [sp.lambdify(syms, hess_exprs[i][j], modules="numpy") for j in range(n)]
        for i in range(n)
    ]

    def f(x):
        val = float(f_lam(*x))
        if not np.isfinite(val):
            raise ValueError("f(x) no es finita en este punto.")
        return val

    def grad_f(x):
        g = np.array([float(gl(*x)) for gl in grad_lams], dtype=float)
        if not np.all(np.isfinite(g)):
            raise ValueError("∇f(x) contiene valores no finitos.")
        return g

    def hess_f(x):
        H = np.array(
            [[float(hess_lams[i][j](*x)) for j in range(n)] for i in range(n)],
            dtype=float,
        )
        return H

    return f, grad_f, hess_f


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda de línea con condiciones de Wolfe (bracket + zoom)
# ─────────────────────────────────────────────────────────────────────────────

def wolfe_line_search(f, grad_f, x, d, c1, c2, alpha_max=1.0, max_ls=60):
    """
    Algoritmo estándar de bracket + zoom para las condiciones fuertes de Wolfe:
      1. Armijo:    f(x+αd) ≤ f(x) + c₁·α·∇f·d
      2. Curvatura: |∇f(x+αd)·d| ≤ c₂·|∇f(x)·d|
    """
    phi0 = f(x)
    g0 = grad_f(x)
    dphi0 = float(np.dot(g0, d))

    if dphi0 >= 0:
        return 1e-10  # No es dirección de descenso

    def phi(a):
        try:
            return f(x + a * d)
        except Exception:
            return np.inf

    def dphi(a):
        try:
            return float(np.dot(grad_f(x + a * d), d))
        except Exception:
            return np.inf

    def zoom(a_lo, a_hi, phi_lo):
        for _ in range(40):
            a_j = 0.5 * (a_lo + a_hi)
            phi_j = phi(a_j)

            if phi_j > phi0 + c1 * a_j * dphi0 or phi_j >= phi_lo:
                a_hi = a_j
            else:
                dphi_j = dphi(a_j)
                if abs(dphi_j) <= -c2 * dphi0:
                    return a_j
                if dphi_j * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo, phi_lo = a_j, phi_j

            if abs(a_hi - a_lo) < 1e-15:
                break
        return 0.5 * (a_lo + a_hi)

    alpha_prev, phi_prev = 0.0, phi0
    alpha = min(alpha_max, 1.0)

    for i in range(max_ls):
        phi_a = phi(alpha)

        if not np.isfinite(phi_a) or phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
            return zoom(alpha_prev, alpha, phi_prev)

        dphi_a = dphi(alpha)
        if abs(dphi_a) <= -c2 * dphi0:
            return alpha
        if dphi_a >= 0:
            return zoom(alpha, alpha_prev, phi_a)

        alpha_prev, phi_prev = alpha, phi_a
        alpha = min(2.0 * alpha, alpha_max)

    return max(alpha, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Métodos de optimización
# ─────────────────────────────────────────────────────────────────────────────

def steepest_descent(f, grad_f, x0, c1, c2, alpha_max, max_iter, tol):
    """Método del Gradiente (Steepest Descent) con condiciones de Wolfe."""
    x = np.array(x0, dtype=float)
    errors, trajectory = [], [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        ng = np.linalg.norm(g)
        errors.append(ng)

        if ng < tol:
            return x, f(x), k + 1, errors, trajectory, "||∇f|| < tolerancia ✓"

        d = -g
        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, alpha_max)
        x = x + alpha * d
        trajectory.append(x.copy())

    return x, f(x), max_iter, errors, trajectory, "Máximo de iteraciones alcanzado"


def conjugate_gradient_fr(f, grad_f, x0, c1, c2, alpha_max, max_iter, tol):
    """Método del Gradiente Conjugado (Fletcher-Reeves) con condiciones de Wolfe."""
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g.copy()
    errors, trajectory = [], [x.copy()]
    n = len(x0)

    for k in range(max_iter):
        ng = np.linalg.norm(g)
        errors.append(ng)

        if ng < tol:
            return x, f(x), k + 1, errors, trajectory, "||∇f|| < tolerancia ✓"

        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, alpha_max)
        x_new = x + alpha * d
        g_new = grad_f(x_new)

        # β de Fletcher-Reeves
        gg_old = np.dot(g, g)
        beta = np.dot(g_new, g_new) / max(gg_old, 1e-30)

        # Reinicio cada n pasos (estabilidad)
        if (k + 1) % n == 0:
            beta = 0.0

        d_new = -g_new + beta * d

        # Verificar que sea dirección de descenso
        if np.dot(d_new, g_new) >= 0:
            d_new = -g_new

        x, g, d = x_new, g_new, d_new
        trajectory.append(x.copy())

    return x, f(x), max_iter, errors, trajectory, "Máximo de iteraciones alcanzado"


def newton_method(f, grad_f, hess_f, x0, c1, c2, alpha_max, max_iter, tol):
    """Método de Newton con regularización y condiciones de Wolfe."""
    x = np.array(x0, dtype=float)
    errors, trajectory = [], [x.copy()]
    n = len(x0)

    for k in range(max_iter):
        g = grad_f(x)
        ng = np.linalg.norm(g)
        errors.append(ng)

        if ng < tol:
            return x, f(x), k + 1, errors, trajectory, "||∇f|| < tolerancia ✓"

        H = hess_f(x)

        # Regularización si la Hessiana no es definida positiva
        eigs = np.linalg.eigvalsh(H)
        min_eig = eigs.min()
        if min_eig <= 1e-10:
            reg = abs(min_eig) + 1e-6
            H = H + reg * np.eye(n)

        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            d = -g  # Fallback a gradiente si Hessiana singular

        # Verificar que sea dirección de descenso
        if np.dot(d, g) >= 0:
            d = -g

        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, alpha_max)
        x = x + alpha * d
        trajectory.append(x.copy())

    return x, f(x), max_iter, errors, trajectory, "Máximo de iteraciones alcanzado"


def run_method(method_name, f, grad_f, hess_f, x0, c1, c2, alpha_max, max_iter, tol):
    """Despacha al método seleccionado."""
    if method_name == "Gradiente (Steepest Descent)":
        return steepest_descent(f, grad_f, x0, c1, c2, alpha_max, max_iter, tol)
    elif method_name == "Gradiente Conjugado (Fletcher-Reeves)":
        return conjugate_gradient_fr(f, grad_f, x0, c1, c2, alpha_max, max_iter, tol)
    else:
        return newton_method(f, grad_f, hess_f, x0, c1, c2, alpha_max, max_iter, tol)


# ─────────────────────────────────────────────────────────────────────────────
# Pantalla principal — instrucciones o resultados
# ─────────────────────────────────────────────────────────────────────────────

if not run_btn:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📌 ¿Cómo usar la aplicación?")
        st.markdown("""
**1. Variables:** usa `x1`, `x2`, ..., `xn` en la función.

**2. Operadores soportados:**
- Potencia: `x1**2`
- Funciones: `exp(x1)`, `log(x1)`, `sin(x1)`, `cos(x2)`, `sqrt(x1)`
- Valor absoluto: `abs(x1)`

**3. Ejemplos de funciones:**
| Nombre | Expresión |
|--------|-----------|
| Paraboloide | `(x1-2)**2 + (x2-3)**2` |
| Rosenbrock | `100*(x2-x1**2)**2 + (1-x1)**2` |
| Himmelblau | `(x1**2+x2-11)**2 + (x1+x2**2-7)**2` |
| Cuadrática | `x1**2 + 2*x2**2 + x1*x2` |
| 3 variables | `x1**2 + x2**2 + x3**2 + x1*x2` |

**4. Punto de partida:** valores separados por coma, ej: `0, 0`
""")

    with col_b:
        st.subheader("📐 Condiciones de Wolfe")
        st.markdown(r"""
La búsqueda de línea encuentra α que satisface **ambas** condiciones:

**Primera condición (Armijo — disminución suficiente):**
$$f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f_k^\top d_k$$

**Segunda condición (Curvatura):**
$$|\nabla f(x_k + \alpha d_k)^\top d_k| \leq c_2 |\nabla f_k^\top d_k|$$

**Valores recomendados:**
- `c₁ = 1e-4` (siempre pequeño, controla Armijo)
- `c₂ = 0.9` para Gradiente y Newton
- `c₂ = 0.1` para Gradiente Conjugado (más restrictivo)
- `0 < c₁ < c₂ < 1` debe cumplirse siempre
""")

    st.info("👈 **Configura los parámetros en el panel izquierdo y presiona ▶ Ejecutar**")

else:
    # ── Validar entradas ──────────────────────────────────────────────────────
    try:
        x0 = [float(v.strip()) for v in x0_input.split(",")]
    except ValueError:
        st.error("⚠️ El punto de partida debe ser números separados por coma. Ej: `0, 0`")
        st.stop()

    if len(x0) != n_vars:
        st.error(f"⚠️ El punto de partida debe tener exactamente **{n_vars}** valores (tienes {len(x0)}).")
        st.stop()

    if not (0 < c1 < c2 < 1):
        st.error("⚠️ Los parámetros de Wolfe deben cumplir: **0 < c₁ < c₂ < 1**")
        st.stop()

    # ── Compilar función simbólica ────────────────────────────────────────────
    with st.spinner("⏳ Compilando función simbólica y calculando gradiente/Hessiana..."):
        try:
            f_func, grad_func, hess_func = build_functions(func_input, n_vars)
            # Test rápido en x0
            _ = f_func(x0)
            _ = grad_func(x0)
        except Exception as e:
            st.error(f"❌ Error al parsear la función: **{e}**")
            st.markdown("Revisa que la sintaxis sea correcta. Ej: `(x1-2)**2 + x2**2`")
            st.stop()

    # ── Ejecutar métodos ──────────────────────────────────────────────────────
    methods_to_run = (
        [
            "Gradiente (Steepest Descent)",
            "Gradiente Conjugado (Fletcher-Reeves)",
            "Método de Newton",
        ]
        if compare_mode
        else [method]
    )

    results = {}
    for m in methods_to_run:
        with st.spinner(f"⚙️ Ejecutando {m}..."):
            try:
                res = run_method(
                    m, f_func, grad_func, hess_func,
                    x0, c1, c2, alpha_init, max_iter, tol
                )
                results[m] = res
            except Exception as e:
                st.warning(f"⚠️ {m}: {e}")

    if not results:
        st.error("No se pudo ejecutar ningún método. Revisa la función y los parámetros.")
        st.stop()

    # ── Mostrar resultados ────────────────────────────────────────────────────
    for m_name, (x_min, f_min, iters, errors, traj, msg) in results.items():
        if compare_mode:
            st.subheader(f"📌 {m_name}")
            st.divider()

        # Métricas principales
        convergió = "✓" in msg
        if convergió:
            st.success(f"✅ {msg} — **{iters}** iteraciones")
        else:
            st.warning(f"⚠️ {msg} — **{iters}** iteraciones")

        cols = st.columns(4)
        cols[0].metric("Iteraciones realizadas", iters)
        cols[1].metric("f(x*)", f"{f_min:.6e}")
        cols[2].metric("||∇f|| final", f"{errors[-1]:.2e}")
        cols[3].metric("Tolerancia objetivo", f"{tol:.0e}")

        # Punto mínimo
        with st.expander("📍 Punto mínimo encontrado x*", expanded=True):
            df_pt = pd.DataFrame(
                {"Variable": [f"x{i+1}" for i in range(len(x_min))], "Valor óptimo": x_min}
            )
            st.dataframe(df_pt, hide_index=True, use_container_width=True)

        # Gráfico de convergencia
        st.subheader("📈 Gráfico de convergencia — ||∇f(xₖ)|| vs iteración")
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=list(range(1, len(errors) + 1)),
            y=errors,
            mode="lines",
            name="||∇f(xₖ)||",
            line=dict(color="#2563EB", width=2.5),
        ))
        fig_cv.add_hline(
            y=tol, line_dash="dash", line_color="#DC2626",
            annotation_text=f"tol = {tol:.0e}",
            annotation_position="bottom right",
        )
        fig_cv.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="||∇f(xₖ)|| (escala log)",
            yaxis_type="log",
            title=f"Convergencia — {m_name}",
            template="plotly_white",
            height=400,
            margin=dict(t=50, b=50),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        # Tabla de convergencia (descargable)
        df_conv = pd.DataFrame({
            "Iteracion": range(1, len(errors) + 1),
            "Norma_Gradiente": errors,
        })
        st.download_button(
            label=f"📥 Descargar datos de convergencia ({m_name.split('(')[0].strip()})",
            data=df_conv.to_csv(index=False),
            file_name=f"convergencia_{m_name[:8].replace(' ','_')}.csv",
            mime="text/csv",
        )

        if compare_mode:
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico comparativo (modo comparación) ────────────────────────────────
    if compare_mode and len(results) > 1:
        st.divider()
        st.subheader("📊 Comparación de convergencia — todos los métodos")

        fig_cmp = go.Figure()
        palette = ["#2563EB", "#D97706", "#059669"]
        for i, (m_name, (_, _, _, errors, _, _)) in enumerate(results.items()):
            label = m_name.split("(")[0].strip()
            fig_cmp.add_trace(go.Scatter(
                x=list(range(1, len(errors) + 1)),
                y=errors,
                mode="lines",
                name=label,
                line=dict(color=palette[i % 3], width=2.5),
            ))
        fig_cmp.add_hline(
            y=tol, line_dash="dash", line_color="#DC2626",
            annotation_text=f"tol = {tol:.0e}",
        )
        fig_cmp.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="||∇f(xₖ)|| (escala log)",
            yaxis_type="log",
            title="Comparación de métodos",
            template="plotly_white",
            height=430,
            legend=dict(orientation="h", yanchor="top", y=-0.2),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Tabla resumen
        st.subheader("📋 Tabla resumen comparativa")
        rows = []
        for m_name, (x_min, f_min, iters, errors, _, msg) in results.items():
            rows.append({
                "Método": m_name.split("(")[0].strip(),
                "Iteraciones": iters,
                "f(x*)": f"{f_min:.6e}",
                "||∇f|| final": f"{errors[-1]:.2e}",
                "Convergió": "✅" if "✓" in msg else "⚠️",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Trayectoria 2D (solo si n=2 y modo individual) ───────────────────────
    if n_vars == 2 and not compare_mode:
        m_name = list(results.keys())[0]
        x_min, _, _, _, traj, _ = results[m_name]
        traj_arr = np.array(traj)

        st.divider()
        st.subheader("🗺️ Trayectoria de optimización sobre curvas de nivel")

        margin = max(np.abs(traj_arr).max() * 0.6, 2.0)
        x1r = np.linspace(traj_arr[:, 0].min() - margin, traj_arr[:, 0].max() + margin, 200)
        x2r = np.linspace(traj_arr[:, 1].min() - margin, traj_arr[:, 1].max() + margin, 200)
        X1, X2 = np.meshgrid(x1r, x2r)
        try:
            Z = np.vectorize(lambda a, b: f_func([a, b]))(X1, X2)
        except Exception:
            Z = np.zeros_like(X1)

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Contour(
            x=x1r, y=x2r, z=Z,
            colorscale="Blues", opacity=0.85,
            contours=dict(showlabels=True, coloring="heatmap"),
            name="f(x)",
            showscale=True,
        ))
        # Trayectoria
        fig_traj.add_trace(go.Scatter(
            x=traj_arr[:, 0], y=traj_arr[:, 1],
            mode="lines+markers",
            name="Trayectoria",
            line=dict(color="white", width=2),
            marker=dict(size=5, color="white"),
        ))
        # Punto inicial
        fig_traj.add_trace(go.Scatter(
            x=[traj_arr[0, 0]], y=[traj_arr[0, 1]],
            mode="markers", name="Inicio x₀",
            marker=dict(size=14, color="#22C55E", symbol="circle", line=dict(color="white", width=2)),
        ))
        # Mínimo
        fig_traj.add_trace(go.Scatter(
            x=[x_min[0]], y=[x_min[1]],
            mode="markers", name="Mínimo x*",
            marker=dict(size=16, color="#EF4444", symbol="star", line=dict(color="white", width=2)),
        ))
        fig_traj.update_layout(
            xaxis_title="x₁", yaxis_title="x₂",
            title=f"Curvas de nivel y trayectoria — {m_name}",
            template="plotly_white",
            height=540,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # ── Info extra: Wolfe aplicado ────────────────────────────────────────────
    with st.expander("ℹ️ Información sobre las condiciones de Wolfe usadas"):
        st.markdown(f"""
| Parámetro | Valor usado |
|-----------|-------------|
| c₁ (Armijo) | `{c1:.2e}` |
| c₂ (Curvatura) | `{c2:.4f}` |
| α inicial | `{alpha_init:.4f}` |
| Algoritmo de búsqueda | Bracket + Zoom (Nocedal & Wright) |

Las condiciones de Wolfe garantizan que el paso α produzca:
1. **Disminución suficiente** del valor de la función (Armijo)
2. **Reducción del gradiente** proyectado en la dirección de búsqueda (Curvatura)
""")
