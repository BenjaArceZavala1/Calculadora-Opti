"""
OptiCalc — Calculadora de Mínimos Numéricos
Proyecto Final · Métodos de Optimización

Valor Agregado:
  ★ 1. Tabla de iteraciones con estimación del orden de convergencia (p)
  ★ 2. Análisis de múltiples puntos de partida (cuencas de atracción)
"""

import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OptiCalc — Mínimos Numéricos",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container{padding:1.5rem 2.5rem;max-width:100%;}
.stButton>button{font-size:15px!important;font-weight:700!important;
    padding:11px 28px!important;width:100%!important;letter-spacing:.5px;
    border-radius:8px!important;border:none!important;}
[data-testid="metric-container"]{border-radius:10px!important;}
[data-testid="stMetricValue"]{font-family:monospace!important;
    font-size:1.35rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{font-size:.7rem!important;
    text-transform:uppercase;letter-spacing:1px;}
.stTabs [data-baseweb="tab-list"]{border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:7px;}
hr{margin:18px 0!important;}
.hl-box{background:#0d2136;border:1px solid #1f6feb;
    border-left:4px solid #1f6feb;border-radius:6px;
    padding:11px 16px;font-family:monospace;font-size:13px;
    color:#79c0ff;margin:8px 0;}
.badge-ok{display:inline-block;background:#0f2916;color:#3fb950;
    border:1px solid #238636;border-radius:16px;
    padding:3px 14px;font-size:13px;font-weight:700;}
.badge-warn{display:inline-block;background:#2d1f00;color:#e3b341;
    border:1px solid #9e6a03;border-radius:16px;
    padding:3px 14px;font-size:13px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

PA = dict(template="plotly_dark", paper_bgcolor="#0d1117",
          plot_bgcolor="#161b22",
          font=dict(family="Consolas,monospace", color="#c9d1d9"))


# ─── SYMBOLIC FUNCTION BUILDER ────────────────────────────────────────────────
def build_functions(func_str, n):
    syms = sp.symbols(" ".join(f"x{i+1}" for i in range(n)))
    if n == 1:
        syms = (syms,)
    loc = {f"x{i+1}": syms[i] for i in range(n)}
    loc.update({"e": sp.E, "pi": sp.pi, "exp": sp.exp, "log": sp.log,
                "ln": sp.log, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "sqrt": sp.sqrt, "abs": sp.Abs})
    expr   = sp.sympify(func_str, locals=loc)
    gexprs = [sp.diff(expr, v) for v in syms]
    hexprs = [[sp.diff(gexprs[i], syms[j]) for j in range(n)] for i in range(n)]
    f_lam  = sp.lambdify(syms, expr, "numpy")
    g_lams = [sp.lambdify(syms, g, "numpy") for g in gexprs]
    h_lams = [[sp.lambdify(syms, hexprs[i][j], "numpy")
               for j in range(n)] for i in range(n)]

    def f(x):
        v = float(f_lam(*x))
        if not np.isfinite(v):
            raise ValueError("f(x) no es finita.")
        return v

    def gf(x):
        g = np.array([float(gl(*x)) for gl in g_lams])
        if not np.all(np.isfinite(g)):
            raise ValueError("gradiente contiene NaN/Inf.")
        return g

    def hf(x):
        return np.array([[float(h_lams[i][j](*x))
                          for j in range(n)] for i in range(n)])
    return f, gf, hf


# ─── WOLFE LINE SEARCH (bracket + zoom) ──────────────────────────────────────
def wolfe_ls(f, gf, x, d, c1, c2, amax):
    p0 = f(x); dp0 = float(np.dot(gf(x), d))
    if dp0 >= 0:
        return 1e-10
    phi  = lambda a: f(x + a * d)
    dphi = lambda a: float(np.dot(gf(x + a * d), d))

    def zoom(lo, hi, plo):
        for _ in range(40):
            aj = .5*(lo+hi); pj = phi(aj)
            if pj > p0 + c1*aj*dp0 or pj >= plo:
                hi = aj
            else:
                dj = dphi(aj)
                if abs(dj) <= -c2*dp0:
                    return aj
                if dj*(hi-lo) >= 0:
                    hi = lo
                lo, plo = aj, pj
            if abs(hi-lo) < 1e-15:
                break
        return .5*(lo+hi)

    ap, pp, a = 0.0, p0, min(amax, 1.0)
    for i in range(60):
        pa = phi(a)
        if not np.isfinite(pa) or pa > p0+c1*a*dp0 or (i > 0 and pa >= pp):
            return zoom(ap, a, pp)
        da = dphi(a)
        if abs(da) <= -c2*dp0:
            return a
        if da >= 0:
            return zoom(a, ap, pa)
        ap, pp, a = a, pa, min(2*a, amax)
    return max(a, 1e-12)


# ─── OPTIMIZATION METHODS ─────────────────────────────────────────────────────
def steepest_descent(f, gf, x0, c1, c2, amax, maxiter, tol):
    x = np.array(x0, dtype=float)
    errors, alphas, traj = [], [], [x.copy()]
    for k in range(maxiter):
        g = gf(x); ng = np.linalg.norm(g); errors.append(ng)
        if ng < tol:
            return x, f(x), k+1, errors, alphas, traj, "Convergencia: ‖∇f‖ < tolerancia"
        a = wolfe_ls(f, gf, x, -g, c1, c2, amax)
        alphas.append(a); x = x - a*g; traj.append(x.copy())
    return x, f(x), maxiter, errors, alphas, traj, "Máximo de iteraciones"


def conjugate_gradient(f, gf, x0, c1, c2, amax, maxiter, tol):
    x = np.array(x0, dtype=float); g = gf(x); d = -g.copy()
    errors, alphas, traj = [], [], [x.copy()]; n = len(x0)
    for k in range(maxiter):
        ng = np.linalg.norm(g); errors.append(ng)
        if ng < tol:
            return x, f(x), k+1, errors, alphas, traj, "Convergencia: ‖∇f‖ < tolerancia"
        a = wolfe_ls(f, gf, x, d, c1, c2, amax)
        x = x + a*d; gn = gf(x)
        beta = np.dot(gn, gn) / max(np.dot(g, g), 1e-30)
        if (k+1) % n == 0:
            beta = 0.0
        dn = -gn + beta*d
        if np.dot(dn, gn) >= 0:
            dn = -gn
        g = gn; d = dn; alphas.append(a); traj.append(x.copy())
    return x, f(x), maxiter, errors, alphas, traj, "Máximo de iteraciones"


def newton_method(f, gf, hf, x0, c1, c2, amax, maxiter, tol):
    x = np.array(x0, dtype=float); n = len(x0)
    errors, alphas, traj = [], [], [x.copy()]
    for k in range(maxiter):
        g = gf(x); ng = np.linalg.norm(g); errors.append(ng)
        if ng < tol:
            return x, f(x), k+1, errors, alphas, traj, "Convergencia: ‖∇f‖ < tolerancia"
        H = hf(x)
        eigs = np.linalg.eigvalsh(H)
        if eigs.min() <= 1e-10:
            H = H + (abs(eigs.min()) + 1e-6) * np.eye(n)
        try:
            d = np.linalg.solve(H, -g)
        except Exception:
            d = -g
        if np.dot(d, g) >= 0:
            d = -g
        a = wolfe_ls(f, gf, x, d, c1, c2, amax)
        alphas.append(a); x = x + a*d; traj.append(x.copy())
    return x, f(x), maxiter, errors, alphas, traj, "Máximo de iteraciones"


def dispatch(mth, f, gf, hf, x0, c1, c2, amax, maxiter, tol):
    if mth == "Gradiente":
        return steepest_descent(f, gf, x0, c1, c2, amax, maxiter, tol)
    elif mth == "Gradiente Conjugado":
        return conjugate_gradient(f, gf, x0, c1, c2, amax, maxiter, tol)
    else:
        return newton_method(f, gf, hf, x0, c1, c2, amax, maxiter, tol)


# ─── CONVERGENCE ORDER ────────────────────────────────────────────────────────
def conv_order(errors):
    ps = []
    for k in range(2, len(errors)):
        e0, e1, e2 = errors[k-2], errors[k-1], errors[k]
        if e0 > 1e-15 and e1 > 1e-15 and e2 > 1e-15 and e1 < e0 and e2 < e1:
            try:
                p = np.log(e2/e1) / np.log(e1/e0)
                if 0.3 <= p <= 4.0:
                    ps.append(p)
            except Exception:
                pass
    return ps


# ══════════════════════════════════════════════════════════════════════════════
# UI — HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #21262d;padding-bottom:18px;margin-bottom:22px;
            display:flex;align-items:center;gap:14px;">
  <span style="font-size:38px;">⚡</span>
  <div>
    <span style="font-size:2rem;font-weight:800;color:#c9d1d9;
                 font-family:monospace;letter-spacing:-1.5px;">OptiCalc</span>
    <p style="margin:0;color:#8b949e;font-size:13px;font-family:monospace;">
      Proyecto Final · Métodos de Optimización ·
      Gradiente · Grad. Conjugado · Newton · Wolfe
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Quick examples ───────────────────────────────────────────────────────────
if "fn" not in st.session_state:
    st.session_state.fn  = "(x1 - 2)**2 + (x2 - 3)**2"
    st.session_state.nv  = 2
    st.session_state.x0d = "0, 0"

EXAMPLES = [
    ("Paraboloide 2D",  "(x1-2)**2 + (x2-3)**2",                    2, "0, 0"),
    ("Rosenbrock",       "100*(x2-x1**2)**2 + (1-x1)**2",            2, "-1, 1"),
    ("Himmelblau",       "(x1**2+x2-11)**2 + (x1+x2**2-7)**2",      2, "0, 0"),
    ("Cuadrática 3D",   "x1**2+2*x2**2+x3**2+x1*x2+x2*x3",         3, "1, 1, 1"),
]
st.markdown("**⚡ Ejemplos rápidos:**")
for col, (lbl, expr, nv, x0d) in zip(st.columns(4), EXAMPLES):
    if col.button(lbl, use_container_width=True, key=f"ex_{lbl}"):
        st.session_state.fn  = expr
        st.session_state.nv  = nv
        st.session_state.x0d = x0d

st.divider()

# ─── Input grid ───────────────────────────────────────────────────────────────
col_L, col_R = st.columns(2, gap="large")

with col_L:
    st.markdown("### 📐 Función")
    n_vars = int(st.number_input(
        "Número de variables (n)", 1, 20,
        value=st.session_state.get("nv", 2), step=1))
    func_input = st.text_area(
        "Función objetivo f(x₁, x₂, ...)",
        value=st.session_state.get("fn", "(x1-2)**2 + (x2-3)**2"),
        height=85,
        help="Potencias con **. Funciones: exp, log, sin, cos, sqrt. "
             "Ej: (x1-2)**2 + (x2-3)**2")
    x0_input = st.text_input(
        "Punto de partida x₀ (separado por comas)",
        value=st.session_state.get("x0d", ", ".join(["0"] * n_vars)))
    method_full = st.selectbox("Método de optimización", [
        "Gradiente (Steepest Descent)",
        "Gradiente Conjugado (Fletcher-Reeves)",
        "Método de Newton"])
    METHOD_MAP = {
        "Gradiente (Steepest Descent)":          "Gradiente",
        "Gradiente Conjugado (Fletcher-Reeves)": "Gradiente Conjugado",
        "Método de Newton":                      "Newton",
    }
    method = METHOD_MAP[method_full]

with col_R:
    st.markdown("### ⚙️ Parámetros")
    ca, cb = st.columns(2)
    with ca:
        max_iter = int(st.number_input(
            "Iteraciones máximas", 10, 100_000, 1000, 100))
    with cb:
        tol_exp = int(st.number_input(
            "Tolerancia (10ⁿ)", -15, -1, -6, 1))
    tol = 10.0 ** tol_exp
    st.caption(f"Tolerancia activa: `{tol:.0e}`")

    st.markdown("### 🔍 Condiciones de Wolfe")
    cc, cd = st.columns(2)
    with cc:
        c1 = float(st.number_input(
            "c₁ — Armijo", 1e-10, 0.49, 1e-4, format="%.2e",
            help="Disminución suficiente. Típico: 1e-4"))
    with cd:
        c2 = float(st.number_input(
            "c₂ — Curvatura", 0.01, 0.9999, 0.9, format="%.4f",
            help="Condición curvatura. GD/Newton: 0.9  |  GC: 0.1"))
    alpha_max = float(st.number_input(
        "α máximo inicial", 1e-4, 100.0, 1.0, format="%.4f"))
    st.markdown("""
    <div class="hl-box">
    0 &lt; c₁ &lt; c₂ &lt; 1 &nbsp;(condición necesaria)<br>
    Wolfe 1: f(x+αd) ≤ f(x) + c₁α ∇f·d<br>
    Wolfe 2: |∇f(x+αd)·d| ≤ c₂|∇f(x)·d|
    </div>""", unsafe_allow_html=True)

st.divider()
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("⚡  CALCULAR MÍNIMO", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    # --- validate ---
    try:
        x0 = [float(v.strip()) for v in x0_input.split(",")]
    except Exception:
        st.error("⚠️ El punto de partida debe contener números separados por comas.")
        st.stop()
    if len(x0) != n_vars:
        st.error(f"⚠️ El punto de partida necesita {n_vars} valores (tienes {len(x0)}).")
        st.stop()
    if not (0 < c1 < c2 < 1):
        st.error("⚠️ Debe cumplirse 0 < c₁ < c₂ < 1.")
        st.stop()

    # --- build functions ---
    with st.spinner("🔣 Calculando gradiente y Hessiana simbólicamente…"):
        try:
            f_func, grad_func, hess_func = build_functions(func_input, n_vars)
            f_func(x0); grad_func(x0)
        except Exception as e:
            st.error(f"❌ Error en la función: {e}"); st.stop()

    # --- run optimizer ---
    with st.spinner(f"⚙️ Ejecutando {method}…"):
        try:
            x_min, f_min, iters, errors, alphas, traj, msg = dispatch(
                method, f_func, grad_func, hess_func,
                x0, c1, c2, alpha_max, max_iter, tol)
        except Exception as e:
            st.error(f"❌ Error durante la optimización: {e}"); st.stop()

    converged = "Convergencia" in msg
    st.divider()

    badge = ('<span class="badge-ok">✓  CONVERGENCIA ALCANZADA</span>'
             if converged else
             '<span class="badge-warn">⚠  MÁXIMO DE ITERACIONES</span>')
    st.markdown(badge + "<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iteraciones realizadas", iters)
    m2.metric("f(x*)", f"{f_min:.6e}")
    m3.metric("‖∇f‖ final", f"{errors[-1]:.2e}")
    m4.metric("Tolerancia objetivo", f"{tol:.0e}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Resumen",
        "📈 Convergencia y trayectoria",
        "🔬 Tabla de iteraciones  ★",
        "🌐 Superficie 3D  ★",
    ])

    # ── Tab 1 · Resumen ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Punto mínimo x*")
        df_pt = pd.DataFrame({
            "Variable":     [f"x{i+1}*" for i in range(len(x_min))],
            "Valor óptimo": [f"{v:.10f}" for v in x_min],
        })
        st.dataframe(df_pt, hide_index=True, use_container_width=True)
        st.markdown(f"""
        <div class="hl-box">
        f(x*) = {f_min:.10f}<br>
        ‖∇f(x*)‖ = {errors[-1]:.4e}<br>
        Método: {method_full}<br>
        Iteraciones: {iters} / {max_iter}<br>
        Estado: {msg}
        </div>""", unsafe_allow_html=True)
        with st.expander("ℹ️ Parámetros de Wolfe utilizados"):
            st.markdown(f"""
| Parámetro | Valor |
|-----------|-------|
| c₁ (Armijo) | `{c1:.2e}` |
| c₂ (Curvatura) | `{c2:.4f}` |
| α máximo inicial | `{alpha_max:.4f}` |
| Algoritmo búsqueda de línea | Bracket + Zoom (Nocedal & Wright, 2006) |
""")

    # ── Tab 2 · Convergencia y trayectoria ────────────────────────────────────
    with tab2:
        pcols = st.columns([3, 2] if n_vars == 2 else [1])
        with pcols[0]:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=list(range(1, len(errors)+1)), y=errors,
                mode="lines", name="‖∇f(xₖ)‖",
                line=dict(color="#00ffc8", width=2.5)))
            fig_cv.add_hline(
                y=tol, line_dash="dash", line_color="#f85149",
                annotation_text=f"tol={tol:.0e}",
                annotation_position="bottom right")
            fig_cv.update_layout(
                title=f"Convergencia — {method}",
                xaxis_title="Iteración k",
                yaxis_title="‖∇f(xₖ)‖",
                yaxis_type="log", height=420,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                **PA)
            st.plotly_chart(fig_cv, use_container_width=True)

        if n_vars == 2:
            with pcols[1]:
                traj_arr = np.array(traj)
                mg = max(abs(traj_arr).max() * 0.6, 2.0)
                x1r = np.linspace(traj_arr[:,0].min()-mg, traj_arr[:,0].max()+mg, 150)
                x2r = np.linspace(traj_arr[:,1].min()-mg, traj_arr[:,1].max()+mg, 150)
                X1, X2 = np.meshgrid(x1r, x2r)
                try:
                    Z = np.vectorize(lambda a, b: f_func([a, b]))(X1, X2)
                except Exception:
                    Z = np.zeros_like(X1)
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Contour(
                    x=x1r, y=x2r, z=Z, colorscale="Blues",
                    contours=dict(showlabels=True), opacity=0.75, showscale=False))
                fig_tr.add_trace(go.Scatter(
                    x=traj_arr[:,0], y=traj_arr[:,1],
                    mode="lines+markers", name="Trayectoria",
                    line=dict(color="#f0883e", width=2),
                    marker=dict(size=4, color="#f0883e")))
                fig_tr.add_trace(go.Scatter(
                    x=[traj_arr[0,0]], y=[traj_arr[0,1]],
                    mode="markers", name="x₀",
                    marker=dict(size=13, color="#3fb950", symbol="circle")))
                fig_tr.add_trace(go.Scatter(
                    x=[x_min[0]], y=[x_min[1]],
                    mode="markers", name="x*",
                    marker=dict(size=15, color="#f85149", symbol="star")))
                fig_tr.update_layout(
                    title="Trayectoria (curvas de nivel)",
                    xaxis_title="x₁", yaxis_title="x₂", height=420,
                    legend=dict(orientation="h", y=-0.25),
                    xaxis=dict(gridcolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d"),
                    **PA)
                st.plotly_chart(fig_tr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ★ Tab 3 · VALOR AGREGADO 1 — Tabla de iteraciones + orden convergencia
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🔬 Tabla completa de iteraciones + orden de convergencia")
        st.markdown(r"""
Registro **iteración a iteración** de todos los valores internos.
La columna **Orden p** estima empíricamente el orden de convergencia:

$$p_k = \frac{\log(\|e_{k+1}\| / \|e_k\|)}{\log(\|e_k\| / \|e_{k-1}\|)}, \qquad e_k = \|\nabla f(x_k)\|$$

- p ≈ 1 → convergencia **lineal** (Gradiente)
- 1 < p < 2 → convergencia **superlineal** (Grad. Conjugado)
- p ≈ 2 → convergencia **cuadrática** (Newton cerca del mínimo)
""")
        n_it = len(errors)
        alphas_p = alphas + [float("nan")] * (n_it - len(alphas))

        ratios_col = ["-", "-"]
        for k in range(2, n_it):
            e0, e1, e2 = errors[k-2], errors[k-1], errors[k]
            if e0 > 1e-15 and e1 > 1e-15 and e2 > 1e-15 and e1 < e0 and e2 < e1:
                try:
                    p = np.log(e2/e1) / np.log(e1/e0)
                    ratios_col.append(f"{p:.4f}" if 0.3 <= p <= 5 else "-")
                except Exception:
                    ratios_col.append("-")
            else:
                ratios_col.append("-")

        fxk_col = []
        for k in range(n_it):
            try:
                fxk_col.append(f"{f_func(traj[k]):.6e}")
            except Exception:
                fxk_col.append("-")

        ratio_s = ["-"] + [
            f"{errors[k]/errors[k-1]:.5f}" if errors[k-1] > 0 else "-"
            for k in range(1, n_it)]

        df_it = pd.DataFrame({
            "k":                range(1, n_it+1),
            "‖∇f(xₖ)‖":        [f"{e:.6e}" for e in errors],
            "f(xₖ)":            fxk_col,
            "αₖ (Wolfe)":       [f"{a:.6f}" if np.isfinite(a) else "-"
                                  for a in alphas_p],
            "‖∇f‖ₖ/‖∇f‖ₖ₋₁":   ratio_s,
            "Orden p estimado":  ratios_col,
        })
        st.dataframe(df_it, hide_index=True, use_container_width=True)

        # --- convergence order summary ---
        ps = conv_order(errors)
        if len(ps) >= 3:
            p_tail = float(np.mean(ps[-min(10, len(ps)):]))
            if p_tail < 1.25:
                lbl, clr = "📏 Lineal (p ≈ 1)", "#e3b341"
            elif p_tail < 1.75:
                lbl, clr = "🔺 Superlineal (1 < p < 2)", "#79c0ff"
            else:
                lbl, clr = "🚀 Cuadrática (p ≈ 2)", "#3fb950"
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #21262d;
            border-left:4px solid {clr};border-radius:6px;
            padding:12px 16px;margin:10px 0;">
  <b style="color:{clr};">Orden de convergencia estimado: p ≈ {p_tail:.3f}</b>
  &nbsp;→ {lbl}
  <br><small style="color:#8b949e;">
  Promedio de los últimos {min(10, len(ps))} valores calculados
  en la cola de la convergencia.</small>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No hay suficientes iteraciones para estimar el orden de convergencia.")

        # --- alpha chart ---
        if len(alphas) > 2:
            st.markdown("**Evolución del paso αₖ por iteración:**")
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(
                x=list(range(1, len(alphas)+1)), y=alphas,
                mode="lines+markers", name="αₖ",
                line=dict(color="#e3b341", width=2),
                marker=dict(size=5)))
            fig_a.update_layout(
                xaxis_title="Iteración k", yaxis_title="αₖ", height=250,
                margin=dict(t=10, b=40),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                **PA)
            st.plotly_chart(fig_a, use_container_width=True)

        st.download_button(
            "📥 Descargar tabla de iteraciones (CSV)",
            data=df_it.to_csv(index=False),
            file_name=f"iteraciones_{method.lower().replace(' ','_')}.csv",
            mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # ★ Tab 4 · VALOR AGREGADO 2 — Superficie 3D con trayectoria
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🌐 Superficie 3D con trayectoria de optimización")
        st.markdown("""
Visualización **tridimensional** de la función f(x₁, x₂) con la trayectoria completa del algoritmo
proyectada sobre la superficie. Muestra intuitivamente cómo el método desciende por la topografía
de la función hasta encontrar el mínimo. Puedes **rotar, hacer zoom y explorar** la superficie con el mouse.
""")
        if n_vars == 2:
            traj_arr = np.array(traj)
            mg = max(float(np.abs(traj_arr).max()) * 0.7, 2.5)
            x1r = np.linspace(traj_arr[:,0].min()-mg, traj_arr[:,0].max()+mg, 80)
            x2r = np.linspace(traj_arr[:,1].min()-mg, traj_arr[:,1].max()+mg, 80)
            X1, X2 = np.meshgrid(x1r, x2r)
            try:
                Z = np.vectorize(lambda a, b: f_func([a, b]))(X1, X2)
                z_med = float(np.nanmedian(Z))
                z_std = float(np.nanstd(Z))
                Z = np.clip(Z, z_med - 6*z_std, z_med + 6*z_std)
            except Exception:
                Z = np.zeros_like(X1)

            traj_z = []
            for pt in traj:
                try:
                    traj_z.append(f_func(pt))
                except Exception:
                    traj_z.append(float("nan"))

            fig_3d = go.Figure()

            fig_3d.add_trace(go.Surface(
                x=x1r, y=x2r, z=Z,
                colorscale="Blues",
                opacity=0.80,
                showscale=True,
                colorbar=dict(title="f(x)", thickness=14, len=0.65,
                              tickfont=dict(color="#c9d1d9")),
                name="f(x₁, x₂)"))

            fig_3d.add_trace(go.Scatter3d(
                x=traj_arr[:,0], y=traj_arr[:,1], z=traj_z,
                mode="lines+markers",
                name="Trayectoria",
                line=dict(color="#f0883e", width=5),
                marker=dict(size=3, color="#f0883e", opacity=0.8)))

            fig_3d.add_trace(go.Scatter3d(
                x=[traj_arr[0,0]], y=[traj_arr[0,1]], z=[traj_z[0]],
                mode="markers", name="Inicio x₀",
                marker=dict(size=10, color="#3fb950", symbol="circle",
                            line=dict(color="white", width=1))))

            fig_3d.add_trace(go.Scatter3d(
                x=[x_min[0]], y=[x_min[1]], z=[f_min],
                mode="markers", name="Mínimo x*",
                marker=dict(size=12, color="#f85149", symbol="diamond",
                            line=dict(color="white", width=1))))

            fig_3d.update_layout(
                title=dict(
                    text=f"Superficie f(x₁,x₂) — {method} | {iters} iteraciones",
                    font=dict(color="#c9d1d9", size=14)),
                scene=dict(
                    xaxis=dict(title="x₁", gridcolor="#30363d",
                               backgroundcolor="#161b22", color="#8b949e"),
                    yaxis=dict(title="x₂", gridcolor="#30363d",
                               backgroundcolor="#161b22", color="#8b949e"),
                    zaxis=dict(title="f(x₁,x₂)", gridcolor="#30363d",
                               backgroundcolor="#161b22", color="#8b949e"),
                    bgcolor="#0d1117",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1))),
                legend=dict(font=dict(color="#c9d1d9"),
                            bgcolor="rgba(13,17,23,0.7)",
                            bordercolor="#30363d", borderwidth=1),
                height=600,
                **PA)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption(
                "💡 Arrastra para rotar · Scroll para zoom · "
                "Doble click para resetear la vista. "
                "Punto verde = inicio x₀ · Punto rojo = mínimo x*")

        elif n_vars == 1:
            x1r = np.linspace(float(x0[0]) - 5, float(x0[0]) + 5, 400)
            try:
                y1r = np.array([f_func([xi]) for xi in x1r])
            except Exception:
                y1r = np.zeros_like(x1r)
            traj_x = [float(t[0]) for t in traj]
            traj_y_vals = []
            for t in traj:
                try:
                    traj_y_vals.append(f_func(t))
                except Exception:
                    traj_y_vals.append(float("nan"))
            fig_1d = go.Figure()
            fig_1d.add_trace(go.Scatter(
                x=x1r, y=y1r, mode="lines", name="f(x₁)",
                line=dict(color="#00ffc8", width=2.5)))
            fig_1d.add_trace(go.Scatter(
                x=traj_x, y=traj_y_vals, mode="markers", name="Iteraciones",
                marker=dict(size=7, color="#f0883e", opacity=0.8)))
            fig_1d.add_trace(go.Scatter(
                x=[x_min[0]], y=[f_min], mode="markers", name="x*",
                marker=dict(size=14, color="#f85149", symbol="star")))
            fig_1d.update_layout(
                xaxis_title="x₁", yaxis_title="f(x₁)", height=420,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                **PA)
            st.plotly_chart(fig_1d, use_container_width=True)

        else:
            st.info(
                f"La visualización 3D está disponible para funciones de "
                f"**2 variables**. Tu función tiene {n_vars} variables. "
                f"Consulta la pestaña de Convergencia y la Tabla de iteraciones.")

