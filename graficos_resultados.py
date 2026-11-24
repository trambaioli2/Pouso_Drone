"""Gráficos dos resultados da apresentação (capítulo 3).

O notebook da apresentação só chama `plotar_cenario(res)`; o código das figuras fica aqui.
`res` é o dicionário devolvido por `resolver_cenario` no notebook.

Ordem das figuras de cada cenário:
1. trajetória em 3D, interativa (gira com o mouse; usa plotly);
2. vistas laterais X–Z e Y–Z com a pose do drone ao longo do voo;
3. velocidade, aceleração e jerk ao longo do tempo, um gráfico para cada.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    TEM_PLOTLY = True
except ImportError:
    TEM_PLOTLY = False

COR = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
COR_LIMITE = "#8a8a85"
COR_PLATAFORMA = "#bdbcb5"
COR_CONE = "#6b7f99"
COR_DRONE = "#2b2b2b"
COR_TRAJ = "#8a8a85"
EIXOS = ["x", "y", "z"]


def rotacao(phi, theta):
    # Orientação do drone com yaw = 0: R = Rx(φ) Ry(θ). Colunas: x_B, y_B, z_B.
    cf, sf, ct, st = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, cf, -sf], [0, sf, cf]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    return Rx @ Ry


def _virgula(valor):
    return f"{valor:g}".replace(".", ",")


# ---------------------------------------------------------------- 1. trajetória 3D interativa
def _faces_cone(plataforma, alfa, z_topo):
    (x0, x1), (y0, y1) = plataforma["x"], plataforma["y"]
    d = z_topo / alfa
    base = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    topo = base + d * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return base, topo


def plotar_trajetoria_3d(res, z_cone=2.5):
    P, mod = res["pos"], res["modelo"]
    plat = mod["plataforma"]
    titulo = f"{res['nome']}: trajetória (arraste com o mouse para girar)"
    base, topo = _faces_cone(plat, mod["alfa"], z_cone)

    if not TEM_PLOTLY:
        print("plotly não está instalado: mostrando a versão estática. "
              "Para girar com o mouse, instale com  .venv/bin/pip install plotly nbformat  e reinicie o kernel.")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        PX, PY = np.meshgrid(plat["x"], plat["y"])
        ax.plot_surface(PX, PY, np.zeros_like(PX), color=COR_PLATAFORMA, alpha=0.8)
        faces = [[(*base[i], 0), (*base[(i + 1) % 4], 0), (*topo[(i + 1) % 4], z_cone), (*topo[i], z_cone)]
                 for i in range(4)]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=COR_CONE, edgecolor=COR_CONE, alpha=0.08))
        ax.plot(*P, color=COR["x"], lw=2)
        ax.scatter(*P[:, 0], s=40, color=COR_DRONE)
        ax.scatter(*P[:, -1], s=60, marker="x", color=COR_DRONE)
        lo = np.minimum(P.min(axis=1), [plat["x"][0], plat["y"][0], 0]) - 0.5
        hi = np.maximum(P.max(axis=1), [plat["x"][1], plat["y"][1], 0]) + 0.5
        lo[2] = 0
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title(f"{res['nome']}: trajetória", fontsize=11)
        plt.show()
        return

    fig = go.Figure()
    # Plataforma (dois triângulos)
    (x0, x1), (y0, y1) = plat["x"], plat["y"]
    fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                            color=COR_PLATAFORMA, opacity=0.9, name="plataforma", hoverinfo="name"))
    # Faces do cone (cada face é um quadrilátero = dois triângulos)
    for f in range(4):
        g = (f + 1) % 4
        xs = [base[f, 0], base[g, 0], topo[g, 0], topo[f, 0]]
        ys = [base[f, 1], base[g, 1], topo[g, 1], topo[f, 1]]
        zs = [0, 0, z_cone, z_cone]
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=[0, 0], j=[1, 2], k=[2, 3], color=COR_CONE, opacity=0.12,
                                name="cone", hoverinfo="name", showlegend=(f == 0)))
    # Trajetória, início e pouso
    t = res["modelo"]["tempo"]
    fig.add_trace(go.Scatter3d(x=P[0], y=P[1], z=P[2], mode="lines", line=dict(color=COR["x"], width=6),
                               name="trajetória", customdata=t,
                               hovertemplate="t = %{customdata:.1f} s<br>x = %{x:.2f}<br>y = %{y:.2f}<br>z = %{z:.2f}"))
    fig.add_trace(go.Scatter3d(x=[P[0, 0]], y=[P[1, 0]], z=[P[2, 0]], mode="markers",
                               marker=dict(size=5, color=COR_DRONE), name="início"))
    fig.add_trace(go.Scatter3d(x=[P[0, -1]], y=[P[1, -1]], z=[P[2, -1]], mode="markers",
                               marker=dict(size=5, color=COR_DRONE, symbol="x"), name="pouso"))
    fig.update_layout(title=titulo, height=600, margin=dict(l=0, r=0, t=50, b=0),
                      scene=dict(xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]",
                                 aspectmode="data"),   # mesma escala nos três eixos
                      legend=dict(x=0.02, y=0.98))
    fig.show()


# ---------------------------------------------------------------- 2. vistas laterais com a pose do drone
def plotar_vistas_laterais(res, passo_desenho=0.5, meia_envergadura=0.35, comp_empuxo=0.7):
    P, fis, mod = res["pos"], res["fis"], res["modelo"]
    plat, alfa = mod["plataforma"], mod["alfa"]
    passo = max(1, round(passo_desenho / mod["DT"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, i, angulo in ((axes[0], 0, "pitch θ"), (axes[1], 1, "roll φ")):
        lo, hi = plat[EIXOS[i]]
        # Região proibida pelo cone (cada face envolve só um eixo horizontal e z)
        s_esq, s_dir = np.linspace(lo - 15, lo, 60), np.linspace(hi, hi + 15, 60)
        ax.fill_between(s_esq, 0, alfa * (lo - s_esq), color=COR_CONE, alpha=0.15, lw=0, label="proibido pelo cone")
        ax.fill_between(s_dir, 0, alfa * (s_dir - hi), color=COR_CONE, alpha=0.15, lw=0)
        ax.plot([lo, hi], [0, 0], color=COR_PLATAFORMA, lw=6, solid_capstyle="butt", label="plataforma")
        ax.plot(P[i], P[2], color=COR_TRAJ, lw=1.3, label="trajetória")
        for k in range(0, mod["N"] + 1, passo):
            R = rotacao(fis["phi"][k], fis["theta"][k])
            c = np.array([P[i, k], P[2, k]])
            braco = meia_envergadura * np.array([R[i, i], R[2, i]])      # x_B (vista X–Z) ou y_B (vista Y–Z)
            empuxo = comp_empuxo * fis["empuxo"][k] / res["peso"] * np.array([R[i, 2], R[2, 2]])
            ax.plot(*np.c_[c - braco, c + braco], color=COR_DRONE, lw=3, solid_capstyle="round")
            ax.annotate("", xy=c + empuxo, xytext=c, arrowprops=dict(arrowstyle="->", color=COR["z"], lw=1.6))
        ax.plot([], [], color=COR_DRONE, lw=3, label="drone (visto de lado)")
        ax.plot([], [], color=COR["z"], lw=1.6, label="empuxo")
        # Mesma escala nos dois eixos (a inclinação aparece com o ângulo real). Os intervalos são
        # ampliados até a proporção do painel, para os dois painéis terem o mesmo tamanho.
        margem, proporcao = 0.8, 1.25            # proporção largura/altura de cada painel
        x0, x1 = min(P[i].min(), lo) - margem, max(P[i].max(), hi) + margem
        z0, z1 = -0.3, P[2].max() + 1.0
        if (x1 - x0) < proporcao * (z1 - z0):
            meio, meia = (x0 + x1) / 2, proporcao * (z1 - z0) / 2
            x0, x1 = meio - meia, meio + meia
        else:
            z1 = z0 + (x1 - x0) / proporcao
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{EIXOS[i].upper()} [m]"); ax.set_ylabel("Z [m]")
        ax.set_title(f"Vista {EIXOS[i].upper()}–Z: aparece o {angulo}", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{res['nome']}: pose do drone a cada {_virgula(passo_desenho)} s "
                 "(ângulos reais; drone fora de escala)", fontsize=12)
    fig.tight_layout(); plt.show()


# ---------------------------------------------------------------- 3. velocidade, aceleração e jerk
def _grafico_componentes(nome, tempo, valores, rotulo, unidade, titulo, limite=None, limite_xy=None, degraus=False):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, e in enumerate(EIXOS):
        if degraus:
            ax.step(tempo, valores[i], where="mid", color=COR[e], label=f"{rotulo}{e}")
        else:
            ax.plot(tempo, valores[i], color=COR[e], label=f"{rotulo}{e}")
    if limite:
        for s in (1, -1):
            ax.axhline(s * limite, ls="--", lw=1, color=COR_LIMITE, label="limite" if s == 1 else None)
    if limite_xy:
        for s in (1, -1):
            ax.axhline(s * limite_xy, ls=":", lw=1.2, color=COR_LIMITE,
                       label="limite em x e y (giro)" if s == 1 else None)
    ax.set_xlabel("Tempo [s]"); ax.set_ylabel(f"{titulo} [{unidade}]")
    ax.set_title(f"{nome}: {titulo.lower()}", fontsize=11)
    ax.legend(ncol=5, fontsize=8, loc="best")
    fig.tight_layout(); plt.show()


def plotar_velocidade(res):
    _grafico_componentes(res["nome"], res["modelo"]["tempo"], res["vel"], "v", "m/s", "Velocidade",
                         limite=res["limites"]["v_max"])


def plotar_aceleracao(res):
    _grafico_componentes(res["nome"], res["modelo"]["tempo"], res["acel"], "a", "m/s²", "Aceleração",
                         limite=res["limites"]["a_max"])


def plotar_jerk(res):
    _grafico_componentes(res["nome"], res["modelo"]["tempo_jerk"], res["jerk"], "j", "m/s³", "Jerk",
                         limite_xy=res["limites"].get("jerk_max"), degraus=True)


def plotar_angulos(res):
    # Um gráfico para cada ângulo, com o limite e o quão perto o drone chegou dele.
    t, fis = res["modelo"]["tempo"], res["fis"]
    lim = np.degrees(res["limites"]["ang_max"])
    for chave, nome, eixo in (("theta", "Pitch θ", "x"), ("phi", "Roll φ", "y")):
        ang = np.degrees(fis[chave])
        k = int(np.argmax(np.abs(ang)))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.axhspan(lim, 1.25 * lim, color="#e34948", alpha=0.08, lw=0)
        ax.axhspan(-1.25 * lim, -lim, color="#e34948", alpha=0.08, lw=0, label="além do limite")
        for s in (1, -1):
            ax.axhline(s * lim, ls="--", lw=1.2, color="#e34948", label=f"limite ±{lim:.0f}°" if s == 1 else None)
        ax.plot(t, ang, color=COR[eixo], label=f"{nome.lower()} (move em {eixo})")
        ax.scatter(t[k], ang[k], s=45, color=COR[eixo], zorder=3)
        texto = f"máximo: {abs(ang[k]):.1f}° ({abs(ang[k]) / lim:.0%} do limite)".replace(".", ",")
        ax.annotate(texto, xy=(t[k], ang[k]), xytext=(12, -18 if ang[k] > 0 else 12), textcoords="offset points",
                    fontsize=9, color=COR_DRONE)
        ax.set_ylim(-1.25 * lim, 1.25 * lim)
        ax.set_xlabel("Tempo [s]"); ax.set_ylabel(f"{nome} [°]")
        ax.set_title(f"{res['nome']}: {nome.lower()} ao longo do pouso", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); plt.show()


def plotar_cenario(res, **opcoes_vistas):
    if res["status"] != "optimal":
        print(f"{res['nome']}: sem solução (status {res['status']}), nada para plotar.")
        return
    plotar_trajetoria_3d(res)
    plotar_vistas_laterais(res, **opcoes_vistas)
    plotar_velocidade(res)
    plotar_aceleracao(res)
    plotar_jerk(res)
