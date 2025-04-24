import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import io

# ─── Fonte e config inicial ─────────────────────────────────────────────────────
mpl.rcParams["font.family"] = "Montserrat"
mpl.rcParams["font.weight"] = "semibold"
st.set_page_config(layout="wide")
st.title("📊 Radar Individual por Rota - CBLOL")

# ─── Upload + editor inline ──────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Faça upload do CSV (Jogador, Time, Lane, Games e métricas)",
    type="csv"
)
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📝 Edite os valores (opcional)")
df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ─── Limpeza & conversão ─────────────────────────────────────────────────────────
df["Games"] = pd.to_numeric(df["Games"], errors="coerce").fillna(0).astype(int)
stat_cols  = [c for c in df.columns if c not in ("Jogador","Time","Lane","Games")]
for c in stat_cols:
    df[c] = df[c].astype(str).str.strip().str.rstrip("%")
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ─── Seleção de rota ────────────────────────────────────────────────────────────
lanes = df["Lane"].dropna().unique().tolist()
selected_lane = st.sidebar.selectbox("🎯 Selecione a rota", lanes)
df_lane = df[df["Lane"] == selected_lane].copy()
if df_lane.empty:
    st.error(f"Nenhum dado para rota '{selected_lane}'")
    st.stop()

# ─── Inverte 'Avg deaths' se existir ─────────────────────────────────────────────
if "Avg deaths" in stat_cols:
    df_lane["Avg deaths"] = df_lane["Avg deaths"].max() - df_lane["Avg deaths"]

# ─── Normalização 0–1 por métrica na rota ────────────────────────────────────────
mins = df_lane[stat_cols].min()
maxs = df_lane[stat_cols].max()
data_norm = (df_lane[stat_cols] - mins) / (maxs - mins).replace(0, 1)
data_norm = data_norm.fillna(0.5)
data_norm.index = df_lane["Jogador"]

# ─── Média da rota (tracejado) ──────────────────────────────────────────────────
avg_norm   = data_norm.mean(axis=0).tolist()
avg_values = avg_norm + [avg_norm[0]]

# ─── Controles visuais ─────────────────────────────────────────────────────────
pad_val         = st.sidebar.slider("Pad das labels",                 0, 30, 12)
title_offset    = st.sidebar.slider("Offset vertical do título (y)", 0.9, 1.3, 1.08)
font_size       = st.sidebar.slider("Tamanho fonte labels",           8, 20, 10)
title_size      = st.sidebar.slider("Tamanho fonte título",          10, 30, 16)
player_color    = st.sidebar.color_picker("Cor dos players",          "#00de55")
avg_color       = st.sidebar.color_picker("Cor da média (tracejado)",  "#FF0000")
alpha_val       = st.sidebar.slider("Transparência do preenchimento",0.0, 1.0, 0.4)
font_color      = st.sidebar.color_picker("Cor da fonte",              "#FFFFFF")
legend_pad      = st.sidebar.slider("Padding interno da legenda",   0.0, 2.0, 0.4)
legend_x_offset = st.sidebar.slider("Offset X da legenda",          1.05, 2.0, 1.2)
legend_y_offset = st.sidebar.slider("Offset Y da legenda",          0.5, 1.5, 1.1)

# ─── Prepara ângulos e métricas ─────────────────────────────────────────────────
metrics = stat_cols.copy()
angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

# ─── Gera todos os gráficos e armazena (player, fig, img_bytes) ───────────────
gallery = []
for player, row in data_norm.iterrows():
    values = row.tolist() + [row.tolist()[0]]
    num_games = int(df_lane.loc[df_lane["Jogador"] == player, "Games"].iloc[0])

    fig, ax = plt.subplots(
        figsize=(5, 5),
        subplot_kw=dict(polar=True),
        facecolor="none"
    )
    fig.patch.set_alpha(0)

    # deixa a borda externa branca
    ax.spines['polar'].set_edgecolor(font_color)
    ax.spines['polar'].set_linewidth(1.5)

    # plot tracejado da média
    ax.plot(
        angles, avg_values,
        linestyle="--",
        color=avg_color,
        linewidth=1.5,
        label="Média rota"
    )

    # plot do jogador
    ax.plot(
        angles, values,
        color=player_color,
        linewidth=2,
        label=player
    )
    ax.fill(
        angles, values,
        color=player_color,
        alpha=alpha_val
    )

    # labels angulares
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        metrics,
        fontsize=font_size,
        fontweight="semibold",
        color=font_color
    )
    ax.tick_params(axis="x", pad=pad_val, colors=font_color)
    ax.tick_params(axis="y", colors=font_color)

    # título
    ax.set_title(
        f"{player} – {num_games} jogos",
        fontsize=title_size,
        fontweight="semibold",
        color=font_color,
        y=title_offset
    )

    ax.set_ylim(0, 1)

    # legenda fora do plot, com offset ajustável
    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(legend_x_offset, legend_y_offset),
        frameon=False,
        borderpad=legend_pad
    )
    for txt in leg.get_texts():
        txt.set_color(font_color)

    # converte figura em PNG bytes para a miniatura
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    img_bytes = buf.getvalue()

    gallery.append((player, fig, img_bytes))

# ─── Exibe galeria de miniaturas + expander com full-size ─────────────────────
st.header(f"Jogadores em {selected_lane} ({len(gallery)} no total)")
cols = st.columns(3)
for idx, (player, fig, img) in enumerate(gallery):
    col = cols[idx % 3]
    col.image(img, width=200)  # miniatura
    with col.expander(f"{player} (clique para ampliar)"):
        col.pyplot(fig)         # full resolution
        col.download_button(
            f"📥 Download {player}",
            img,
            file_name=f"{player}_{selected_lane}.png",
            mime="image/png"
        )
