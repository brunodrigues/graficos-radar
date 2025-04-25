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
st.title("📊 Radar Dinâmico + Tabela por Rota (CBLOL)")

# ─── Session state ───────────────────────────────────────────────────────────────
for key in ("plot_ready","selected_metrics","selected_pos"):
    if key not in st.session_state:
        st.session_state[key] = False if key=="plot_ready" else None

# ─── Upload + preview ───────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Faça upload do CSV (Player, Position, Games e métricas)", type="csv"
)
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)
st.sidebar.markdown("### Preview dos dados")
st.sidebar.dataframe(df.head(), use_container_width=True)

# ─── Detecta colunas-chave ───────────────────────────────────────────────────────
player_col   = next((c for c in df.columns if c.lower()=="player"), df.columns[0])
position_col = next((c for c in df.columns if c.lower() in ("position","lane")), None)
games_col    = next((c for c in df.columns if c.lower()=="games"), None)
if position_col is None or games_col is None:
    st.error("O CSV precisa ter colunas 'Player', 'Position' (ou 'Lane') e 'Games'.")
    st.stop()

# ─── Escolha de métricas ─────────────────────────────────────────────────────────
base_cols        = {player_col, position_col, games_col}
all_metrics      = [c for c in df.columns if c not in base_cols]
if st.session_state.selected_metrics is None:
    st.session_state.selected_metrics = all_metrics
prev_metrics     = st.session_state.selected_metrics

selected_metrics = st.sidebar.multiselect(
    "🎯 Quais métricas incluir?",
    options=all_metrics,
    default=prev_metrics
)
if selected_metrics != prev_metrics:
    st.session_state.plot_ready = False
st.session_state.selected_metrics = selected_metrics

if len(selected_metrics) < 3:
    st.sidebar.error("Selecione ao menos 3 métricas.")
    st.stop()

# ─── Editor inline ───────────────────────────────────────────────────────────────
st.subheader("📝 Edite os valores (opcional)")
df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ─── Conversão numérica ─────────────────────────────────────────────────────────
df[games_col] = pd.to_numeric(df[games_col], errors="coerce").fillna(0).astype(int)
for m in selected_metrics:
    df[m] = pd.to_numeric(df[m].astype(str).str.rstrip("%"), errors="coerce")

# ─── Seleção de rota ────────────────────────────────────────────────────────────
prev_pos     = st.session_state.selected_pos
positions    = df[position_col].dropna().unique().tolist()
selected_pos = st.sidebar.selectbox(f"🚩 Selecione a rota ({position_col})", positions)
if selected_pos != prev_pos:
    st.session_state.plot_ready = False
st.session_state.selected_pos = selected_pos

df_lane = df[df[position_col]==selected_pos].copy()
if df_lane.empty:
    st.error(f"Nenhum dado para {position_col} = '{selected_pos}'")
    st.stop()

# ─── Inverte 'Avg deaths' apenas para o cálculo do radar ───────────────────────
norm_df = df_lane[selected_metrics].copy()
if "Avg deaths" in selected_metrics:
    norm_df["Avg deaths"] = norm_df["Avg deaths"].max() - norm_df["Avg deaths"]

# ─── Controles visuais ──────────────────────────────────────────────────────────
st.sidebar.markdown("### Estilo do Radar e Tabela")
pad_val      = st.sidebar.slider("Pad labels",           0,30,12)
title_offset = st.sidebar.slider("Y-offset título",     0.9,1.3,1.08)
label_size   = st.sidebar.slider("Tamanho labels",       8,20,12)
title_size   = st.sidebar.slider("Tamanho título",      10,30,16)
player_color = st.sidebar.color_picker("Cor do player",    "#00de55")
avg_color    = st.sidebar.color_picker("Cor da média",     "#FF0000")
alpha_val    = st.sidebar.slider("Transparência área",   0.0,1.0,0.4)
font_color   = st.sidebar.color_picker("Cor da fonte",     "#FFFFFF")
line_color   = st.sidebar.color_picker("Cor das linhas",   "#FFFFFF")

# ─── Espaçamento entre radar e tabela ────────────────────────────────────────────
wspace = st.sidebar.slider("Espaçamento gráfico/tabela", 0.0,1.0,0.3)

# ─── Larguras das colunas da tabela ─────────────────────────────────────────────
metric_w = st.sidebar.number_input("Largura coluna 'Métrica' (px)", min_value=40, max_value=300, value=100)
value_w  = st.sidebar.number_input("Largura coluna 'Valor'   (px)", min_value=40, max_value=300, value=80)
rank_w   = st.sidebar.number_input("Largura coluna 'Rank'    (px)", min_value=40, max_value=300, value=80)

# ─── Botão para disparar plotagem ───────────────────────────────────────────────
if st.sidebar.button("▶️ Criar gráficos"):
    st.session_state.plot_ready = True
if not st.session_state.plot_ready:
    st.info("📋 Ajuste métricas/rota e clique em ▶️ Criar gráficos")
    st.stop()

# ─── Normalização 0–1 por métrica ───────────────────────────────────────────────
mins      = norm_df.min()
maxs      = norm_df.max()
data_norm = (norm_df - mins) / (maxs - mins).replace(0,1)
data_norm = data_norm.fillna(0.5)
data_norm.index = df_lane[player_col]

avg_norm   = data_norm.mean(axis=0).tolist()
avg_values = avg_norm + [avg_norm[0]]

angles = np.linspace(0, 2*np.pi, len(selected_metrics), endpoint=False).tolist()
angles += angles[:1]

# ─── Função que monta os dados da tabela (com valores **originais**) ──────────
def build_table_data(player):
    row = df_lane[df_lane[player_col]==player].iloc[0]
    table_data = []
    for m in selected_metrics:
        v    = row[m]
        disp = "-" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")
        asc  = (m!="Avg deaths")
        if pd.isna(v):
            rk_disp = "-"
        else:
            rk = int(df_lane[m].rank(ascending=asc, method="min").loc[row.name])
            rk_disp = f"{rk}°"
        table_data.append([m, disp, rk_disp])
    return table_data

# ─── Exibe galeria ──────────────────────────────────────────────────────────────
st.header(f"Jogadores em {selected_pos} — {len(data_norm)} no total")
cols = st.columns(3)

for idx, player in enumerate(data_norm.index):
    vals  = data_norm.loc[player].tolist() + [data_norm.loc[player].tolist()[0]]
    games = int(df_lane[df_lane[player_col]==player][games_col].iat[0])

    # monta figura única com radar + tabela
    fig = plt.figure(figsize=(8,4), facecolor="none")
    gs  = fig.add_gridspec(1,2, width_ratios=[3,2], wspace=wspace)

    # ─── Radar ───────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0], polar=True)
    ax0.patch.set_alpha(0)
    ax0.spines['polar'].set_edgecolor(line_color)
    ax0.spines['polar'].set_linewidth(1.5)
    ax0.grid(color=line_color)
    ax0.plot(angles, avg_values, '--', color=avg_color, linewidth=1.5)
    ax0.plot(angles, vals,       color=player_color, linewidth=2)
    ax0.fill(angles, vals,       color=player_color, alpha=alpha_val)
    ax0.set_thetagrids(
        np.degrees(angles[:-1]),
        selected_metrics,
        fontsize=label_size,
        color=font_color
    )
    ax0.tick_params(axis='x', pad=pad_val, colors=font_color)
    ax0.tick_params(axis='y', colors=font_color)
    ax0.set_title(
        f"{player} – {games} jogos",
        color=font_color, fontsize=title_size,
        fontweight="semibold", y=title_offset
    )
    ax0.set_ylim(0,1)

    # ─── Tabela (transparente) ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    tbl_data = build_table_data(player)
    total    = metric_w + value_w + rank_w
    colWidths = [metric_w/total, value_w/total, rank_w/total]
    tbl = ax1.table(
        cellText=tbl_data,
        cellLoc="center",
        colWidths=colWidths,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for cell in tbl.properties()["celld"].values():
        cell.set_facecolor((1,1,1,0))         # fundo transparente
        cell.set_edgecolor(line_color)        # mesma cor das linhas
        cell.get_text().set_color(font_color) # texto na cor escolhida

    # ─── Exporta PNG em buffer ────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    img = buf.getvalue()

    # ─── Miniatura + expander ───────────────────────────────────────────────────
    col = cols[idx % 3]
    col.image(img, width=200)
    with col.expander(f"{player} (clique para ampliar)"):
        st.pyplot(fig, use_container_width=True)
        st.download_button(
            "📥 Download Radar + Tabela",
            img,
            file_name=f"{player}_{selected_pos}.png",
            mime="image/png"
        )
