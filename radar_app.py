import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ─── Configuração global de fonte Montserrat Semi-Bold ─────────────────────────
mpl.rcParams["font.family"] = "Montserrat"
mpl.rcParams["font.weight"] = "semibold"

# ─── Interface Streamlit ──────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📊 Radar de Desempenho - MIDs do CBLOL")

uploaded_file = st.file_uploader(
    "📁 Faça upload de um CSV com as métricas (formato: métrica na 1ª coluna)",
    type="csv"
)

if uploaded_file:
    # Lê e exibe o DataFrame editável
    df = pd.read_csv(uploaded_file)
    st.subheader("📝 Edite os valores (opcional)")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df = edited_df.copy()

    # Prepara os dados
    all_players = df.columns[1:].tolist()
    metrics    = df.iloc[:, 0].tolist()
    data       = df.set_index(df.columns[0])

    # Inverte 'Avg deaths' (quanto menos, melhor)
    if "Avg deaths" in data.index:
        data.loc["Avg deaths"] = data.loc["Avg deaths"].max() - data.loc["Avg deaths"]

    # Normaliza entre 0 e 1 por métrica
    norm_data = data.copy()
    for m in norm_data.index:
        mn = norm_data.loc[m].min()
        mx = norm_data.loc[m].max()
        norm_data.loc[m] = (norm_data.loc[m] - mn) / (mx - mn) if mx != mn else 0.5

    # Calcula a média normalizada por métrica
    avg_norm   = norm_data.mean(axis=1)
    avg_values = avg_norm.tolist()
    avg_values += avg_values[:1]

    # ─── Controles Visuais na Sidebar ──────────────────────────────────────────
    st.sidebar.header("⚙️ Ajustes Visuais")
    pad_val       = st.sidebar.slider("Espaçamento das labels (pad)",          min_value=0,   max_value=30,   value=12)
    title_offset  = st.sidebar.slider("Offset vertical do título (y)",       min_value=0.9, max_value=1.3,  value=1.08)
    font_size     = st.sidebar.slider("Tamanho da fonte das labels",         min_value=8,   max_value=20,   value=10)
    title_size    = st.sidebar.slider("Tamanho da fonte do título",          min_value=10,  max_value=30,   value=16)
    radar_color   = st.sidebar.color_picker("Cor do radar",                     value="#00de55")
    avg_color     = st.sidebar.color_picker("Cor da linha de média",           value="#FF0000")
    alpha_val     = st.sidebar.slider("Transparência do preenchimento",       min_value=0.0, max_value=1.0, value=0.4)

    # Seleção de jogadores
    st.sidebar.header("🎯 Selecione os jogadores")
    selected_players = st.sidebar.multiselect("Jogadores", all_players, default=all_players)

    if selected_players:
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for player in selected_players:
            values = norm_data[player].tolist()
            values += values[:1]

            fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

            # Linha tracejada da média
            ax.plot(angles, avg_values,
                    linestyle='--',
                    color=avg_color,
                    linewidth=1.5,
                    label='Média')

            # Radar do jogador
            ax.plot(angles, values,
                    color=radar_color,
                    linewidth=2,
                    label=player)
            ax.fill(angles, values,
                    color=radar_color,
                    alpha=alpha_val)

            # Grades e labels dos eixos em Semi-Bold
            ax.set_thetagrids(
                np.degrees(angles[:-1]),
                metrics,
                fontsize=font_size,
                fontweight='semibold'
            )
            # Afastar labels angulares
            ax.tick_params(axis='x', pad=pad_val)

            # Título em Montserrat Semi-Bold, elevado
            ax.set_title(
                player,
                fontsize=title_size,
                fontweight='semibold',
                y=title_offset
            )

            ax.set_ylim(0, 1)

            # Legenda com Semi-Bold
            leg = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            for txt in leg.get_texts():
                txt.set_fontweight('semibold')

            st.pyplot(fig)
