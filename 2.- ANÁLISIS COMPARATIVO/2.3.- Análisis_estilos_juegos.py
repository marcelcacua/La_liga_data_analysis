#3.- ANALISIS ESTILOS DE JUEGO

# Se compararán los estilos de juego de los equipos: Posesión, Juego Directo,
# Ataque Interior, Finalización, Presión Alta, Defensa Baja y Juego Aéreo.
# Cada eje del Radar Chart representa un estilo de juego, usando métricas normalizadas 
# por 90 minutos (per90), permitiendo visualizar un perfil de juego de manera clara e intuitiva.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("C:\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip
df_expand

per90 = [
    "Sho", "SoT", "Gls", "Ast", "G+A", "KP", "Crs", "PrgC", "PrgP", "PrgR", "PPA",
    "Att", "Cmp", "P-1/3", "TotDist", "Tkl", "TklW", "Tkl-Dri", "Int", "Clr", "Blocks",
    "Aerial-W", "Aerial-L","Off", "Fld", "Att-L", "Sh"
]

LALIGA_COLORS = {
    "1": "#E30613", 
    "2": "#0072CE",    
    "3": "#FFC600",     
    "4": "#888888"     
}

for col in per90:
    df[col+'_per90'] = df[col]/df['90s']

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

style_metrics = {
    "Posesion": ["Cmp%", "Cmp%-C", "Cmp%-M", "PrgP_per90", "TotDist_per90", "P-1/3_per90"],
    "Juego_directo": ["Cmp%-L", "Att-L_per90", "PrgC_per90", "PrgR_per90", "Crs_per90", "Off_per90"],
    "Ataque_interior": ["KP_per90", "Ast_per90", "PPA_per90", "PrgP_per90"],
    "Finalizacion": ["Sho_per90", "SoT_per90", "SoT%", "G+A_per90"],
    "Presion_alta": ["Tkl_per90", "Int_per90", "Att 3rd_per90", "Tkl-Dri_per90"],
    "Defensa_baja": ["Clr_per90", "Blocks_per90", "Sh_per90", "Def 3rd_per90"],
    "Juego_aereo": ["Aerial-W_per90", "Aerial-L_per90", "Aerial-W%"]
}

team_stats_season = df.groupby(['Squad', 'Season']).mean(numeric_only=True)
team_styles_season = pd.DataFrame(index=team_stats_season.index)

for style, metrics in style_metrics.items():
    
    available_metrics = [m for m in metrics if m in team_stats_season.columns]
    team_styles_season[style] = team_stats_season[available_metrics].mean(axis=1)

team_styles_season.replace([np.inf, -np.inf], np.nan, inplace=True)
team_styles_season.fillna(0, inplace=True)

scaler = StandardScaler()
team_styles_scaled = pd.DataFrame(
    scaler.fit_transform(team_styles_season),
    columns=team_styles_season.columns,
    index=team_styles_season.index
)

def radar_style_squads(team1, season1, team2, season2, color1=LALIGA_COLORS['1'], color2=LALIGA_COLORS['2']):
    categories = team_styles_scaled.columns
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values1 = team_styles_scaled.loc[(team1, season1)].values.flatten().tolist()
    values1 += values1[:1]
    values2 = team_styles_scaled.loc[(team2, season2)].values.flatten().tolist()
    values2 += values2[:1]

    fig, axes = plt.subplots(1, 2, figsize=(12,6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(angles, values1, linewidth=2, color=color1)
    ax.fill(angles, values1, alpha=0.3, color=color1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_title(f"Perfil de Estilo: {team1} ({season1})", size=14, fontweight="bold", pad=20)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(angles, values2, linewidth=2, color=color2)
    ax.fill(angles, values2, alpha=0.3, color=color2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_title(f"Perfil de Estilo: {team2} ({season2})", size=14, fontweight="bold", pad=20)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Comparación de Estilos de Juego por Temporada", size=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

radar_style_squads(team1="Barcelona", season1='2024-2025', team2="Real Madrid", season2='2024-2025', color1=LALIGA_COLORS["1"], color2=LALIGA_COLORS["2"])

