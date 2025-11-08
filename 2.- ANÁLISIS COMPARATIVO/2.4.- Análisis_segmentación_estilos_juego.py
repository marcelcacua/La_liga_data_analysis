#4.- SEGMENTACIÓN DE ESTILOS DE JUEGOS POR TEMPORADA.

#Este análisis segmenta a los equipos de LaLiga según su estilo de juego por temporada.
#Se normalizan métricas por 90 minutos y se calculan indicadores de estilo (posesión, juego directo, finalización, presión alta, defensa baja y juego aéreo).
#Luego se aplica PCA para reducir la dimensionalidad y K-Means para agrupar equipos con estilos similares, evaluando la cantidad óptima de clusters con la métrica de Silhouette.
#Finalmente, se generan gráficos de varianza, heatmaps de cargas y visualizaciones 3D interactivas para explorar los clusters y perfiles de juego de los equipos

import pandas as pd
import numpy as np                
import matplotlib.pyplot as plt   
import seaborn as sns             
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA             
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
import plotly.express as px

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip
df_expand

per90 = [
    "Sho", "SoT", "SoT%", "Gls", "Ast", "KP", "Crs", "PrgC", "PrgP", "PrgR",
    "Tkl", "Int", "Clr", "Blocks", "Off", "Fld"
]

for col in per90:
    df[col+'_per90'] = df[col]/df['90s']

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

style_metrics = {
    "Posesion": {
        "Cmp%": 0.2,
        "Cmp%-C": 0.2,
        "Cmp%-M": 0.2,
        "PrgP_per90": 0.25,
        "PPA": 0.15
    },
    "Juego_directo": {
        "Cmp%-L": 0.3,
        "PrgC_per90": 0.3,
        "Crs_per90": 0.25,
        "Off_per90": 0.15
    },
    "Finalizacion": {
        "Gls_per90": 0.5,
        "SoT_per90": 0.3,
        "Sho_per90": 0.2
    },
    "Presion_alta": {
        "Tkl_per90": 0.4,
        "Int_per90": 0.4,
        "Att 3rd": 0.2
    },
    "Defensa_baja": {
        "Clr_per90": 0.4,
        "Blocks_per90": 0.4,
        "Def 3rd": 0.2
    },
    "Juego_aereo": {
        "Aerial-W": 0.6,
        "Aerial-L": 0.4
    }
}

all_metrics = set(m for metrics_dict in style_metrics.values() for m in metrics_dict)
df_scaled = df.copy()

scaler = StandardScaler()
available_metrics = [m for m in all_metrics if m in df.columns] 
df_scaled[available_metrics] = scaler.fit_transform(df_scaled[available_metrics])

df_styles = df_scaled.copy()

for style, metrics_dict in style_metrics.items():
    metrics_in_df = [m for m in metrics_dict if m in df_scaled.columns]
    if metrics_in_df:
        df_styles[style] = sum(
            df_scaled[m] * metrics_dict[m] for m in metrics_in_df
        )

def analyze_team_styles_by_category(df, style_metrics, season_filter, n_components=10):

    team_stats = df.groupby(['Squad','Season']).mean(numeric_only=True).reset_index()

    team_stats = team_stats[team_stats['Season'] == season_filter].copy()

    all_metrics = set(m for metrics_dict in style_metrics.values() for m in metrics_dict)
    available_metrics = [m for m in all_metrics if m in df.columns]

    scaler = StandardScaler()
    team_stats_scaled = team_stats.copy()
    team_stats_scaled[available_metrics] = scaler.fit_transform(team_stats_scaled[available_metrics])

    # -------------------------------
    # 1. Calcular estilos (promedio de métricas escaladas por estilo)
    # -------------------------------

    df_styles = team_stats_scaled.copy()
    for style, metrics_dict in style_metrics.items():
        metrics_in_df = [m for m in metrics_dict if m in team_stats_scaled.columns]
        if metrics_in_df:
            df_styles[style] = sum(
                team_stats_scaled[m] * metrics_dict[m] for m in metrics_in_df
        )

    style_cols = list(style_metrics.keys())

    # -------------------------------
    # 2. Escalar los estilos
    # -------------------------------

    scaler_styles = StandardScaler()
    X_scaled = scaler_styles.fit_transform(df_styles[style_cols])
    df_styles_scaled = df_styles.copy()
    df_styles_scaled[style_cols] = X_scaled

    # -------------------------------
    # 3. Silhouette y K-Means
    # -------------------------------

    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Número óptimo de clusters según Silhouette: {optimal_k}")

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    df_styles_scaled['Cluster'] = kmeans_final.fit_predict(X_scaled)

    #-------------------------------
    # 4. PCA
    #-------------------------------
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    for i in range(n_components):
        df_styles_scaled[f'PCA{i+1}'] = pca_result[:, i]
    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza explicada por los {n_components} componentes: {explained_variance}")

    # -----------------------------
    # 5. Gráfico de Codo - Autovalores
    # -----------------------------

    autovalores = pca.explained_variance_

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(autovalores)+1), autovalores, marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=8, label='Autovalor')
    plt.axhline(y=1, color='#d62728', linestyle='-', linewidth=2, label='Autovalor = 1') 
    plt.scatter(np.where(autovalores < 1)[0]+1, autovalores[autovalores < 1], color='#ff7f0e', s=120, edgecolors='k', label='Autovalor < 1')

    plt.title(f'Gráfico de codo (autovalores)', fontsize=16, fontweight='bold')
    plt.xlabel('Número de Componentes Principales', fontsize=13)
    plt.ylabel('Autovalor (varianza explicada por componente)', fontsize=13)
    plt.xticks(range(1, len(autovalores)+1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 6. Varianza Explicada Acumulada
    # -----------------------------

    cum_var_exp = explained_variance.cumsum()
    components = [f'PC{i+1}' for i in range(len(explained_variance))]

    plt.bar(components, explained_variance, color='#d62728', alpha=0.7, label='Varianza individual')
    plt.plot(components, cum_var_exp, marker='o', color='#ff7f0e', linewidth=2, label='Varianza Explicada Acumulada')
    
    for i, val in enumerate(cum_var_exp):
        plt.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='#333333')

    plt.title(f'Varianza explicada por PCA)', fontsize=16, fontweight='bold')
    plt.xlabel('Componentes principales', fontsize=13)
    plt.ylabel('Proporción de varianza explicada', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 7. Heatmap de Cargas (Loadings)
    # -----------------------------

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index = style_cols
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
    plt.title(f'Cargas de las Variables en los {n_components} Primeros Componentes Principales', fontsize=14, fontweight='bold')
    plt.ylabel('Variables', fontsize=12)
    plt.xlabel('Componentes Principales', fontsize=12)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 8. Filtrar por temporada y graficar
    # -------------------------

    df_plot = df_styles_scaled.copy()
    df_plot['Team'] = df_plot['Squad']

    fig = px.scatter_3d(
        df_plot,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color='Cluster',
        hover_data=['Squad', 'Season'],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        text = 'Team'
    )

    fig.update_traces(textposition='top center')
    fig.update_layout(
        title=f"Estilos de Juego por Temporada: {season_filter}",
        legend_title="Cluster",
        width=1000,
        height=600
    )

    fig.show()

    return df_styles_scaled

df_results = analyze_team_styles_by_category(df, style_metrics, season_filter='2024-2025', n_components=3)
