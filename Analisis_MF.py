import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_excel("C:\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")
df.head()
df.columns

#Creamos una nueva columna en la cual separe los jugadores que cuentan con mas de una posición.

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip()
df_expand

#ANALISIS MEDIO CENTROS - MF

def mf_stats_global (df_expand, season, min_col='Min', top_teams=20, mostrar_plots=True):

    midfielder = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))
    ].copy()

    n_registros = len(midfielder)
    n_jugadores_unicos = midfielder['Player'].nunique()
    equipos_unicos = midfielder['Squad'].nunique()

    if min_col in midfielder.columns:
        minutos_total= midfielder[min_col].sum(skipna=True)
        minutos_mean=midfielder[min_col].mean(skipna=True)
        minutos_median=midfielder[min_col].median(skipna=True)
        minutos_std=midfielder[min_col].std(skipna=True)
    else:
        minutos_total=minutos_mean=minutos_median=minutos_std=None

    noventas_col = '90s'
    if noventas_col in midfielder.columns:
        noventas_total=midfielder[noventas_col].sum(skipna=True)
        noventas_mean=midfielder[noventas_col].mean(skipna=True)
    else:
        noventas_total=noventas_mean=None
    
    distrib_team = (midfielder
                    .groupby('Squad')['Player']
                    .nunique()
                    .sort_values(ascending=False)
                    .rename('midfielder_unicos')
                    .reset_index())

    top_por_minutos = None
    if min_col in midfielder.columns:
        top_por_minutos = (midfielder
                           .groupby('Player')[min_col]
                           .sum()
                           .sort_values(ascending=False)
                           .head(15)
                           .reset_index()
                           .rename(columns={min_col: 'Minutos'}))
        
    print("=== RESUMEN GLOBAL: temporada", season, "===\n")
    print(f"Registros (filas) de medio centros: {n_registros}")
    print(f"Jugadores medio centros únicos: {n_jugadores_unicos}")
    print(f"Equipos con medios centros: {equipos_unicos}")
    if minutos_total is not None:
        print(f"Minutos totales (suma): {minutos_total:.0f}")
        print(f"Minutos promedio por registro: {minutos_mean:.1f}")
        print(f"Minutos mediana: {minutos_median:.1f}")
        print(f"Desviación estándar minutos: {minutos_std:.1f}")
    if noventas_total is not None:
        print(f"Total 90s (suma): {noventas_total:.1f}, media 90s: {noventas_mean:.2f}")
    print("\nTop equipos por número de medio centros únicos (primeras filas):")
    print(distrib_team.head(top_teams).to_string(index=False))

    if mostrar_plots:
        # Gráfico 1: Barras defensores únicos por equipo
        TopN = distrib_team.head(top_teams)
        plt.figure(figsize=(10,6))
        bars = plt.bar(TopN['Squad'], TopN['midfielder_unicos'], color='#E30613')
        plt.grid(True, which='major', color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=9, color='#333333')
        plt.ylabel('Medio centros', fontsize=12)
        plt.title(f'Medio centros únicos por equipo — Temporada {season}', fontweight='bold', color='#333333')

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

        if min_col in midfielder.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(midfielder[min_col], bins=30, color='#E30613', edgecolor='#333333')
            plt.xlabel('Minutos jugados', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.title(f'Distribución de minutos jugados por medio centros — {season}', fontweight='bold', color='#333333')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6,4))
            sns.boxplot(x=midfielder[min_col], color='#E30613')
            plt.xlabel('Minutos jugados', fontsize=12)
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.title(f'Boxplot de minutos jugados — {season}', fontweight='bold', color='#333333')
            plt.tight_layout()
            plt.show()          
    
    resultados = {
        'df_defensa': midfielder,
        'n_registros': n_registros,
        'n_jugadores_unicos': n_jugadores_unicos,
        'equipos_unicos': equipos_unicos,
        'minutos_total': minutos_total,
        'minutos_mean': minutos_mean,
        'minutos_median': minutos_median,
        'minutos_std': minutos_std,
        '90s_total': noventas_total,
        '90s_mean': noventas_mean,
        'distrib_team': distrib_team,
        'top_por_minutos': top_por_minutos
    }
    return resultados

# Ejecución para temporada 2024-2025
mf_stats_global(df_expand, season='2024-2025', min_col='Min', mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS CARACTERISTICAS DE CREACIÓN - MF

#Se procede a analizar los distintos aspectos relacionados con la contribución y la distribución del juego, considerados como los elementos más relevantes en esta posición.
#Para asegurar que el análisis sea representativo, se aplica un filtro que incluye únicamente a los jugadores que han disputado al menos el 30% de los minutos posibles en la temporada.
 
minutos_minimos = 1026
jugadores_relevantes = df_expand[df_expand['Min']>= minutos_minimos]

def mf_stats(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    midfielder = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    metricas = {
    #Volumen
    'Pases completados totales': lambda d: d['Cmp'],
    'Pases intentados totales': lambda d: d['Att'],
    'Pases progresivos totales': lambda d: d['PrgP'],
    'Regates progresivos totales': lambda d: d['PrgC'],
    'Pases clave totales': lambda d: d['KP'],
    'Asistencias totales': lambda d: d['Ast'],
    #Normalizado (por 90min)
    'Pases completados / 90min': lambda d: d['Cmp'] / d['90s'],
    'Pases intentados / 90min': lambda d: d['Att'] / d['90s'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    'Pases progresivos recibidos / 90min': lambda d: d['PrgR'] / d['90s'],
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'Pases en zona ofensiva / 90min': lambda d: d['P-1/3'] / d['90s'],
    'Pases al área penal / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    # Eficiencia
    'Precisión en pases (%)': lambda d: d['Cmp%'],
    'Pases cortos completados (%)': lambda d: d['Cmp%-C'],
    'Pases medios completados (%)': lambda d: d['Cmp%-M'],
    'Pases largos completados (%)': lambda d: d['Cmp%-L'],
    # Distancias (volumen y normalizado)
    'Distancia total de pases (yardas)': lambda d: d['TotDist'],
    'Distancia progresiva de pases (yardas)': lambda d: d['PrgDist'],
    'Distancia total de pases (yardas/90min)': lambda d: d['TotDist'] / d['90s'],
    'Distancia progresiva de pases (yardas/90min)': lambda d: d['PrgDist'] / d['90s'],
}

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'
    
    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, color=color_barras, order=top_mejores['Player'])
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.title(f'Top{top_n} — {nombre_metrica} ({season})', fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

def mf_stats_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    midfielder = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    midfielder['Relevante_30%_temporada'] = midfielder['Min'] >= minutos_minimos
    
    metricas = {
    'Pases completados totales': lambda d: d['Cmp'],
    'Pases intentados totales': lambda d: d['Att'],
    'Pases progresivos totales': lambda d: d['PrgP'],
    'Regates progresivos totales': lambda d: d['PrgC'],
    'Pases clave totales': lambda d: d['KP'],
    'Asistencias totales': lambda d: d['Ast'],
    'Pases completados / 90min': lambda d: d['Cmp'] / d['90s'],
    'Pases intentados / 90min': lambda d: d['Att'] / d['90s'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    'Pases completados progresivos / 90min': lambda d: d['PrgR'] / d['90s'],
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'Pases en zona ofensiva / 90min': lambda d: d['P-1/3'] / d['90s'],
    'Pases al área penal / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    'Precisión en pases (%)': lambda d: d['Cmp%'],
    'Pases cortos completados (%)': lambda d: d['Cmp%-C'],
    'Pases medios completados (%)': lambda d: d['Cmp%-M'],
    'Pases largos completados (%)': lambda d: d['Cmp%-L'],
    'Distancia total de pases (yardas)': lambda d: d['TotDist'],
    'Distancia progresiva de pases (yardas)': lambda d: d['PrgDist'],
    'Distancia total de pases (yardas/90min)': lambda d: d['TotDist'] / d['90s'],
    'Distancia progresiva de pases (yardas/90min)': lambda d: d['PrgDist'] / d['90s'],
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots: 

            plt.figure(figsize=(8,5))
                
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'Analisis defensivo MF {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#333333')
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#1A1A1A')
            plt.ylabel('Jugador', fontsize=12, color='#1A1A1A')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.3)

            from matplotlib.patches import Patch
            legend_elements = [
                    Patch(facecolor='#E30613', label=f'Jugador con al menos {minutos_minimos} min (30% temporada)'),
                    Patch(facecolor='#ff7f0e', label=f'Jugador con menos de {minutos_minimos} min.')
                ]
            plt.legend(handles=legend_elements, title='Tiempo jugado', fontsize=9, title_fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_squad(df_expand, season='2024-2025', squad='Real Madrid', mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS DEFENSIVO - MF

#Se procede a analizar caracteristicas defensivas de los medio centros.
def mf_stats_def(df_expand, season, minutos_minimos, top_n = 20, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    metricas = {
    #Entradas y duelos
    'Entradas totales': lambda d: d['Tkl'],
    'Entradas totales / 90min': lambda d: d['Tkl'] / d['90s'],
    'Entradas ganadas totales': lambda d: d['TklW'],
    'Entradas ganadas / 90min': lambda d: d['TklW'] / d['90s'],
    'Éxito en entradas (%)': lambda d: d['Tkl%'],
    'Entradas vs regates totales': lambda d: d['Tkl-Dri'],
    'Entradas vs regates / 90min': lambda d: d['Tkl-Dri'] / d['90s'],
    # Zonas del campo
    'Entradas en zona defensiva totales': lambda d: d['Def 3rd'],
    'Entradas en zona defensiva / 90min': lambda d: d['Def 3rd'] / d['90s'],
    'Entradas en zona media totales': lambda d: d['Mid 3rd'],
    'Entradas en zona media / 90min': lambda d: d['Mid 3rd'] / d['90s'],
    'Entradas en zona de ataque totales': lambda d: d['Att 3rd'],
    'Entradas en zona de ataque / 90min': lambda d: d['Att 3rd'] / d['90s'],
    # Intercepciones y recuperaciones
    'Intercepciones totales': lambda d: d['Int'],
    'Intercepciones / 90min': lambda d: d['Int'] / d['90s'],
    'Bloqueos totales': lambda d: d['Blocks'],
    'Bloqueos totales / 90min': lambda d: d['Blocks'] / d['90s'],
    'Bloqueos de remates totales': lambda d: d['Sh'],
    'Bloqueos de remates / 90min': lambda d: d['Sh'] / d['90s'],
    'Bloqueos de pases totales': lambda d: d['Pass'],
    'Bloqueos de pases / 90min': lambda d: d['Pass'] / d['90s'],
    'Despejes totales': lambda d: d['Clr'],
    'Despejes / 90min': lambda d: d['Clr'] / d['90s'],
    # Duelos aéreos
    'Duelos aéreos ganados totales': lambda d: d['Aerial-W'],
    'Duelos aéreos ganados / 90min': lambda d: d['Aerial-W'] / d['90s'],
    'Duelos aéreos perdidos totales': lambda d: d['Aerial-L'],
    'Duelos aéreos perdidos / 90min': lambda d: d['Aerial-L'] / d['90s'],
    'Éxito en duelos aéreos (%)': lambda d: d['Aerial-W'] / (d['Aerial-W'] + d['Aerial-L']),
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', nombre_metrica]
        ].head(top_n)
        
        resultados[nombre_metrica] = top_mejores

        sns.set_style('whitegrid')
        sns.set_context('talk')
        color_barras = '#E30613'

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(
                x=nombre_metrica, 
                y='Player', 
                data=top_mejores, 
                color=color_barras
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Top {top_n} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_def(df_expand, season='2024-2025', minutos_minimos=1026, top_n = 20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.
def mf_stats_def_squad(df_expand, season, squad, minutos_minimos, top_n = 20, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    midfielder['Relevante_30%_temporada'] = midfielder['Min'] >= minutos_minimos

    metricas = {
    #Entradas y duelos
    'Entradas totales': lambda d: d['Tkl'],
    'Entradas totales / 90min': lambda d: d['Tkl'] / d['90s'],
    'Entradas ganadas totales': lambda d: d['TklW'],
    'Entradas ganadas / 90min': lambda d: d['TklW'] / d['90s'],
    'Éxito en entradas (%)': lambda d: d['Tkl%'],
    'Entradas vs regates totales': lambda d: d['Tkl-Dri'],
    'Entradas vs regates / 90min': lambda d: d['Tkl-Dri'] / d['90s'],
    # Zonas del campo
    'Entradas en zona defensiva totales': lambda d: d['Def 3rd'],
    'Entradas en zona defensiva / 90min': lambda d: d['Def 3rd'] / d['90s'],
    'Entradas en zona media totales': lambda d: d['Mid 3rd'],
    'Entradas en zona media / 90min': lambda d: d['Mid 3rd'] / d['90s'],
    'Entradas en zona de ataque totales': lambda d: d['Att 3rd'],
    'Entradas en zona de ataque / 90min': lambda d: d['Att 3rd'] / d['90s'],
    # Intercepciones y recuperaciones
    'Intercepciones totales': lambda d: d['Int'],
    'Intercepciones / 90min': lambda d: d['Int'] / d['90s'],
    'Bloqueos totales': lambda d: d['Blocks'],
    'Bloqueos totales / 90min': lambda d: d['Blocks'] / d['90s'],
    'Bloqueos de remates totales': lambda d: d['Blocks-Sh'],
    'Bloqueos de remates / 90min': lambda d: d['Blocks-Sh'] / d['90s'],
    'Bloqueos de pases totales': lambda d: d['Pass'],
    'Bloqueos de pases / 90min': lambda d: d['Pass'] / d['90s'],
    'Despejes totales': lambda d: d['Clr'],
    'Despejes / 90min': lambda d: d['Clr'] / d['90s'],
    # Duelos aéreos
    'Duelos aéreos ganados totales': lambda d: d['Aerial-W'],
    'Duelos aéreos ganados / 90min': lambda d: d['Aerial-W'] / d['90s'],
    'Duelos aéreos perdidos totales': lambda d: d['Aerial-L'],
    'Duelos aéreos perdidos / 90min': lambda d: d['Aerial-L'] / d['90s'],
    'Éxito en duelos aéreos (%)': lambda d: d['Aerial-W'] / (d['Aerial-W'] + d['Aerial-L']),
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]
        ]
        
        resultados[nombre_metrica] = top_mejores

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()
            sns.barplot(
                x=nombre_metrica, 
                y='Player', 
                data=top_mejores, 
                palette=colores
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Registro de infracciones y sanciones del {squad} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.1)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#E30613', label=f'≥ {minutos_minimos} min (30% temporada)'),
                Patch(facecolor='#ff7f0e', label=f'< {minutos_minimos} min')
            ]
            plt.legend(
                handles=legend_elements,
                title="Tiempo jugado",
                fontsize=9,
                title_fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.02, 1)
            )

            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_def_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS DISCIPLINARIO - MF 

#Se procede a realizar un analisis sobre las faltas concedidas y sanciones obtenidas por parte de los medio centros.
def mf_stats_disc(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf')) &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    metricas = {
    # Totales
    'Tarjetas amarillas totales': lambda d: d['CrdY'],
    'Dobles amarillas totales': lambda d: d['2CrdY'],
    'Tarjetas rojas directas totales': lambda d: d['CrdR'],
    'Tarjetas totales': lambda d: d['CrdY'] + d['CrdR'],
    # Normalizadas por 90 min
    'Tarjetas amarillas / 90min': lambda d: d['CrdY'] / d['90s'],
    'Dobles amarillas / 90min': lambda d: d['2CrdY'] / d['90s'],
    'Tarjetas rojas / 90min': lambda d: d['CrdR'] / d['90s'],
    'Tarjetas totales / 90min': lambda d: (d['CrdY']  + d['CrdR']) / d['90s'],
    # Relacionadas con faltas
    'Faltas cometidas / 90min': lambda d: d['Fls'] / d['90s'],
    'Faltas recibidas / 90min': lambda d: d['Fld'] / d['90s'],
    # Métrica de eficiencia disciplinaria
    'Faltas por tarjeta': lambda d: d['Fls'] / (d['CrdY'] + d['CrdR']).replace(0, 1)
    }
    
    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', nombre_metrica]
        ].head(top_n)
        
        resultados[nombre_metrica] = top_mejores

        sns.set_style('whitegrid')
        sns.set_context('talk')
        color_barras = '#E30613'

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(
                x=nombre_metrica,
                y='Player',
                data=top_mejores,
                color=color_barras
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Top {top_n} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.1)
            plt.tight_layout()
            plt.show()

    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_disc(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.
def mf_stats_disc_squad(df_expand, season, squad, minutos_minimos, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf')) &
        (df_expand['Squad'] == squad)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    midfielder['Relevante_30%_temporada'] = midfielder['Min'] >= minutos_minimos

    metricas = {
    # Totales
    'Tarjetas amarillas totales': lambda d: d['CrdY'],
    'Dobles amarillas totales': lambda d: d['2CrdY'],
    'Tarjetas rojas directas totales': lambda d: d['CrdR'],
    'Tarjetas totales': lambda d: d['CrdY'] + d['CrdR'],
    # Normalizadas por 90 min
    'Tarjetas amarillas / 90min': lambda d: d['CrdY'] / d['90s'],
    'Dobles amarillas / 90min': lambda d: d['2CrdY'] / d['90s'],
    'Tarjetas rojas / 90min': lambda d: d['CrdR'] / d['90s'],
    'Tarjetas totales / 90min': lambda d: (d['CrdY']  + d['CrdR']) / d['90s'],
    # Relacionadas con faltas
    'Faltas cometidas / 90min': lambda d: d['Fls'] / d['90s'],
    'Faltas recibidas / 90min': lambda d: d['Fld'] / d['90s'],
    # Métrica de eficiencia disciplinaria
    'Faltas por tarjeta': lambda d: d['Fls'] / (d['CrdY'] + d['CrdR']).replace(0, 1)
    }
    
    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]
        ]
        
        resultados[nombre_metrica] = top_mejores

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()
            sns.barplot(
                x=nombre_metrica, 
                y='Player', 
                data=top_mejores, 
                palette=colores
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Registro de infracciones y sanciones del {squad} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.1)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#E30613', label=f'≥ {minutos_minimos} min (30% temporada)'),
                Patch(facecolor='#ff7f0e', label=f'< {minutos_minimos} min')
            ]
            plt.legend(
                handles=legend_elements,
                title="Tiempo jugado",
                fontsize=9,
                title_fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.02, 1)
            )

            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_disc_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS OFENSIVO - MF

#Se procede a realizar un analisis con respecto a la aportación ofensiva de los medio centros de La Liga.
def mf_stats_of(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf')) &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    metricas = {
    # Producción directa
    'Goles totales': lambda d: d['Gls'],
    'Asistencias totales': lambda d: d['Ast'],
    'G+A totales': lambda d: d['Gls'] + d['Ast'],
    # Normalizadas a 90 min
    'Goles / 90min': lambda d: d['Gls'] / d['90s'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'G+A / 90min': lambda d: (d['Gls'] + d['Ast']) / d['90s'],
    # Remates
    'Remates totales': lambda d: d['Sh'],
    'Remates / 90min': lambda d: d['Sh'] / d['90s'],
    'Remates a puerta / 90min': lambda d: d['Sot'] / d['90s'],
    'Precisión remates a puerta (%)': lambda d: d['Sot%'],
    # Penaltis
    'Penaltis convertidos': lambda d: d['PK'],
    'Penaltis ejecutados': lambda d: d['Pkatt'],
    'Eficacia en penaltis (%)': lambda d: (d['PK'] / d['Pkatt']) if d['Pkatt'] > 0 else 0,
    # Creación de ocasiones
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Pases al área penal / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    # Progresión ofensiva (aportación en construcción hacia gol)
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Pases progresivos completados / 90min': lambda d: d['PrgR'] / d['90s'],
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores,
        }

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, color=color_barras, order=top_mejores['Player'])
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.title(f'Top{top_n} — {nombre_metrica} ({season})', fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_of(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.
def mf_stats_of_squad(df_expand, season, squad, minutos_minimos, mostrar_plots=True):

    midfielder=df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf')) &
        (df_expand['Squad'] == squad)
    ].copy()

    midfielder.columns = midfielder.columns.str.strip()

    midfielder['Relevante_30%_temporada'] = midfielder['Min'] >= minutos_minimos

    metricas = {
    # Producción directa
    'Goles totales': lambda d: d['Gls'],
    'Asistencias totales': lambda d: d['Ast'],
    'G+A totales': lambda d: d['Gls'] + d['Ast'],
    # Normalizadas a 90 min
    'Goles / 90min': lambda d: d['Gls'] / d['90s'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'G+A / 90min': lambda d: (d['Gls'] + d['Ast']) / d['90s'],
    # Remates
    'Remates totales': lambda d: d['Sh'],
    'Remates / 90min': lambda d: d['Sh'] / d['90s'],
    'Remates a puerta / 90min': lambda d: d['Sot'] / d['90s'],
    'Precisión remates a puerta (%)': lambda d: d['Sot%'],
    # Penaltis
    'Penaltis convertidos': lambda d: d['PK'],
    'Penaltis ejecutados': lambda d: d['Pkatt'],
    'Eficacia en penaltis (%)': lambda d: (d['PK'] / d['Pkatt']) if d['Pkatt'] > 0 else 0,
    # Creación de ocasiones
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Pases al área penal / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    # Progresión ofensiva (aportación en construcción hacia gol)
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Pases progresivos completados / 90min': lambda d: d['PrgR'] / d['90s'],
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        midfielder[nombre_metrica] = formula(midfielder)
        midfielder[nombre_metrica] = midfielder[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = midfielder.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]
        ]
        
        resultados[nombre_metrica] = top_mejores

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()
            sns.barplot(
                x=nombre_metrica, 
                y='Player', 
                data=top_mejores, 
                palette=colores
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Analisis ofensivo MF del {squad} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.1)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#E30613', label=f'≥ {minutos_minimos} min (30% temporada)'),
                Patch(facecolor='#ff7f0e', label=f'< {minutos_minimos} min')
            ]
            plt.legend(
                handles=legend_elements,
                title="Tiempo jugado",
                fontsize=9,
                title_fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.02, 1)
            )

            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
mf_stats_of_squad(df_expand, season='2024-2025', squad='Barcelona', minutos_minimos=1026,  mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#RADAR CHART DE LOS MEJORES 6 MEDIO CENTROS - MF

#Se procede a realizar un radar chart que nos muestre los seis mejores medio centros de la liga según las categorias seleccionadas.
def radar_top6_mf(df_expand, season, minutos_minimos=1026, top_n=6):
    # Filtrar por temporada, posición y minutos
    df_temp = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('mf') &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    if df_temp.empty:
        print(f"No hay medio centros relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    categorias = ['Cmp%', 'KP', 'PrgP', 'TklW', 'Int', 'P-1/3', 'Ast']
    
    scaler = MinMaxScaler()
    df_temp[categorias] = scaler.fit_transform(df_temp[categorias])

    df_temp['total_score'] = df_temp[categorias].sum(axis=1)

    top_midfielder = df_temp.sort_values(by='total_score', ascending=False).head(top_n)

    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    laliga_colors = ['#E30613']

    n_cols = min(top_n, 3)
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_midfielder.iterrows())):
        values = row[categorias].tolist()
        values += values[:1]  
        
        color = laliga_colors[idx % len(laliga_colors)]

        ax.plot(angles, values, color=color, linewidth=1)
        ax.fill(angles, values, color=color, alpha=0.15)

        best_idx = np.argmax(row[categorias].values)
        ax.plot(angles[best_idx], values[best_idx], 'o', color='#333333', markersize=5, markeredgecolor='black') 

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, fontsize=10, fontweight='bold', color='#333333')
        ax.set_yticklabels([])
        ax.grid(color="#cccccc", linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_facecolor("white")
        ax.set_title(f"{row['Player']} - {row['Squad']}" , size=13, fontweight='bold', pad=15, color='#111111')
    
    for j in range(len(top_midfielder), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.suptitle(f"Mejores medio centros de LaLiga - {season}",fontsize=18, fontweight='bold', color="#111111")

    plt.show()

#Ejecución para analizar la última temporada correspondiente a 2024-2025
radar_top6_mf(df_expand, season='2024-2025', minutos_minimos=1026, top_n=6)

#Se procede a realizar el analisis anterior estudiando de manera especificas para el equipo deseado.
def radar_top6_mf_squad(df_expand, season, squad, minutos_minimos=1026, top_n=6):
    
    midfielder = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('mf')
    ].copy()

    if midfielder.empty:
        print(f"No hay medios centros relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    midfielder['Relevante_30%_temporada'] = midfielder['Min'] >= minutos_minimos

    categorias = ['Cmp%', 'PrgP', 'P-1/3', 'KP', 'Ast', 'TklW', 'Int', ]
    
    scaler = MinMaxScaler()
    midfielder[categorias] = scaler.fit_transform(midfielder[categorias])

    df_temp = midfielder[midfielder['Squad'] == squad].copy()

    if df_temp.empty:
        print(f"No hay medio centros relevantes para el equipo {squad} en la temporada {season}.")
        return

    df_temp['total_score'] = df_temp[categorias].sum(axis=1)

    top_defensas = df_temp.sort_values(by='total_score', ascending=False).head(top_n)

    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colores = top_defensas['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

    n_cols = min(top_n, 3)
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_defensas.iterrows())):
        values = row[categorias].tolist()
        values += values[:1]  
        
        color = str(colores[idx])

        ax.plot(angles, values, color=color, linewidth=1)
        ax.fill(angles, values, color=color, alpha=0.15)

        best_idx = np.argmax(row[categorias].values)
        ax.plot(angles[best_idx], values[best_idx], 'o', color='#333333', markersize=5, markeredgecolor='black') 

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, fontsize=10, fontweight='bold', color='#333333')
        ax.set_yticklabels([])
        ax.grid(color="#cccccc", linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_facecolor("white")
        ax.set_title(row['Player'], size=13, fontweight='bold', pad=15, color='#111111')
    
    for j in range(len(top_defensas), len(axes)):
        fig.delaxes(axes[j])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E30613', label=f'≥ {minutos_minimos} min (30% temporada)'),
        Patch(facecolor='#ff7f0e', label=f'< {minutos_minimos} min')
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.95, 1),
        fontsize=9,
        title="Tiempo jugado",
        title_fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.suptitle(f"Mejores medio centros del {squad} - {season}",fontsize=18, fontweight='bold', color="#111111")

    plt.show()

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.
radar_top6_mf_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, top_n=6)
