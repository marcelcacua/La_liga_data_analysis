import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")
df.head()
df.columns

#Creamos una nueva columna en la cual separe los jugadores que cuentan con mas de una posición.

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip()
df_expand

#ANALISIS DELANTEROS - FW

#Se lleva a cabo un análisis global de los jugadores cuya posición principal o secundaria es la de delantero

def fw_stats_global(df_expand, season, min_col='Min', top_teams=20, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('fw'))
    ].copy()

    n_registros = len(delantero)
    n_jugadores_unicos = delantero['Player'].nunique()
    equipos_unicos = delantero['Squad'].nunique()

    if min_col in delantero.columns:
        minutos_total= delantero[min_col].sum(skipna=True)
        minutos_mean=delantero[min_col].mean(skipna=True)
        minutos_median=delantero[min_col].median(skipna=True)
        minutos_std=delantero[min_col].std(skipna=True)
    else:
        minutos_total=minutos_mean=minutos_median=minutos_std=None

    noventas_col = '90s'
    if noventas_col in delantero.columns:
        noventas_total=delantero[noventas_col].sum(skipna=True)
        noventas_mean=delantero[noventas_col].mean(skipna=True)
    else:
        noventas_total=noventas_mean=None
    
    distrib_team = (delantero
                    .groupby('Squad')['Player']
                    .nunique()
                    .sort_values(ascending=False)
                    .rename('delanteros_unicos')
                    .reset_index())

    top_por_minutos = None
    if min_col in delantero.columns:
        top_por_minutos = (delantero
                           .groupby('Player')[min_col]
                           .sum()
                           .sort_values(ascending=False)
                           .head(15)
                           .reset_index()
                           .rename(columns={min_col: 'Minutos'}))
        
    print("=== RESUMEN GLOBAL: temporada", season, "===\n")
    print(f"Registros (filas) de delanteros: {n_registros}")
    print(f"Jugadores delanteros únicos: {n_jugadores_unicos}")
    print(f"Equipos con medios centros: {equipos_unicos}")
    if minutos_total is not None:
        print(f"Minutos totales (suma): {minutos_total:.0f}")
        print(f"Minutos promedio por registro: {minutos_mean:.1f}")
        print(f"Minutos mediana: {minutos_median:.1f}")
        print(f"Desviación estándar minutos: {minutos_std:.1f}")
    if noventas_total is not None:
        print(f"Total 90s (suma): {noventas_total:.1f}, media 90s: {noventas_mean:.2f}")
    print("\nTop equipos por número de delanteros únicos (primeras filas):")
    print(distrib_team.head(top_teams).to_string(index=False))

    if mostrar_plots:
        
        TopN = distrib_team.head(top_teams)
        plt.figure(figsize=(10,6))
        bars = plt.bar(TopN['Squad'], TopN['delanteros_unicos'], color='#E30613')
        plt.grid(True, which='major', color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=9, color='#333333')
        plt.ylabel('Delanteros', fontsize=12)
        plt.title(f'Delanteros únicos por equipo — Temporada {season}', fontweight='bold', color='#333333')

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

        if min_col in delantero.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(delantero[min_col], bins=30, color='#E30613', edgecolor='#333333')
            plt.xlabel('Minutos jugados', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.title(f'Distribución de minutos jugados por delanteros — {season}', fontweight='bold')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6,4))
            sns.boxplot(x=delantero[min_col], color='#E30613')
            plt.xlabel('Minutos jugados', fontsize=12)
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.title(f'Boxplot de minutos jugados — {season}', fontweight='bold')
            plt.tight_layout()
            plt.show()          
    
    resultados = {
        'df_defensa': delantero,
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

#Ejecución para analizar la última temporada correspondiente a 2024-2025
fw_stats_global(df_expand, season='2024-2025', min_col='Min', top_teams=20, mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS CARACTERISTICAS DE FINALIZACIÓN Y PELIGRO DE GOL - FW

#Se procede a analizar los distintos aspectos con respecto a la finalización y amenaza de gol de los delanteros 
#con al menos un 30% de participación en el total de los minutos de juego de La Liga.
 
def fw_stats(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('fw'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    metricas = {
    'Goles totales': lambda d: d['Gls'],
    'Goles / 90min': lambda d: d['Gls'] / d['90s'],
    'Goles sin penales': lambda d: d['G-PK'],
    'Goles sin penales / 90min': lambda d: d['G-PK'] / d['90s'],
    'Remates totales': lambda d: d['Sho'],
    'Remates / 90min': lambda d: d['Sho'] / d['90s'],
    'Remates a puerta': lambda d: d['SoT'],
    'Remates a puerta / 90min': lambda d: d['SoT'] / d['90s'],
    'Precisión de tiro (%)': lambda d: d['SoT%'],
    'Distancia media de remate': lambda d: d['Dist'],
    'Contribución directa en goles': lambda d: d['G+A'],
    'Contribución / 90min': lambda d: d['G+A'] / d['90s'],
    'Penales anotados': lambda d: d['PK'],
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'
    
    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)

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
fw_stats(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.

def fw_stats_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    delantero['Relevante_30%_temporada'] = delantero['Min'] >= minutos_minimos
    
    metricas = {
    'Goles totales': lambda d: d['Gls'],
    'Goles / 90min': lambda d: d['Gls'] / d['90s'],
    'Goles sin penales': lambda d: d['G-PK'],
    'Goles sin penales / 90min': lambda d: d['G-PK'] / d['90s'],
    'Remates totales': lambda d: d['Sho'],
    'Remates / 90min': lambda d: d['Sho'] / d['90s'],
    'Remates a puerta': lambda d: d['SoT'],
    'Remates a puerta / 90min': lambda d: d['SoT'] / d['90s'],
    'Precisión de tiro (%)': lambda d: d['SoT%'],
    'Distancia media de remate': lambda d: d['Dist'],
    'Contribución directa en goles': lambda d: d['G+A'],
    'Contribución / 90min': lambda d: d['G+A'] / d['90s'],
    'Penales anotados': lambda d: d['PK'],
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots: 

            plt.figure(figsize=(8,5))
                
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'FW | {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#333333')
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
fw_stats_squad(df_expand, season='2024-2025', squad='Barcelona', minutos_minimos=1026, mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS GENERACIÓN Y CREATIVDAD OFENSIVA

#Se procede a analizar los distintos aspectos con respecto a la creación ofensivo.

def fw_stats_cam(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('fw'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    metricas = {
    'Asistencias totales': lambda d: d['Ast'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'Pases clave': lambda d: d['KP'],
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Pases al último tercio': lambda d: d['P-1/3'],
    'Pases al último tercio / 90min': lambda d: d['P-1/3'] / d['90s'],
    'Pases dentro del área rival': lambda d: d['PPA'],
    'Pases dentro del área / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros totales': lambda d: d['Crs'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    'Pases progresivos totales': lambda d: d['PrgP'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Pases progresivos completados': lambda d: d['PrgR'],
    'Pases progresivos completados / 90min': lambda d: d['PrgR'] / d['90s'],
    'Regates progresivos': lambda d: d['PrgC'],
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'
    
    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)

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
fw_stats_cam(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True )

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.

def fw_stats_cam_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    delantero['Relevante_30%_temporada'] = delantero['Min'] >= minutos_minimos
    
    metricas = {
    'Asistencias totales': lambda d: d['Ast'],
    'Asistencias / 90min': lambda d: d['Ast'] / d['90s'],
    'Pases clave': lambda d: d['KP'],
    'Pases clave / 90min': lambda d: d['KP'] / d['90s'],
    'Pases al último tercio': lambda d: d['P-1/3'],
    'Pases al último tercio / 90min': lambda d: d['P-1/3'] / d['90s'],
    'Pases dentro del área rival': lambda d: d['PPA'],
    'Pases dentro del área / 90min': lambda d: d['PPA'] / d['90s'],
    'Centros totales': lambda d: d['Crs'],
    'Centros / 90min': lambda d: d['Crs'] / d['90s'],
    'Pases progresivos totales': lambda d: d['PrgP'],
    'Pases progresivos / 90min': lambda d: d['PrgP'] / d['90s'],
    'Pases progresivos completados': lambda d: d['PrgR'],
    'Pases progresivos completados / 90min': lambda d: d['PrgR'] / d['90s'],
    'Regates progresivos': lambda d: d['PrgC'],
    'Regates progresivos / 90min': lambda d: d['PrgC'] / d['90s'],
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots: 

            plt.figure(figsize=(8,5))
                
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'FW | {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#333333')
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
fw_stats_cam_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, mostrar_plots=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS PARTICIPACIÓN JUEGO COLECTIVO

#Se procede a analizar los distintos aspectos con respecto a la paticipación de pases en el juego colectivo.

def fw_stats_pas(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('fw'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    metricas = {
    'Pases completados': lambda d: d['Cmp'],
    'Pases completados / 90min': lambda d: d['Cmp'] / d['90s'],
    'Pases intentados': lambda d: d['Att'],
    'Pases intentados / 90min': lambda d: d['Att'] / d['90s'],
    'Precisión de pase (%)': lambda d: d['Cmp%'],
    'Distancia total de pase': lambda d: d['TotDist'],
    'Distancia total de pase / 90min': lambda d: d['TotDist'] / d['90s'],
    'Metros progresivos en pases': lambda d: d['PrgDist'],
    'Metros progresivos / 90min': lambda d: d['PrgDist'] / d['90s'],
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'
    
    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)

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
fw_stats_pas(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.

def fw_stats_pas_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    delantero['Relevante_30%_temporada'] = delantero['Min'] >= minutos_minimos
    
    metricas = {
    'Pases completados': lambda d: d['Cmp'],
    'Pases completados / 90min': lambda d: d['Cmp'] / d['90s'],
    'Pases intentados': lambda d: d['Att'],
    'Pases intentados / 90min': lambda d: d['Att'] / d['90s'],
    'Precisión de pase (%)': lambda d: d['Cmp%'],
    'Distancia total de pase': lambda d: d['TotDist'],
    'Distancia total de pase / 90min': lambda d: d['TotDist'] / d['90s'],
    'Metros progresivos en pases': lambda d: d['PrgDist'],
    'Metros progresivos / 90min': lambda d: d['PrgDist'] / d['90s'],
   }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots: 

            plt.figure(figsize=(8,5))
                
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'FW | {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#333333')
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
fw_stats_pas_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS PARTICIPACIÓN DUELOS FISICOS.

#Se procede a analizar los distintos aspectos con respecto a los duelos fisicos.

def fw_stats_phy(df_expand, season, minutos_minimos, top_n=20, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('fw'))&
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    metricas = {
    'Faltas recibidas': lambda d: d['Fld'],
    'Faltas recibidas / 90min': lambda d: d['Fld'] / d['90s'],
    'Duelos aéreos ganados': lambda d: d['Aerial-W'],
    'Duelos aéreos ganados / 90min': lambda d: d['Aerial-W'] / d['90s'],
    'Duelos aéreos perdidos': lambda d: d['Aerial-L'],
    'Duelos aéreos perdidos / 90min': lambda d: d['Aerial-L'] / d['90s'],
    'Fueras de juego cometidos': lambda d: d['Off'],
    'Fueras de juego / 90min': lambda d: d['Off'] / d['90s'],
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'
    
    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)

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
fw_stats_phy(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.

def fw_stats_phy_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    delantero = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('mf'))&
        (df_expand['Squad'] == squad)
    ].copy()

    delantero.columns = delantero.columns.str.strip()

    delantero['Relevante_30%_temporada'] = delantero['Min'] >= minutos_minimos
    
    metricas = {
    'Faltas recibidas': lambda d: d['Fld'],
    'Faltas recibidas / 90min': lambda d: d['Fld'] / d['90s'],
    'Duelos aéreos ganados': lambda d: d['Aerial-W'],
    'Duelos aéreos ganados / 90min': lambda d: d['Aerial-W'] / d['90s'],
    'Duelos aéreos perdidos': lambda d: d['Aerial-L'],
    'Duelos aéreos perdidos / 90min': lambda d: d['Aerial-L'] / d['90s'],
    'Fueras de juego cometidos': lambda d: d['Off'],
    'Fueras de juego / 90min': lambda d: d['Off'] / d['90s'],
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        delantero[nombre_metrica] = formula(delantero)
        delantero[nombre_metrica] = delantero[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = delantero.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]

        resultados[nombre_metrica] = top_mejores

        if mostrar_plots: 

            plt.figure(figsize=(8,5))
                
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'FW | {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#333333')
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
fw_stats_phy_squad(df_expand, season='2024-2025', squad='Barcelona', minutos_minimos=1026, mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANALISIS RADAR CHART

#Se procede a realizar un radar chart con los mejores delanteros de la liga.

def radar_top6_fw(df_expand, season, minutos_minimos=1026, top_n=6):
    
    delantero = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('fw') &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    if delantero.empty:
        print(f"No hay delanteros relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    categorias = ['Gls', 'Sho', 'SoT%', 'PrgC', 'Ast', 'KP']
    
    scaler = MinMaxScaler()
    delantero[categorias] = scaler.fit_transform(delantero[categorias])
    
    delantero['total_score'] = delantero[categorias].sum(axis=1)

    delantero = delantero.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    delantero['rank'] = delantero.index + 1
    
    tabla_ranking = delantero[['rank', 'Player', 'Squad', 'Min', 'total_score'] + categorias].copy()
    pd.set_option('display.float_format', '{:,.3f}'.format)
    print("\n=== TOP DELANTEROS — RANKING ===\n")
    print(tabla_ranking.head(top_n).to_string(index=False))

    top_delanteros = delantero.sort_values(by='total_score', ascending=False).head(top_n)

    df_long = top_delanteros.melt(
        id_vars=['Player', 'Squad', 'Season'],
        value_vars=categorias,
        var_name='Métrica',
        value_name='Valor Normalizado'
    )

    output_file = f"C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\1.- ANALISIS DESCRIPTIVO\\POWER BI\\TOP6_DELANTEROS_{season}.xlsx"
    df_long.to_excel(output_file, index=False)
    print(f"Datos exportados a {output_file} en formato long (para Power BI).")

    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    laliga_colors = ['#E30613']

    n_cols = min(top_n, 3)
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_delanteros.iterrows())):
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
    
    for j in range(len(top_delanteros), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.suptitle(f"Mejores delanteros de LaLiga - {season}",fontsize=18, fontweight='bold', color="#111111")

    plt.show()

#Ejecución para analizar la última temporada correspondiente a 2024-2025
radar_top6_fw(df_expand, season='2024-2025', minutos_minimos=1026, top_n=6)

#Se procede a realizar el analisis de las metricas anteriormente estudiadas de manera especificas para el equipo deseado.

def radar_top6_fw_squad(df_expand, season, squad, minutos_minimos=1026, top_n=6):
    
    delantero = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('fw')
    ].copy()

    if delantero.empty:
        print(f"No hay delanteros relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    delantero['Relevante_30%_temporada'] = delantero['Min'] >= minutos_minimos

    categorias = ['Gls', 'Sho', 'SoT%', 'PrgC', 'Ast', 'KP']
    
    scaler = MinMaxScaler()
    delantero[categorias] = scaler.fit_transform(delantero[categorias])

    delantero = delantero[delantero['Squad'] == squad].copy()

    if delantero.empty:
        print(f"No hay delanteros relevantes para el equipo {squad} en la temporada {season}.")
        return

    delantero['total_score'] = delantero[categorias].sum(axis=1)

    delantero = delantero.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    delantero['rank'] = delantero.index + 1
    
    tabla_ranking = delantero[['rank', 'Player', 'Squad', 'Min', 'total_score'] + categorias].copy()
    pd.set_option('display.float_format', '{:,.3f}'.format)
    print("\n=== TOP DELANTEROS — RANKING ===\n")
    print(tabla_ranking.head(top_n).to_string(index=False))

    top_delantero = delantero.sort_values(by='total_score', ascending=False).head(top_n)

    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colores = top_delantero['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

    n_cols = min(top_n, 3)
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_delantero.iterrows())):
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
    
    for j in range(len(top_delantero), len(axes)):
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

    fig.suptitle(f"Mejores FW del {squad} - {season}",fontsize=18, fontweight='bold', color="#111111")

    plt.show()

#Ejecución para analizar la última temporada correspondiente a 2024-2025

radar_top6_fw_squad(df_expand, season='2024-2025', squad='Real Madrid', minutos_minimos=1026, top_n=6)
