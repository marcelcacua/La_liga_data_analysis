import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Se realiza un primer analisis global y profundo sobre los jugadores que juegan como posicion principal o secundaria de defensor.

df = pd.read_excel("C:\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")
df.head()
df.columns

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip()
df_expand

#1.- ANALISIS DEFENSORES - DEF.

#Se realizó un análisis exploratorio general con el propósito de obtener una visión inicial sobre la participación de los defensores 
#durante la temporada 2024-2025. Este análisis incluyó la elaboración de varios gráficos descriptivos que permitieron observar, por un lado, 
#el número de defensores utilizados por cada equipo, y por otro, la distribución de los minutos jugados por estos jugadores. Asimismo, se 
#construyó un boxplot para identificar la dispersión y las diferencias en la cantidad de minutos acumulados entre los distintos defensores.
#Además, se elaboró un resumen global que sintetiza la información principal del conjunto de datos, incluyendo la cantidad de registros y 
# jugadores únicos, el número de equipos representados, y las estadísticas generales de minutos jugados. Este resumen permite comprender de 
# forma rápida el volumen total de participación, los valores promedio y la variabilidad en el tiempo de juego entre los defensores.

def df_stats_global(df_expand, season,
                               min_col='Min', mp_col='MP', starts_col='Starts',
                               top_teams=20, mostrar_plots=True):

    defensa = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df'))
    ].copy()

    n_registros = len(defensa)
    n_jugadores_unicos = defensa['Player'].nunique()
    equipos_unicos = defensa['Squad'].nunique()

    if min_col in defensa.columns:
        minutos_total = defensa[min_col].sum(skipna=True)
        minutos_mean = defensa[min_col].mean(skipna=True)
        minutos_median = defensa[min_col].median(skipna=True)
        minutos_std = defensa[min_col].std(skipna=True)
    else:
        minutos_total = minutos_mean = minutos_median = minutos_std = None

    noventas_col = '90s'
    if noventas_col in defensa.columns:
        noventas_total = defensa[noventas_col].sum(skipna=True)
        noventas_mean = defensa[noventas_col].mean(skipna=True)
    else:
        noventas_total = noventas_mean = None 

    distrib_team = (defensa
                    .groupby('Squad')['Player']
                    .nunique()
                    .sort_values(ascending=False)
                    .rename('Defensores_unicos')
                    .reset_index())

    top_por_minutos = None
    if min_col in defensa.columns:
        top_por_minutos = (defensa
                           .groupby('Player')[min_col]
                           .sum()
                           .sort_values(ascending=False)
                           .head(15)
                           .reset_index()
                           .rename(columns={min_col: 'Minutos'}))
        
    print("=== RESUMEN GLOBAL: temporada", season, "===\n")
    print(f"Registros (filas) de defensas: {n_registros}")
    print(f"Jugadores defensores únicos: {n_jugadores_unicos}")
    print(f"Equipos con defensas: {equipos_unicos}")
    if minutos_total is not None:
        print(f"Minutos totales (suma): {minutos_total:.0f}")
        print(f"Minutos promedio por registro: {minutos_mean:.1f}")
        print(f"Minutos mediana: {minutos_median:.1f}")
        print(f"Desviación estándar minutos: {minutos_std:.1f}")
    if noventas_total is not None:
        print(f"Total 90s (suma): {noventas_total:.1f}, media 90s: {noventas_mean:.2f}")
    print("\nTop equipos por número de defensores únicos (primeras filas):")
    print(distrib_team.head(top_teams).to_string(index=False))

    if mostrar_plots:

        # Gráfico 1: Barras defensores únicos por equipo
        TopN = distrib_team.head(top_teams)
        plt.figure(figsize=(10,6))
        bars = plt.bar(TopN['Squad'], TopN['Defensores_unicos'], color='#E30613')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Defensores únicos')
        plt.title(f'Defensores únicos por equipo — Temporada {season} (Top {top_teams})', fontweight='bold', color='#333333')

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

        # Gráfico 2: Histograma minutos
        if min_col in defensa.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(defensa[min_col], bins=30, color='#E30613', edgecolor='#333333')
            plt.xlabel('Minutos jugados')
            plt.ylabel('Frecuencia')
            plt.title(f'Distribución de minutos jugados por defensas — {season}', fontweight='bold')
            plt.tight_layout()
            plt.show()

            # Gráfico 3: Boxplot minutos
            plt.figure(figsize=(6,4))
            sns.boxplot(x=defensa[min_col], color='#333333')
            plt.xlabel('Minutos jugados')
            plt.title(f'Boxplot de minutos jugados — {season}', fontweight='bold')
            plt.tight_layout()
            plt.show()          
    
    resultados = {
        'df_defensa': defensa,
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
df_stats_global(df_expand, season='2024-2025', min_col='Min', mostrar_plots=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#En una segunda etapa del análisis, se llevó a cabo un estudio detallado de las métricas individuales más relevantes 
#para evaluar el desempeño de los defensores durante la temporada seleccionada y para aquellos jugadores que han jugado
#al menos el 30% de los minutos totales del campeonato.

#De manera paralela se realiza el mismo análisis, pero incorporando una función que permite seleccionar un equipo específico. Esto ofrece una visión más focalizada y menos global, 
#centrada en el rendimiento defensivo de los jugadores de un solo club. De esta forma, se puede evaluar y comparar el desempeño individual dentro del equipo, 
#obteniendo una perspectiva más directa sobre su gestión defensiva.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#2.- ANALISIS CARACTERISTICAS DEFENSIVAS - DEF
 
minutos_minimos = 1026
jugadores_relevantes = df_expand[df_expand['Min']>= minutos_minimos]

def df_stats_def(df_expand, season, minutos_minimos=1026, top_n = 20, mostrar_plots=True):

    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    defensas.columns = defensas.columns.str.strip()

    metricas = {
        'Entradas ganadas / 90 min' : lambda d: d['TklW'] / d['90s'],
        'Éxito en entradas (%)': lambda d: d['Tkl-Dri%'],
        'Intercepciones / 90 min': lambda d: d['Int'] / d['90s'],
        'Despejes / 90 min': lambda d: d['Clr'] / d['90s'],
        'Bloqueos totales / 90min': lambda d: d['Blocks'] / d['90s'],
        'Bloqueos de remates / 90min': lambda d: d['Block-Sh'] / d['90s'],
        'Entradas zona defensiva / 90min': lambda d: d['Def 3rd'] / d['90s'],
        'Entradas zona media / 90min': lambda d: d['Mid 3rd'] / d['90s'],
        'Entradas zona ataque / 90min': lambda d: d['Att 3rd'] / d['90s'],
        'Éxito en duelos aéreos (%)': lambda d: d['Aerial-W'] / (d['Aerial-W'] + d['Aerial-L'])
    }

    resultados = {}

    sns.set_style('whitegrid')
    sns.set_context('talk')
    color_barras = '#E30613'

    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores,
        }

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, color=color_barras, order=top_mejores['Player'])
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.title(f'Top{top_n} — {nombre_metrica} ({season})', fontsize=14, fontweight='bold', color='#E30613')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
df_stats_def(df_expand, season='2024-2025', minutos_minimos=1026, top_n=20, mostrar_plots=True)

def df_stats_def_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Squad'] == squad) &
        (df_expand['Pos_jug'].str.lower().str.contains('df'))
    ].copy()

    defensas.columns = defensas.columns.str.strip()

    defensas['Relevante_30%_temporada'] = defensas['Min'] >= minutos_minimos

    metricas = {
        'Entradas ganadas / 90 min' : lambda d: d['TklW'] / d['90s'],
        'Éxito en entradas (%)': lambda d: d['Tkl%'],
        'Intercepciones / 90 min': lambda d: d['Int'] / d['90s'],
        'Despejes / 90 min': lambda d: d['Clr'] / d['90s'],
        'Bloqueos totales / 90min': lambda d: d['Blocks'] / d['90s'],
        'Bloqueos de remates / 90min': lambda d: d['Block-Sh'] / d['90s'],
        'Entradas zona defensiva / 90min': lambda d: d['Def 3rd'] / d['90s'],
        'Entradas zona media / 90min': lambda d: d['Mid 3rd'] / d['90s'],
        'Entradas zona ataque / 90min': lambda d: d['Att 3rd'] / d['90s'],
        'Éxito en duelos aéreos (%)': lambda d: d['Aerial-W'] / (d['Aerial-W'] + d['Aerial-L'])
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)

        defensas[nombre_metrica] = defensas[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores,
        }

        if mostrar_plots:
            plt.figure(figsize=(8,5))
             
            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.title(f'Analisis defensivo {squad} — {nombre_metrica} - {season}', fontsize=14, fontweight='bold', color='#E30613')
            plt.xlabel(nombre_metrica, fontsize=12, color='#1A1A1A')
            plt.ylabel('Jugador', fontsize=12, color='#1A1A1A')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)

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
df_stats_def_squad(df_expand, season='2024-2025', squad='Real Madrid', mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#3.- ANALISIS SOBRE LA POSESIÓN DE BALON, PASES Y SALIDA OFENSIVA - DEF

def df_stats_pas(df_expand, season, minutos_minimos=1026, top_n = 20, mostrar_plots=True):

    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    defensas.columns = defensas.columns.str.strip()

    metricas = {
    'Pases totales por 90 minutos': lambda d: d['Att'] / d['90s'],
    'Pases completados por 90 minutos': lambda d: d['Cmp'] / d['90s'],
    'Porcentaje de pases completados (%)': lambda d: d['Cmp%'],
    'Pases largos completados por 90 minutos': lambda d: d['Cmp-L'] / d['90s'],
    'Éxito de pases largos (%)': lambda d: d['Cmp%-L'],
    'Pases de media distancia completados por 90 minutos': lambda d: d['Cmp-M'] / d['90s'],
    'Éxito de pases media distancia (%)': lambda d: d['Cmp%-M'],
    'Pases cortos completados por 90 minutos': lambda d: d['Cmp-C'] / d['90s'],
    'Éxito de pases cortos (%)': lambda d: d['Cmp%-C'],

    'Distancia total recorrida por pases (yds)': lambda d: d['TotDist'],
    'Distancia progresiva hacia portería rival (yds)': lambda d: d['PrgDist'],

    'Total de asistencias': lambda d: d['Ast'],
    'Total de pases clave': lambda d: d['KP'],
    'Pases en el tercio final del campo': lambda d: d['P-1/3'],
    'Pases en el área de penal': lambda d: d['PPA'],

    'Total de centros realizados': lambda d: d['Crs'],
    'Centros por 90 minutos': lambda d: d['Crs'] / d['90s']
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', nombre_metrica]].head(top_n)
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores,
        }

        sns.set_style('whitegrid')
        sns.set_context('talk')
        color_barras = '#E30613'

        if mostrar_plots:
            plt.figure(figsize=(8,5))
            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, color=color_barras)

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
df_stats_pas(df_expand, '2024-2025', minutos_minimos=1026, top_n = 20, mostrar_plots=True)

def df_stats_pas_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):

    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Squad'] == squad)
    ].copy()

    defensas.columns = defensas.columns.str.strip()

    defensas['Relevante_30%_temporada'] = defensas['Min'] >= minutos_minimos

    metricas = {
    'Pases totales por 90 minutos': lambda d: d['Att'] / d['90s'],
    'Pases completados por 90 minutos': lambda d: d['Cmp'] / d['90s'],
    'Porcentaje de pases completados (%)': lambda d: d['Cmp%'],
    'Pases largos completados por 90 minutos': lambda d: d['Cmp-L'] / d['90s'],
    'Éxito de pases largos (%)': lambda d: d['Cmp%-L'],
    'Pases de media distancia completados por 90 minutos': lambda d: d['Cmp-M'] / d['90s'],
    'Éxito de pases media distancia (%)': lambda d: d['Cmp%-M'],
    'Pases cortos completados por 90 minutos': lambda d: d['Cmp-C'] / d['90s'],
    'Éxito de pases cortos (%)': lambda d: d['Cmp%-C'],

    'Distancia total recorrida por pases (yds)': lambda d: d['TotDist'],
    'Distancia progresiva hacia portería rival (yds)': lambda d: d['PrgDist'],

    'Total de asistencias': lambda d: d['Ast'],
    'Total de pases clave': lambda d: d['KP'],
    'Pases en el tercio final del campo': lambda d: d['P-1/3'],
    'Pases en el área de penal': lambda d: d['PPA'],

    'Total de centros realizados': lambda d: d['Crs'],
    'Centros por 90 minutos': lambda d: d['Crs'] / d['90s']
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace([pd.NA, float('inf'), -float('inf')], 0)

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores,
        }

        if mostrar_plots:
            plt.figure(figsize=(8,5))

            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e'}).tolist()

            sns.barplot(x=nombre_metrica, y='Player', data=top_mejores, palette=colores)
            plt.title(f'Analisis de pases {squad} — {nombre_metrica} ({season})', fontsize=14, fontweight='bold', color='#E30613')
            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, linestyle='--', color='lightgray', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.3)

            from matplotlib.patches import Patch
            legend_element = [
                Patch(facecolor='#E30613', label=f'Jugador con al menos {minutos_minimos} min. (30% de la temporada)'),
                Patch(facecolor='#ff7f0e', label=f'Jugador con menos de {minutos_minimos} min.')
            ]

            plt.legend(handles= legend_element, title='Tiempo jugado', fontsize=9, title_fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
df_stats_pas_squad(df_expand, '2024-2025', squad='Barcelona', minutos_minimos=1026, mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#3.- ANALISIS SOBRE LAS FALTAS COMETIDAS Y SANCIONES RECIBIDAS - DEF

def df_stats_disc(df_expand, season, minutos_minimos=1026, top_n=10, mostrar_plots=True):
    # Filtrar defensas de la temporada
    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Min']>=minutos_minimos)
    ].copy()
    
    defensas.columns = defensas.columns.str.strip()
  
    metricas = {
        'Total faltas' : lambda d: d['Fls'],
        'Total tarjetas amarillas' : lambda d: d['CrdY'],
        'Total tarjetas rojas' : lambda d: d['CrdR'],
        'Faltas cometidas por 90 min': lambda d: d['Fls'] / d['90s'],
        'Tarjetas amarillas por 90 min': lambda d: d['CrdY'] / d['90s'],
        'Tarjetas rojas por 90 min': lambda d: d['CrdR'] / d['90s']
    }

    resultados = {}
    
    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[
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
                      fontsize=14, fontweight='bold', color=color_barras)
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=9, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
    
    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025
df_stats_disc(df_expand, season='2024-2025', minutos_minimos=1026, top_n=10, mostrar_plots=True)

def df_stats_disc_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots=True):
    
    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df'))&
        (df_expand['Squad'] == squad)
    ].copy()

    if defensas.empty:
        print(f'No hay datos para la teporada {season}.')
        return
    
    defensas.columns = defensas.columns.str.strip()

    defensas['Relevante_30%_temporada'] = defensas['Min'] >= minutos_minimos

    metricas = {
        'Total faltas' : lambda d: d['Fls'],
        'Total tarjetas amarillas' : lambda d: d['CrdY'],
        'Total tarjetas rojas' : lambda d: d['CrdR'],
        'Faltas cometidas por 90 min': lambda d: d['Fls'] / d['90s'],
        'Tarjetas amarillas por 90 min': lambda d: d['CrdY'] / d['90s'],
        'Tarjetas rojas por 90 min': lambda d: d['CrdR'] / d['90s']
    }

    resultados = {}
    
    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]]
        
        resultados[nombre_metrica] = {
            'Mejores' : top_mejores
        }

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
df_stats_disc_squad(df_expand, season='2024-2025', squad='Atlético Madrid', mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#4.- ANALISIS DE ESTADISTICAS OFENSIVAS - DF

def df_stats_of(df_expand, season, minutos_minimos=1026, top_n = 10, mostrar_plots = True):

    defensa = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Min']>=minutos_minimos)
    ].copy()

    defensa.columns = defensa.columns.str.strip()

    metricas = {
        'Goles totales': lambda d: d['Gls'],
        'Goles cada 90 min': lambda d: d['Gls'] / d['90s'],
        'Asistencias totales': lambda d: d['Ast'],
        'Asistencias cada 90 min': lambda d: d['Ast'] / d['90s'],
        'Remates totales cada 90 min': lambda d: d['Sh'] / d['90s'],
        'Remates a puerta cada 90 min': lambda d: d['SoT'] / d['90s']
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        defensa[nombre_metrica] = formula(defensa)
        defensa[nombre_metrica] = defensa[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = defensa.sort_values(nombre_metrica, ascending=False)[
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

#Ejecución para analizar la última temporada correspondiente a 2024-2025.
df_stats_of(df_expand, season='2024-2025', minutos_minimos=1026, top_n=5, mostrar_plots=True)

def df_stats_of_squad(df_expand, season, squad, minutos_minimos=1026, mostrar_plots = True):

    defensas = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Pos_jug'].str.lower().str.contains('df')) &
        (df_expand['Squad'] == squad)
    ].copy()

    if defensas.empty:
        print(f'No hay datos para la teporada {season}.')
        return

    defensas.columns = defensas.columns.str.strip()

    defensas['Relevante_30%_temporada'] = defensas['Min'] >= minutos_minimos

    metricas = {
        'Goles totales': lambda d: d['Gls'],
        'Goles cada 90 min': lambda d: d['Gls'] / d['90s'],
        'Asistencias totales': lambda d: d['Ast'],
        'Asistencias cada 90 min': lambda d: d['Ast'] / d['90s'],
        'Remates totales cada 90 min': lambda d: d['Sh'] / d['90s'],
        'Remates a puerta cada 90 min': lambda d: d['SoT'] / d['90s']
    }

    resultados = {}

    for nombre_metrica, formula in metricas.items():
        defensas[nombre_metrica] = formula(defensas)
        defensas[nombre_metrica] = defensas[nombre_metrica].replace(
            [pd.NA, float('inf'), -float('inf')], 0
        )

        top_mejores = defensas.sort_values(nombre_metrica, ascending=False)[
            ['Player', 'Squad', 'Relevante_30%_temporada', nombre_metrica]
        ]
        
        resultados[nombre_metrica] = top_mejores

        if mostrar_plots:
            plt.figure(figsize=(8,5))

            colores = top_mejores['Relevante_30%_temporada'].map({True : '#E30613', False : '#ff7f0e' }).tolist()

            sns.barplot(
                x=nombre_metrica,
                y='Player',
                data=top_mejores,
                palette=colores
            )

            for i, v in enumerate(top_mejores[nombre_metrica]):
                plt.text(v + 0.01, i, f'{v: .2f}', va='center', fontsize=10, color='#333333')

            plt.title(f"Analisis aportación ofensiva {squad} — {nombre_metrica} ({season})", 
                      fontsize=14, fontweight='bold', color='#333333')
            plt.xlabel(nombre_metrica, fontsize=12, color='#333333')
            plt.ylabel('Jugador', fontsize=12, color='#333333')
            plt.xticks(fontsize=10, color='#333333')
            plt.yticks(fontsize=10, color='#333333')
            plt.grid(True, which='major', linewidth=0.5, color='lightgrey', linestyle='--', alpha=0.5)
            plt.xlim(0, top_mejores[nombre_metrica].max() * 1.3)
            from matplotlib.patches import Patch

            legend_element = [
                Patch(facecolor='#E30613', label=f'Jugador con al menos {minutos_minimos} min (30% temporada)'),
                Patch(facecolor='#ff7f0e', label=f'Jugador con menos de {minutos_minimos} min')
            ]
            plt.legend(handles=legend_element, title='Tiempo jugado', fontsize=9, title_fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            plt.show()

    return resultados

#Ejecución para analizar la última temporada correspondiente a 2024-2025.
df_stats_of_squad(df_expand, season='2024-2025', squad='Barcelona', minutos_minimos=1026, mostrar_plots=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#5.- RADAR CHART DE LOS JUGADORES MAS DESTACADOS DEFENSIVAMENTE - DF

def radar_top6_def(df_expand, season, minutos_minimos=1026, top_n=6):
    
    df_temp = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('df') &
        (df_expand['Min'] >= minutos_minimos)
    ].copy()

    if df_temp.empty:
        print(f"No hay defensas relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    categorias = ['TklW', 'Tkl-Dri%', 'Blocks','Int','Clr','Aerial%']
    
    scaler = MinMaxScaler()
    df_temp[categorias] = scaler.fit_transform(df_temp[categorias])

    df_temp['total_score'] = df_temp[categorias].sum(axis=1)

    df_temp = df_temp.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    df_temp['rank'] = df_temp.index + 1
    
    tabla_ranking = df_temp[['rank', 'Player', 'Squad', 'Min', 'total_score'] + categorias].copy()
    pd.set_option('display.float_format', '{:,.3f}'.format)
    print("\n=== TOP DEFENSAS — RANKING ===\n")
    print(tabla_ranking.head(top_n).to_string(index=False))

    top_defensas = df_temp.sort_values(by='total_score', ascending=False).head(top_n)

    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    laliga_colors = ['#E30613']

    n_cols = min(top_n, 3)
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    
    cmap= plt.cm.get_cmap('tab10', top_n)
    color_list = [cmap(i) for i in range(top_n)]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, top_defensas.iterrows())):
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
    
    for j in range(len(top_defensas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.suptitle(f"Mejores defensas de LaLiga - {season}",fontsize=18, fontweight='bold', color="#111111")
     
    plt.show()
    print(df_temp['total_score'])

#Ejecución para analizar la última temporada correspondiente a 2024-2025.
radar_top6_def(df_expand, season='2024-2025', minutos_minimos=1026, top_n=6)

def radar_top6_def_squad(df_expand, season, squad, minutos_minimos=1026, top_n=6):
    
    defensas = df_expand[
        (df_expand['Season'] == season) &
        df_expand['Pos_jug'].str.lower().str.contains('df')
    ].copy()

    if defensas.empty:
        print(f"No hay defensas relevantes para la temporada {season} con al menos {minutos_minimos} minutos.")
        return
    
    defensas['Relevante_30%_temporada'] = defensas['Min'] >= minutos_minimos

    categorias = ['TklW', 'Tkl-Dri%', 'Blocks','Int','Clr','Aerial-W']
    
    scaler = MinMaxScaler()
    defensas[categorias] = scaler.fit_transform(defensas[categorias])

    df_temp = defensas[defensas['Squad'] == squad].copy()

    if df_temp.empty:
        print(f"No hay defensas relevantes para el equipo {squad} en la temporada {season}.")
        return

    df_temp['total_score'] = df_temp[categorias].sum(axis=1)

    df_temp = df_temp.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    df_temp['rank'] = df_temp.index + 1
    
    tabla_ranking = df_temp[['rank', 'Player', 'Squad', 'Min', 'total_score'] + categorias].copy()
    pd.set_option('display.float_format', '{:,.3f}'.format)
    print("\n=== TOP DEFENSAS — RANKING ===\n")
    print(tabla_ranking.head(top_n).to_string(index=False))

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
        Patch(facecolor='#E30613', label=f'Jugador con al menos {minutos_minimos} min (30% temporada)'),
        Patch(facecolor='#ff7f0e', label=f'Jugador con menos de {minutos_minimos} min')
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

    fig.suptitle(f"Mejores defensas de LaLiga - {season}",fontsize=18, fontweight='bold', color="#111111")

    plt.show()

#Ejecución para analizar la última temporada correspondiente a 2024-2025.
radar_top6_def_squad(df_expand, season='2024-2025', squad='Barcelona', minutos_minimos=1026, top_n=6)
