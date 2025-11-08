#2.- ANALISIS POR JUGADOR Y TEMPORADAS EN LA LIGA

# Este análisis permite evaluar la evolución del rendimiento de un jugador de LaLiga
# a lo largo de las temporadas, comparando su desempeño con el promedio de jugadores
# de la misma posición. Se normalizan las métricas clave por posición y temporada,
# generando un índice de rendimiento relativo que se visualiza para observar tendencias
# individuales y comparativas dentro de la liga.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip
df_expand

def evolucion_jugador(df_expand, jugador):

    categorias_por_posicion = {
        'df': ['TklW', 'Blocks', 'Int', 'Clr', 'Aerial-W'],
        'mf': ['Cmp%', 'KP', 'PrgP', 'TklW', 'Int', 'P-1/3', 'Ast'],
        'fw': ['Gls', 'Sho', 'SoT%', 'PrgC', 'Ast', 'KP']
    }

    colores = {'df': '#003DA5', 'mf': "#00D620", 'fw': '#E30613'}

    df_all = df_expand.copy()

    df_all['Pos_jug'] = df_all['Pos_jug'].astype(str).str.lower()
    df_all['Player'] = df_all['Player'].astype(str)

    df_jugador = df_all[df_all['Player'].str.lower() == jugador.lower()].copy()
    if df_jugador.empty:
        print(f"No se encontraron datos para {jugador}.")
        return

    posiciones = {pos for p in df_jugador['Pos_jug'] for pos in ['df','mf','fw'] if pos in p}
    if not posiciones:
        print(f"{jugador} no tiene posiciones reconocidas como DF, MF o FW.")
        return

    resultados = []
    promedios_liga = []

    df_numeric = df_all.copy()
    for pos, cats in categorias_por_posicion.items():
        df_numeric[cats] = df_numeric[cats].apply(pd.to_numeric, errors='coerce')

    for pos in posiciones:
        categorias = categorias_por_posicion[pos]

        df_pos_jugador = df_jugador[df_jugador['Pos_jug'].str.contains(pos)].copy()
        df_pos_jugador[categorias] = df_pos_jugador[categorias].apply(pd.to_numeric, errors='coerce')
        df_pos_jugador = df_pos_jugador.dropna(subset=categorias + ['Season'])
        if df_pos_jugador.empty:
            continue

        df_pos_scaled_list = []
        liga_avg_list = []

        seasons = sorted(df_pos_jugador['Season'].unique(), key=lambda s: s) 
        for season in seasons:
            
            df_season_pos = df_numeric[
                (df_numeric['Season'] == season) &
                (df_numeric['Pos_jug'].str.contains(pos))
            ].copy()

            if df_season_pos.empty:
                continue

            df_season_pos = df_season_pos.dropna(subset=categorias)
            if df_season_pos.empty:
                continue

            scaler = MinMaxScaler()
            scaler.fit(df_season_pos[categorias])

            df_season_pos_scaled = df_season_pos.copy()
            df_season_pos_scaled[categorias] = scaler.transform(df_season_pos_scaled[categorias])
            df_season_pos_scaled['Rendimiento'] = df_season_pos_scaled[categorias].mean(axis=1)

            liga_avg = df_season_pos_scaled['Rendimiento'].mean()
            liga_avg_list.append({'Season': season, 'Pos': pos, 'Rendimiento': liga_avg})

            df_temp = df_pos_jugador[df_pos_jugador['Season'] == season].copy()
            df_temp[categorias] = scaler.transform(df_temp[categorias])
            df_temp['Rendimiento'] = df_temp[categorias].mean(axis=1)
            df_pos_scaled_list.append(df_temp)

        if not df_pos_scaled_list:
            continue

        df_pos_scaled = pd.concat(df_pos_scaled_list)
        resumen = df_pos_scaled.groupby('Season', as_index=False)['Rendimiento'].mean()
        resumen['Pos'] = pos
        resultados.append(resumen)

        df_liga_avg = pd.DataFrame(liga_avg_list)
        promedios_liga.append(df_liga_avg)

    if not resultados:
        print("No se pudo calcular el rendimiento.")
        return

    df_rendimiento = pd.concat(resultados).sort_values('Season')
    df_promedios = pd.concat(promedios_liga).sort_values('Season')

    fig, ax = plt.subplots(figsize=(10,6))

    temporadas_orden = sorted(df_rendimiento['Season'].unique(), key=lambda s: s)

    for pos in df_rendimiento['Pos'].unique():
        df_plot = df_rendimiento[df_rendimiento['Pos'] == pos].set_index('Season').reindex(temporadas_orden)
        df_liga = df_promedios[df_promedios['Pos'] == pos].set_index('Season').reindex(temporadas_orden)

        ax.plot(temporadas_orden, df_plot['Rendimiento'].values,
                marker='o', color=colores[pos], linewidth=2, label=f"{pos.upper()} - {jugador}")

        ax.plot(temporadas_orden, df_liga['Rendimiento'].values,
                linestyle='--', color=colores[pos], alpha=0.8, label=f"{pos.upper()} - Promedio Liga")

        y_player = df_plot['Rendimiento'].values
        y_liga = df_liga['Rendimiento'].values
        x = np.arange(len(temporadas_orden))
        mask = ~np.isnan(y_player) & ~np.isnan(y_liga)
        if mask.any():
            ax.fill_between(
                np.array(temporadas_orden)[mask],
                y_player[mask],
                y_liga[mask],
                where=(y_player[mask] >= y_liga[mask]),
                interpolate=True,
                alpha=0.12,
                color=colores[pos]
            )
            ax.fill_between(
                np.array(temporadas_orden)[mask],
                y_player[mask],
                y_liga[mask],
                where=(y_player[mask] < y_liga[mask]),
                interpolate=True,
                alpha=0.08,
                color='gray'
            )

    ax.set_title(f"Evolución del rendimiento relativo de {jugador}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Temporada", fontsize=12)
    ax.set_ylabel("Rendimiento normalizado (vs liga)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Referencia", loc='best')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    print("\nRendimiento del jugador por temporada y posición:")
    print(df_rendimiento.reset_index(drop=True))
    print("\nPromedios de la liga (por temporada y posición):")
    print(df_promedios.reset_index(drop=True))

    return df_rendimiento.reset_index(drop=True), df_promedios.reset_index(drop=True)

evolucion_jugador(df_expand, 'Vinicius Júnior')
evolucion_jugador(df_expand, 'Lamine Yamal')
