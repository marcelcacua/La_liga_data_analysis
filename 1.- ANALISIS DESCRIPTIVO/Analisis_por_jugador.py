import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")
df.head()

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip()
df_expand

# Se realiza un análisis mediante un Radar Chart que muestra el rendimiento de los jugadores
# durante la temporada seleccionada. Además, compara el desempeño de aquellos jugadores
# que pueden desenvolverse en más de una posición dentro del campo.

def radar_jugador(df_expand, season, jugador):

    categorias_por_posicion = {
        'df' : ['TklW', 'Tkl%', 'Blocks', 'Int', 'Clr', 'Aerial-W'],
        'mf' : ['Cmp%', 'KP', 'PrgP', 'TklW', 'Int', 'P-1/3', 'Ast'],
        'fw' : ['Gls', 'Sho', 'SoT%', 'PrgC', 'Ast', 'KP']
    }

    colores = {
    'df' : '#003DA5',
    'mf' : "#00D620",
    'fw' : '#E30613'
}
    leyenda_categorias = {
    'TklW': 'Tackles ganados',
    'Tkl%': 'Porcentaje de tackles ganados',
    'Blocks': 'Bloqueos',
    'Int': 'Intercepciones',
    'Clr': 'Despejes',
    'Aerial-W': 'Duelo aéreo ganado',
    'Cmp%': 'Porcentaje de pases completados',
    'KP': 'Key passes (pases clave)',
    'PrgP': 'Pases progresivos',
    'P-1/3': 'Pases al tercio final',
    'Ast': 'Asistencias',
    'Gls': 'Goles',
    'Sh': 'Disparos',
    'SoT%': 'Porcentaje de disparos a portería',
    'PrgC': 'Carreras progresivas'
    }

    jugador_info = df_expand[
        (df_expand['Season'] == season) &
        (df_expand['Player'].str.lower() == jugador.lower())
    ].copy()

    if jugador_info.empty:
        print(f'No hay datos para la teporada {season}.')
        return
     
    posiciones = set()
    for pos in jugador_info['Pos_jug'].str.lower().unique():
        if 'df' in pos: posiciones.add('df')
        if 'mf' in pos: posiciones.add('mf')
        if 'fw' in pos: posiciones.add('fw')

    if not posiciones:
        print(f"{jugador} no tiene posiciones reconocidas como DF, MF o FW en {season}.")
        return

    fig, axes = plt.subplots(1, len(posiciones), figsize=(6*len(posiciones), 6),
                             subplot_kw=dict(polar=True))

    if len(posiciones) == 1:
        axes = [axes]  # Para iterar aunque sea un solo gráfico

    # Crear radar para cada posición detectada
    for ax, pos in zip(axes, posiciones):
        categorias = categorias_por_posicion[pos]

        # Filtrar jugadores de esa posición y temporada
        df_pos = df_expand[
            (df_expand['Season'] == season) &
            (df_expand['Pos_jug'].str.lower().str.contains(pos))
        ].copy()
    
        scaler = MinMaxScaler()
        df_pos[categorias] = scaler.fit_transform(df_pos[categorias])

        jugador_row = df_pos[df_pos['Player'].str.lower() == jugador.lower()].iloc[0]

        # Mostrar valores escalados en tabla
        tabla_stats = pd.DataFrame({
        'Categoría': categorias,
        'Valor escalado (0-1)': jugador_row[categorias].values
        })

        print(f"\nResultados de {jugador} como {pos.upper()}:")
        print(tabla_stats)
        print("-" * 50)

        values = jugador_row[categorias].tolist()
        values += values[:1]

            # Ángulos
        N = len(categorias)
        angles = [n/float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        color_pos = colores[pos]

            # Dibujar radar
        ax.plot(angles, values, color= color_pos, linewidth=2)
        ax.fill(angles, values, color=color_pos, alpha=0.25)

            # Resaltar mejor valor
        best_idx = np.argmax(values[:-1])
        ax.plot(angles[best_idx], values[best_idx], 'o',
                    color='#333333', markersize=6, markeredgecolor='black')

            # Configuración estética
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, fontsize=11, fontweight='bold', color='#333333')
        ax.set_yticklabels([])
        ax.grid(color="#cccccc", linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_facecolor("white")
        ax.set_title(f"{jugador} — {pos.upper()}", size=15, fontweight='bold', pad=15, color='#111111')
    
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=color_pos, edgecolor='black', label=f"{str(k)}: {leyenda_categorias.get(str(k), 'Descripción no disponible')}")
            for k in list(categorias)
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.95, 1), ncol=1, fontsize=5, frameon=False)
    
    plt.suptitle(f"Radar de {jugador} ({season})", fontsize=17, fontweight='bold', color="#111111")
    plt.show()

#Ejecución para analizar la última temporada correspondiente a 2024-2025 de Raphinha y Vinicius Jr.
radar_jugador(df_expand, season='2024-2025', jugador='Raphinha')
radar_jugador(df_expand, season='2024-2025', jugador='Vinicius Júnior')

