import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip
df_expand

#1.- ANÁLISIS GENERAL.

# Este análisis estudia la evolución del rendimiento ofensivo y táctico en LaLiga
# durante las últimas cinco temporadas. Se examinan tendencias en goles, asistencias,
# distribución de tipos de pase, juego progresivo (pases y conducciones progresivas),
# acciones defensivas por zonas del campo y la evolución de faltas cometidas y recibidas.
# El estudio se realiza tanto a nivel global como por equipo, con el objetivo de
# identificar patrones y posibles estilos de juego dentro de la competición.

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.frameon": False
})

LALIGA_COLORS = {
    "1": "#E30613", 
    "2": "#0072CE",    
    "3": "#FFC600",     
    "4": "#888888"     
}

def global_stats_laliga(df, squad):

    df = df.copy()

    if squad:
        df = df[df['Squad'] == squad].copy()
        print(f'Se ha filtrado correctamente por el equipo: {squad}')
    else:
        print(f'Se realiza el Analisis Global para toda La Liga')
    
    #Tendencia de goles y asistencias

    trend = df.groupby('Season').agg({
        'Gls' : 'sum',
        'Ast' : 'sum',
        '90s' : 'sum'
    }).reset_index()

    trend['Gls_per90'] = trend['Gls']/trend['90s']
    trend['Ast_per90'] = trend['Ast']/trend['90s']

    plt.figure(figsize=(8,5))
    plt.plot(trend['Season'], trend['Gls_per90'], marker = 'o', color=LALIGA_COLORS['1'], linewidth=3, label='Goles/90')
    plt.plot(trend['Season'], trend['Ast_per90'], marker = 'o', color=LALIGA_COLORS['2'], linewidth=3, label='Asistencias/90')
    for i, val in enumerate(trend['Gls_per90']):
        plt.text(trend["Season"].iloc[i], val+0.002, f"{val:.2f}", ha="center", color='#333333', fontsize=10, weight='bold')
    for i, val in enumerate(trend['Ast_per90']):
        plt.text(trend["Season"].iloc[i], val-0.002, f"{val:.2f}", ha="center", color="#333333", fontsize=10, weight='bold')
    plt.title(f'Tendencia de goles y asistencias en La Liga - {squad}')
    plt.xlabel('Temporada')
    plt.ylabel('Promedio por 90')
    plt.legend(loc='upper left')
    sns.despine()
    plt.show()

    #Variación sobre el estilo de pase

    style = df.groupby("Season").agg({
    "Cmp-C":"sum","Att-C":"sum",
    "Cmp-M":"sum","Att-M":"sum",
    "Cmp-L":"sum","Att-L":"sum"
    }).reset_index()
    style["Short%"] = style["Att-C"] / (style["Att-C"]+style["Att-M"]+style["Att-L"])
    style["Medium%"] = style["Att-M"] / (style["Att-C"]+style["Att-M"]+style["Att-L"])
    style["Long%"] = style["Att-L"] / (style["Att-C"]+style["Att-M"]+style["Att-L"])


    ax = style.plot(x="Season", y=["Short%","Medium%","Long%"],
               kind="bar", stacked=True, figsize=(10,6),
               color=[LALIGA_COLORS['1'], LALIGA_COLORS['2'], LALIGA_COLORS['3'] ],
               title=f"Distribución de tipos de pase (corto/medio/largo) - {squad}")
    plt.ylabel("Proporción")
    plt.xlabel('Temporada')
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="center", fontsize=9, color="white", weight="bold")
    sns.despine()
    plt.legend(loc='best')
    plt.show()

    #Variación sobre la progresión del juego.

    prog = df.groupby("Season").agg({"PrgP":"mean","PrgC":"mean"}).reset_index()
    plt.figure(figsize=(8,5))
    plt.plot(prog["Season"], prog["PrgP"], marker="o", linewidth=3, color=LALIGA_COLORS['1'], label="Pases progresivos")
    plt.plot(prog["Season"], prog["PrgC"], marker="o", linewidth=3, color=LALIGA_COLORS['2'], label="Conducciones progresivas")
    for i, val in enumerate(prog["PrgP"]):
        plt.text(prog["Season"].iloc[i], val+0.5, f"{val:.2f}", ha="center", color='#333333', fontsize=10, weight="bold")
    for i, val in enumerate(prog["PrgC"]):
        plt.text(prog["Season"].iloc[i], val-1, f"{val:.2f}", ha="center", color='#333333', fontsize=10, weight="bold")
    plt.title(f"Juego progresivo promedio por temporada - {squad}")
    plt.xlabel("Temporada")
    plt.ylabel("Acciones por jugador")
    plt.legend(loc='best')
    sns.despine()
    plt.show()

    #Variación defensiva por sector del campo

    if {"Att 3rd","Mid 3rd","Def 3rd"}.issubset(df.columns):
        press = df.groupby("Season").agg({"Att 3rd":"mean","Mid 3rd":"mean","Def 3rd":"mean"}).reset_index()
        ax = press.plot(
            x="Season", y=["Att 3rd","Mid 3rd","Def 3rd"], kind="bar",
            figsize=(10,6), color=[LALIGA_COLORS["1"], LALIGA_COLORS["2"], LALIGA_COLORS["3"]],
            title=f"Entradas por sector del campo (promedio) - {squad}"
        )
        plt.ylabel("Promedio entradas")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, color="black", weight="bold")
        sns.despine()
        plt.legend(loc='upper left')
        plt.show()

    #Variación sobre las faltas realizadas y recibidas

    disc = df.groupby("Season").agg({"Fls":"sum","Fld":"sum"}).reset_index()
    plt.figure(figsize=(10,6))
    plt.plot(disc["Season"], disc["Fls"], marker="o", linewidth=3, color=LALIGA_COLORS["1"], label="Faltas cometidas")
    plt.plot(disc["Season"], disc["Fld"], marker="o", linewidth=3, color=LALIGA_COLORS["2"], label="Faltas recibidas")
    for i, val in enumerate(disc["Fls"]):
        plt.text(prog["Season"].iloc[i], val+15, f"{val:.2f}", ha="center", color='#333333', fontsize=10, weight="bold")
    for i, val in enumerate(disc["Fld"]):
        plt.text(prog["Season"].iloc[i], val-15, f"{val:.2f}", ha="center", color='#333333', fontsize=10, weight="bold")
    plt.title(f"Evolución disciplina: faltas cometidas vs recibidas - {squad}")
    plt.xlabel("Temporada")
    plt.ylabel("Total de faltas")
    plt.legend(loc='upper right')
    sns.despine()
    plt.show()

    print("✅ Análisis completo generado.")

global_stats_laliga(df, squad='')
global_stats_laliga(df, squad='Getafe')
global_stats_laliga(df, squad='Barcelona')