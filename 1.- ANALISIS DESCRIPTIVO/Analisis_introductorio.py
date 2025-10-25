import pandas as pd
import plotly.express as px

df = pd.read_excel("C:\\Users\\marce\\Desktop\\PROYECTO LA LIGA\\OFICIAL\\LA_LIGA_DATOS_JUG_OFICIAL.xlsx")
df.head()
df.columns

#Se realiza un primer analisis preliminar para conocer la evolución de numero de jugadores por posición ç
#correspondientes a las temporadas comprendidas desde 2020-2021 a 2024-2025.

df['Pos'] = df['Pos'].str.strip()
df_expand = df.assign(Pos_jug = df['Pos'].str.split(',\s*')).explode('Pos_jug')
df_expand['Pos'] = df_expand['Pos'].str.strip()
df_expand

conteo_pos = df_expand.groupby(['Season', 'Pos_jug'])['Player'].nunique().reset_index()
conteo_pos.rename(columns={'Player' : 'num_jug_pos'}, inplace=True)
print(conteo_pos)

colores = {
    'DF' : '#003DA5',
    'MF' : "#00D620",
    'FW' : '#E30613',
    'GK' : '#7F3F98'
}

fig_barras = px.bar(conteo_pos,
                    x = 'Season',
                    y = 'num_jug_pos',
                    color = 'Pos_jug',
                    title= 'Distribución de jugadores por posición y temporada',
                    color_discrete_map= colores,
                     text='num_jug_pos')

fig_barras.update_layout(
    template='plotly_white',
    title_font_size=20,
    legend_title_text='Posición',
    plot_bgcolor='white'
)

fig_barras.update_traces(textposition = 'outside')

fig_lineas = px.line(
    conteo_pos,
    x = 'Season',
    y = 'num_jug_pos',
    color = 'Pos_jug',
    title='Evolución del numero de jugadores por posición',
    markers=True,
    color_discrete_map=colores
)

fig_lineas.update_layout(
    template='plotly_white',
    title_font_size=20,
    legend_title_text='Posición',
    plot_bgcolor='white'
)

def reorder_legend(fig, orden):
    trace_dict = {trace.name: trace for trace in fig.data}
    fig.data = ()
    for pos in orden:
        if pos in trace_dict:
            fig.add_trace(trace_dict[pos])

orden_posiciones = ['GK', 'DF', 'MF', 'FW']

reorder_legend(fig_barras, orden_posiciones)
reorder_legend(fig_lineas, orden_posiciones)

fig_barras.show()
fig_lineas.show()
