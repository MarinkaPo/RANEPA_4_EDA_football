import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('Статистика игроков NFL')

# image = Image.open('NFL_logo.png')
# st.image(image, width=50)

st.markdown("""
Простой и понятный анализ статистики игроков NFL (с акцентом на скорость)!
* **Библиотеки Python:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Источник:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

st.sidebar.header('Выбор настроек')
selected_year = st.sidebar.selectbox('Год', list(reversed(range(1990,2020))))

# Смотрим
# https://www.pro-football-reference.com/years/2019/rushing.htm
@st.cache
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header = 1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

# Боковая панель - Выбор команды
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Команда', sorted_unique_team, sorted_unique_team)

# Боковая панель - Выбор позиции игрока
unique_pos = ['RB','QB','WR','FB','TE']
selected_pos = st.sidebar.multiselect('Позиция игрока', unique_pos, unique_pos)

# Фильтры
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
df_selected_team = df_selected_team.astype(str)

st.header('Обзор игроков команд(ы)')
st.write('Размер таблицы: ' + str(df_selected_team.shape[0]) + ' строк и ' + str(df_selected_team.shape[1]) + ' столбцов.')
st.dataframe(df_selected_team)

# Загружаем данные.
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Загрузить CSV-файл</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Посмотреть корреляции'):
    st.header('Матрица корреляций - Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(14, 10))
        ax = sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, 
        cmap= 'Spectral', fmt='.1g', linewidths=1, linecolor='black') # mask=mask, vmax=1, square=True
    st.pyplot(f)
