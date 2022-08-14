# ============================
# 0. IMPORTS
# ============================
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import geopandas

# ============================
# 1. SETTINGS AND FUNCTIONS
# ============================


# Configura página
st.set_page_config(page_title='KC HOUSE ENTERPRISE',
                   page_icon=':shark:',
                   layout='wide',  # 'centered',
                   initial_sidebar_state='expanded',
                   menu_items={
                       'Get Help': "https://www.google.com/?client=safari",
                       'Report a bug': "https://g1.globo.com",
                       'About': "# Testando as mensagens do menu!"
                   }
                   )

# When you mark a function with Streamlit’s cache annotation, it tells Streamlit that whenever
# the function is called it should check three things:

# The name of the function
# The actual code that makes up the body of the function
# The input parameters that you called the function with
# If this is the first time Streamlit has seen those three items, with those exact values,
# and in that exact combination, it runs the function and stores the result in a local cache.
# Then, next time the function is called, if those three values have not changed Streamlit knows
# it can skip executing the function altogether. Instead, it just reads the output from the local
# cache and passes it on to the caller.


@st.cache(allow_output_mutation=True)
# get path of data
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
# get url of geofile
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

st.write('''
# Testando o Markdown
''')
# get data
path = 'datasets/kc_house_data.csv'
data = get_data(path)

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# add column price/sqft2
data['price_m2'] = data['price']/data['sqft_lot']

# ============================
# 2. DATA OVERVIEW
# ============================

# Select one or more columns
f_attributes = st.sidebar.multiselect(label='Enter columns',
                                      options=data.columns.sort_values())

# Select one or more zipcodes
f_zipcode = st.sidebar.multiselect(label='Enter zipcodes',
                                   options=data.zipcode.sort_values().unique())

# Obs: No need to do st.write(f_attributes)

st.title('Data Overview')

# Condições para se ter uma melhor experiência na construção do dataframe
# filtrado pelo usuário
if (f_zipcode != []) & (f_attributes != []):
    data = data.loc[data.zipcode.isin(
        f_zipcode), f_attributes]
elif (f_zipcode != []) & (f_attributes == []):
    data = data.loc[data.zipcode.isin(f_zipcode), :]
elif (f_zipcode == []) & (f_attributes != []):
    data = data.loc[:, f_attributes]
else:
    data = data.copy()

st.dataframe(data)
st.write('Número de imóveis selecionados: {}'.format(len(data)))

# Creating columns
c1, c2 = st.columns((1, 1))

# Average Metrics
df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

# Merging dataframes
m1 = pd.merge(df1, df2, on='zipcode', how="inner")
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
df = pd.merge(m2, df4, on='zipcode', how='inner')

df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

c1.header('AVERAGE VALUES')
c1.dataframe(df, height=600)

# Statistic Description
num_attributes = data.select_dtypes(include=['int64', 'float64'])

# central tendency - media, mediana
media = np.mean(num_attributes)
# Da forma abaixo a função gera apenas um número, não um dataframe
# mediana = np.median(num_attributes)
mediana = num_attributes.apply(np.median)

# st.write(media)
# st.write(mediana)

# dispersion - desvio-padrão, mínimo, máximo
std = np.std(num_attributes)
min_ = np.min(num_attributes)
max_ = np.max(num_attributes)

# st.write(std)
# st.write(min_)
# st.write(max_)

# Concatenate columns
df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
c2.header('Descriptive Analysis')
c2.dataframe(df1, height=600)

# ======================================
# Densidade de Portfolio
# ======================================

st.title('Region Overview')

# Space out the maps so the first one is 2x the size of the other three
# c1, c2, c3, c4 = st.columns((2, 1, 1, 1))
c1, c2 = st.columns((1, 1))
c1.header('Portfolio Density')

df = data.sample(100).copy()
df['date_ts'] = pd.to_datetime(df['date']).dt.strftime('%d/%m/%Y')

# Base Map - Folium
density_map = folium.Map(
    location=[data.lat.mean(), data.long.mean()], default_zoom_start=15)

# Provides Beautiful Animated Marker Clustering functionality for maps.
marker_cluster = MarkerCluster().add_to(density_map)

for row in df.itertuples():
    folium.Marker([row.lat, row.long],
                  popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row.price, row.date_ts, row.sqft_living, row.bedrooms, row.bathrooms, row.yr_built)).add_to(marker_cluster)

with c1:
    folium_static(density_map)

# Region Price Map
c2.header('Price Density')

# df = data.sample(10).copy()
df = data

df = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
df.columns = ['ZIP', 'PRICE_M2']

# Filtrar o geofile só com os zips que tenho no dataset
geofile = geofile.loc[geofile.ZIP.isin(df.ZIP.tolist())]

region_price_map = folium.Map(
    location=[data.lat.mean(), data.long.mean()], default_zoom_start=15)

# Apply a GeoJSON overlay to the map.
# If data is passed as a Pandas DataFrame, the “columns” and “key-on” keywords must be included,
# the first to indicate which DataFrame columns to use, the second to indicate the layer in the
# GeoJSON on which to key the data. The ‘columns’ keyword does not need to be passed for a Pandas series.
region_price_map.choropleth(data=df,
                            geo_data=geofile,
                            columns=['ZIP', 'PRICE_M2'],
                            key_on='feature.properties.ZIP',
                            fill_color='YlOrRd',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name='AVG PRICE/M2')

with c2:
    folium_static(region_price_map)
