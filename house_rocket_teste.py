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
# # get data
# path = 'datasets/kc_house_data.csv'
# data = get_data(path)

# # get geofile
# url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
# geofile = get_geofile(url)

# # add column price/sqft2
# data['price_m2'] = data['price']/data['sqft_lot']

# # ============================
# # 2. DATA OVERVIEW
# # ============================

# # Select one or more columns
# f_attributes = st.sidebar.multiselect(label='Enter columns',
#                                       options=data.columns.sort_values())

# # Select one or more zipcodes
# f_zipcode = st.sidebar.multiselect(label='Enter zipcodes',
#                                    options=data.zipcode.sort_values().unique())

# # Obs: No need to do st.write(f_attributes)

# st.title('Data Overview')

# # Condições para se ter uma melhor experiência na construção do dataframe
# # filtrado pelo usuário
# if (f_zipcode != []) & (f_attributes != []):
#     data = data.loc[data.zipcode.isin(
#         f_zipcode), f_attributes]
# elif (f_zipcode != []) & (f_attributes == []):
#     data = data.loc[data.zipcode.isin(f_zipcode), :]
# elif (f_zipcode == []) & (f_attributes != []):
#     data = data.loc[:, f_attributes]
# else:
#     data = data.copy()

# st.dataframe(data)
# st.write('Número de imóveis selecionados: {}'.format(len(data)))

# # Creating columns
# c1, c2 = st.columns((1, 1))

# # Average Metrics
# df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
# df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
# df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
# df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

# # Merging dataframes
# m1 = pd.merge(df1, df2, on='zipcode', how="inner")
# m2 = pd.merge(m1, df3, on='zipcode', how='inner')
# df = pd.merge(m2, df4, on='zipcode', how='inner')

# df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

# c1.header('AVERAGE VALUES')
# c1.dataframe(df, height=600)

# # Statistic Description
# num_attributes = data.select_dtypes(include=['int64', 'float64'])

# # central tendency - media, mediana
# media = np.mean(num_attributes)
# # Da forma abaixo a função gera apenas um número, não um dataframe
# # mediana = np.median(num_attributes)
# mediana = num_attributes.apply(np.median)

# # st.write(media)
# # st.write(mediana)

# # dispersion - desvio-padrão, mínimo, máximo
# std = np.std(num_attributes)
# min_ = np.min(num_attributes)
# max_ = np.max(num_attributes)

# # st.write(std)
# # st.write(min_)
# # st.write(max_)

# # Concatenate columns
# df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

# df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
# c2.header('Descriptive Analysis')
# c2.dataframe(df1, height=600)

# # ======================================
# # Densidade de Portfolio
# # ======================================

# st.title('Region Overview')

# # Space out the maps so the first one is 2x the size of the other three
# # c1, c2, c3, c4 = st.columns((2, 1, 1, 1))
# c1, c2 = st.columns((1, 1))
# c1.header('Portfolio Density')

# df = data.sample(100).copy()
# df['date_ts'] = pd.to_datetime(df['date']).dt.strftime('%d/%m/%Y')

# # Base Map - Folium
# density_map = folium.Map(
#     location=[data.lat.mean(), data.long.mean()], default_zoom_start=15)

# # Provides Beautiful Animated Marker Clustering functionality for maps.
# marker_cluster = MarkerCluster().add_to(density_map)

# for row in df.itertuples():
#     folium.Marker([row.lat, row.long],
#                   popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row.price, row.date_ts, row.sqft_living, row.bedrooms, row.bathrooms, row.yr_built)).add_to(marker_cluster)

# with c1:
#     folium_static(density_map)

# # Region Price Map
# c2.header('Price Density')

# # df = data.sample(10).copy()
# df = data

# df = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
# df.columns = ['ZIP', 'PRICE_M2']

# # Filtrar o geofile só com os zips que tenho no dataset
# geofile = geofile.loc[geofile.ZIP.isin(df.ZIP.tolist())]

# region_price_map = folium.Map(
#     location=[data.lat.mean(), data.long.mean()], default_zoom_start=15)

# # Apply a GeoJSON overlay to the map.
# # If data is passed as a Pandas DataFrame, the “columns” and “key-on” keywords must be included,
# # the first to indicate which DataFrame columns to use, the second to indicate the layer in the
# # GeoJSON on which to key the data. The ‘columns’ keyword does not need to be passed for a Pandas series.
# region_price_map.choropleth(data=df,
#                             geo_data=geofile,
#                             columns=['ZIP', 'PRICE_M2'],
#                             key_on='feature.properties.ZIP',
#                             fill_color='YlOrRd',
#                             fill_opacity=0.7,
#                             line_opacity=0.2,
#                             legend_name='AVG PRICE/M2')

# with c2:
#     folium_static(region_price_map)


# ============================
# 0. PREDICTION
# ============================

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import folium
from folium.plugins import HeatMap
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


st.write(''' 
# PREDICTIONS
''')
evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})

df = pd.read_csv('datasets/kc_house_data.csv')
#df.describe()
#df.info()
#df.head()
st.dataframe(df.head())

def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)

#%%capture
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)

lr = linear_model.LinearRegression()
X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['price'], dtype=pd.Series)
lr.fit(X_train,y_train)

X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['price'], dtype=pd.Series)

pred = lr.predict(X_test)
rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rtrsm = float(format(lr.score(X_train, y_train),'.3f'))
rtesm = float(format(lr.score(X_test, y_test),'.3f'))
cv = float(format(cross_val_score(lr,df[['sqft_living']],df['price'],cv=5).mean(),'.3f'))

print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
print('Intercept: {}'.format(lr.intercept_))
print('Coefficient: {}'.format(lr.coef_))

r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]
evaluation

st.write('''## Let's Show the Result''')

# sns.set(style="white", font_scale=1)

# plt.figure(figsize=(6.5,5))
# plt.figure(figsize=(10,4))

# plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)
# plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")
# plt.xlabel("Living Space (sqft)", fontsize=15)
# plt.ylabel("Price ($)", fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.legend()

# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)

# st.pyplot(plt.gcf())

# # TESTE COM DATASET DO PINGUIM
# # fig, ax = plt.subplots()

# # sns.scatterplot(
# #     data = sns.load_dataset("penguins"),
# #     x = "flipper_length_mm",
# #     y = "bill_length_mm",
# #     ax = ax
# # )

# # st.pyplot(fig)

# st.write('''## Visualizing and Examining Data''')
# f, axes = plt.subplots(1, 2,figsize=(15,5))
# sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])
# sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])
# sns.despine(left=True, bottom=True)
# axes[0].set(xlabel='Bedrooms', ylabel='Price')
# axes[0].yaxis.tick_left()
# axes[1].yaxis.set_label_position("right")
# axes[1].yaxis.tick_right()
# axes[1].set(xlabel='Floors', ylabel='Price')

# st.pyplot(f)

# f, axe = plt.subplots(1, 1,figsize=(12.18,5))
# sns.despine(left=True, bottom=True)
# sns.boxplot(x=df['bathrooms'],y=df['price'], ax=axe)
# axe.yaxis.tick_left()
# axe.set(xlabel='Bathrooms / Bedrooms', ylabel='Price');

# st.pyplot(f)

# st.write('''In this dataset, we have latitude and longtitude information for the houses. By using *lat* and *long* columns, I displayed the below heat map which is very useful for the people who does not know Seattle well. Also, if you select a spesific zip code, you may just see the heat map of this zip code's neighborhood.''')

# # find the row of the house which has the highest price
# maxpr=df.loc[df['price'].idxmax()]

# # define a function to draw a basemap easily
# def generateBaseMap(default_location=[47.5112, -122.257], default_zoom_start=9.4):
#     base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
#     return base_map

# df_copy = df.copy()
# # select a zipcode for the heatmap
# #set(df['zipcode'])
# #df_copy = df[df['zipcode']==98001].copy()
# df_copy['count'] = 1
# basemap = generateBaseMap()
# # add carton position map
# folium.TileLayer('cartodbpositron').add_to(basemap)
# s=folium.FeatureGroup(name='icon').add_to(basemap)
# # add a marker for the house which has the highest price
# folium.Marker([maxpr['lat'], maxpr['long']],popup='Highest Price: $'+str(format(maxpr['price'],'.0f')),
#               icon=folium.Icon(color='green')).add_to(s)
# # add heatmap
# HeatMap(data=df_copy[['lat','long','count']].groupby(['lat','long']).sum().reset_index().values.tolist(),
#         radius=8,max_zoom=13,name='Heat Map').add_to(basemap)
# folium.LayerControl(collapsed=False).add_to(basemap)
# folium_static(basemap)


st.write('''
# Data Preprocessing

A preprocessing on data might improve the model accuracy and make the model more reliable. It does not always have to improve our results but when we are conscious of the features and use a proper input, we might reach some outcomes easier. I tried various data mining techniques like transformation or normalization but in the end, decided to just use binning and created a new dataframe called ***df_dm***.

## Binning
Data binning is a preprocessing technique used to reduce the effects of minor observation errors. I think it is worthwhile applying to some columns of this dataset. I applied binning to *yr_built* and *yr_renovated*. I added the ages and renovation ages of the houses when they were sold. Also, I partitioned these columns to intervals and you can observe this in the below **histograms**. 
''')

df_dm=df.copy()

# just take the year from the date column
df_dm['sales_yr']=df_dm['date'].astype(str).str[:4]

# add the age of the buildings when the houses were sold as a new column
df_dm['age']=df_dm['sales_yr'].astype(int)-df_dm['yr_built']
# add the age of the renovation when the houses were sold as a new column
df_dm['age_rnv']=0
df_dm['age_rnv']=df_dm['sales_yr'][df_dm['yr_renovated']!=0].astype(int)-df_dm['yr_renovated'][df_dm['yr_renovated']!=0]
df_dm['age_rnv'][df_dm['age_rnv'].isnull()]=0

# partition the age into bins
bins = [-2,0,5,10,25,50,75,100,100000]
labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']
df_dm['age_binned'] = pd.cut(df_dm['age'], bins=bins, labels=labels)
# partition the age_rnv into bins
bins = [-2,0,5,10,25,50,75,100000]
labels = ['<1','1-5','6-10','11-25','26-50','51-75','>75']
df_dm['age_rnv_binned'] = pd.cut(df_dm['age_rnv'], bins=bins, labels=labels)

# histograms for the binned columns
f, axes = plt.subplots(1, 2,figsize=(15,5))
p1=sns.countplot(df_dm['age_binned'],ax=axes[0])
for p in p1.patches:
    height = p.get_height()
    p1.text(p.get_x()+p.get_width()/2,height + 50,height,ha="center")   
p2=sns.countplot(df_dm['age_rnv_binned'],ax=axes[1])
sns.despine(left=True, bottom=True)
for p in p2.patches:
    height = p.get_height()
    p2.text(p.get_x()+p.get_width()/2,height + 200,height,ha="center")
    
axes[0].set(xlabel='Age')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Renovation Age');

st.pyplot(f)

# transform the factor values to be able to use in the model
df_dm = pd.get_dummies(df_dm, columns=['age_binned','age_rnv_binned'])


st.write(''' ##  Multiple Regression - 1 
É possível ordenar a tabela clicando em cima da feature desejada''')

train_data_dm,test_data_dm = train_test_split(df_dm,train_size = 0.8,random_state=3)

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
complex_model_1 = linear_model.LinearRegression()
complex_model_1.fit(train_data_dm[features],train_data_dm['price'])

print('Intercept: {}'.format(complex_model_1.intercept_))
print('Coefficients: {}'.format(complex_model_1.coef_))

pred = complex_model_1.predict(test_data_dm[features])
rmsecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['price'],pred)),'.3f'))
rtrcm = float(format(complex_model_1.score(train_data_dm[features],train_data_dm['price']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_1.score(train_data_dm[features],train_data_dm['price']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm = float(format(complex_model_1.score(test_data_dm[features],test_data_dm['price']),'.3f'))
artecm = float(format(adjustedR2(complex_model_1.score(test_data_dm[features],test_data['price']),test_data_dm.shape[0],len(features)),'.3f'))
cv = float(format(cross_val_score(complex_model_1,df_dm[features],df_dm['price'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression-1','selected features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]
#st.dataframe(evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False))
evaluation



st.write('''
## Multiple Regression - 4
This time I used the data obtained after preprocessing step. 
''')


features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','age_binned_<1', 
            'age_binned_1-5', 'age_binned_6-10','age_binned_11-25', 'age_binned_26-50',
            'age_binned_51-75','age_binned_76-100', 'age_binned_>100','age_rnv_binned_<1',
            'age_rnv_binned_1-5', 'age_rnv_binned_6-10', 'age_rnv_binned_11-25',
            'age_rnv_binned_26-50', 'age_rnv_binned_51-75', 'age_rnv_binned_>75',
            'zipcode','lat','long','sqft_living15','sqft_lot15']
complex_model_4 = linear_model.LinearRegression()
complex_model_4.fit(train_data_dm[features],train_data_dm['price'])

print('Intercept: {}'.format(complex_model_4.intercept_))
print('Coefficients: {}'.format(complex_model_4.coef_))

pred = complex_model_4.predict(test_data_dm[features])
rmsecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['price'],pred)),'.3f'))
rtrcm = float(format(complex_model_4.score(train_data_dm[features],train_data_dm['price']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_4.score(train_data_dm[features],train_data_dm['price']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm = float(format(complex_model_4.score(test_data_dm[features],test_data_dm['price']),'.3f'))
artecm = float(format(adjustedR2(complex_model_4.score(test_data_dm[features],test_data_dm['price']),test_data_dm.shape[0],len(features)),'.3f'))
cv = float(format(cross_val_score(complex_model_4,df_dm[features],df_dm['price'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression-4','all features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]
#evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)
evaluation