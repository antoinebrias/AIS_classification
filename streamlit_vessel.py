# Web app displaying vessel trajectories. 

import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas 
import matplotlib.pyplot as plt

from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta
from holoviews import opts, dim
from os.path import exists
from urllib.request import urlretrieve

import warnings
import plotly.express as px
import streamlit as st

import plotly.io as pio
pio.templates.default = "plotly"


st.set_page_config(layout="wide")

# Load data
def load_data():
    df = gpd.read_file('./data/ais.gpkg')
    return df

df = load_data()

# Sidebar for ship type filter
selected_ship_type = st.sidebar.selectbox('Select Ship Type', ['All'] + df['ShipType'].unique().tolist())

# Filter data by selected ship type
if selected_ship_type != 'All':
    filtered_df = df[df['ShipType'] == selected_ship_type]
else:
    filtered_df = df

filtered_df['lon'] = filtered_df.geometry.x
filtered_df['lat'] = filtered_df.geometry.y

# Create first figure
fig = px.line_mapbox(filtered_df, lat="lat", lon="lon", color="Name", zoom=3)
fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12, mapbox_center_lon = np.mean(df.geometry.x), mapbox_center_lat = np.mean(df.geometry.y))
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Create second figure
fig2 = px.density_mapbox(filtered_df, lat='lat', lon='lon', center=dict(lat=np.mean(df.geometry.y), lon=np.mean(df.geometry.x)), range_color=[0, 200], zoom=2, radius=15, opacity=0.5, mapbox_style='open-street-map')
fig2.update_layout(mapbox_style="open-street-map", mapbox_zoom=12, mapbox_center_lon = np.mean(df.geometry.x), mapbox_center_lat = np.mean(df.geometry.y))
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Display figures side by side
st.header('Ship Trajectories and Density Maps')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)
