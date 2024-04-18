#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:19:29 2024

@author: antoine
"""

# pixelize vessel trajectories. output are save in categorical subfolders

import os
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import geopandas as gpd
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import movingpandas as mpd
from shapely.geometry import Point
from movingpandas import Trajectory, TrajectoryCollection
import matplotlib.pyplot as plt

# Load data
def load_data():
    df = gpd.read_file('./data/ais.gpkg')
    return df

df = load_data()
num_pixels = 128
output_dir = "./output"


# Step 1: Filter out categories with few samples
category_counts = df['ShipType'].value_counts()
valid_categories = category_counts[category_counts > 10].index.tolist()  # Adjust the threshold as needed
df_filtered = df[df['ShipType'].isin(valid_categories)]


df_filtered['lon'] = df_filtered.geometry.x
df_filtered['lat'] = df_filtered.geometry.y

# Compute minimum and maximum latitude
min_latitude = df_filtered['lat'].min()
max_latitude = df_filtered['lat'].max()

# Compute minimum and maximum longitude
min_longitude = df_filtered['lon'].min()
max_longitude = df_filtered['lon'].max()

# Print the results
print("Minimum latitude:", min_latitude)
print("Maximum latitude:", max_latitude)
print("Minimum longitude:", min_longitude)
print("Maximum longitude:", max_longitude)


# Step 2: Determine the median time frame
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'])

# Calculate duration of each trajectory for each 'Name'
df_filtered['Duration'] = df_filtered.groupby('Name')['Timestamp'].transform(lambda x: x.max() - x.min())

# Step 2: Find the median duration
segment_duration = df_filtered['Duration'].median()*0.5

# Step 3: Filter out trajectories with duration less than half the median
filtered_df = df_filtered[df_filtered['Duration'] >= segment_duration]



# Convert lon and lat to pixel coordinates
lon_pixels = np.linspace(min_longitude, max_longitude, num_pixels)
lat_pixels = np.linspace(min_latitude, max_latitude, num_pixels)



# Create a subfolder for each category if they don't exist
categories = filtered_df['ShipType'].unique()
for category in categories:
    category_folder = f"{output_dir}/{category}/"
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

df_grouped = filtered_df.groupby('Name')


for group_name, df_group in df_grouped:
    df_group['Timestamp'] = pd.to_datetime(df_group['Timestamp'])
    df_group.set_index('Timestamp', inplace=True)
    
    # Create GeoDataFrame from DataFrame
    geometry = [Point(xy) for xy in zip(df_group['lon'], df_group['lat'])]
    gdf = gpd.GeoDataFrame(df_group[['lon', 'lat']], geometry=geometry)
    
    # Create MovingPandas Trajectory
    traj = Trajectory(gdf,'Timestamp')
    
    n_iter = int(traj.get_duration().total_seconds()//segment_duration.total_seconds())
    
    for iter in range(n_iter):
        try:
            seg_traj = traj.get_segment_between(traj.get_start_time()+iter*segment_duration,traj.get_start_time()+(iter+1)*segment_duration)
    
            # Get the start time in Unix timestamp format (seconds since the epoch)
            start_time = seg_traj.get_start_time().timestamp()
            
            # Calculate the end time by adding the median duration (assuming median_duration is in seconds)
            end_time = start_time + segment_duration.total_seconds()
            
            # Generate timestamps between the start and end times
            timestamps = np.linspace(start_time, end_time, num=10000)
            
            # Convert the generated timestamps back to Timestamp objects
            timestamps = pd.to_datetime(timestamps, unit='s')
        
            #timestamps = np.linspace(df_group['Timestamp'].min(), df_group['Timestamp'].min()+median_duration, num=100)
            interpolated_positions = []
            for ts in timestamps:
                interpolated_position = traj.interpolate_position_at(ts) #nn_model.predict([[ts]])[0]
                interpolated_positions.append(interpolated_position)
        
            
            lon_indices = np.searchsorted(lon_pixels, [pos.x for pos in interpolated_positions])
            lat_indices = np.searchsorted(lat_pixels, [pos.y for pos in interpolated_positions])
        
        
            # Create and save the image
            image = np.zeros((num_pixels, num_pixels))
            for lon_idx, lat_idx in zip(lon_indices, lat_indices):
                if lon_idx < num_pixels and lat_idx < num_pixels:
                    image[lat_idx, lon_idx] = 1  # Color the pixel
        
            if image.sum()>10:
                plt.imshow(image, cmap='gray', origin='lower')
                plt.axis('off')  # Turn off axis
                # Save the plot in the respective category subfolder
                category_folder = f"{output_dir}/{df_group['ShipType'].iloc[0]}/"
                plt.savefig(f'{category_folder}trajectory_{group_name}_{iter}.png')
                plt.close()
        
        except ValueError: 
            print("not enough data")