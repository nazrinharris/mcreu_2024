import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pulp import *
import geopandas as gpd
import json
from geodatasets import get_path
import requests


# Read the CSV files
df_substations = pd.read_csv('datasets/pa_substations.csv')
df_renewable = pd.read_csv('datasets/irena_roughly_pa_renewable_power_plants.csv')

# Filter for Pennsylvania substations
df_pa_sub = df_substations[df_substations['STATE'] == 'PA']

# Filter for Pennsylvania renewable energy sites
df_pa_renewable = df_renewable[df_renewable['country'] == 'United States of America']
df_pa_renewable = df_pa_renewable[(df_pa_renewable['latitude'] >= 39.7) & 
                                  (df_pa_renewable['latitude'] <= 42.5) &
                                  (df_pa_renewable['longitude'] >= -80.5) & 
                                  (df_pa_renewable['longitude'] <= -74.7)]

# Lengths
num_substations = len(df_pa_sub)
num_renewable = len(df_pa_renewable)

# Pennsylvania bounding box
min_lat, max_lat = 39.7, 42.5
min_lon, max_lon = -80.5, -74.7

# Setup DataFrame
substations = df_pa_sub.head(num_substations)[['LATITUDE', 'LONGITUDE', 'NAME']].rename(columns={'LATITUDE': 'latitude', 'LONGITUDE': 'longitude', 'NAME': 'name'})
renewable = df_pa_renewable.head(num_renewable)[['latitude', 'longitude', 'name', 'capacity_m']]

# Define cable types
cable_types = {
    'small': {'capacity_m': 50, 'cost_per_km': 100000},
    'medium': {'capacity_m': 100, 'cost_per_km': 200000},
    'large': {'capacity_m': 200, 'cost_per_km': 300000}
}

# Create the problem
prob = LpProblem("Enhanced_Renewable_to_Substation_Connection", LpMinimize)

# Define indices
substations_indices = range(len(substations))
renewable_indices = range(len(renewable))
cable_type_indices = range(len(cable_types))

# Define variables
# x[i,j,k] = 1 if renewable i is connected to substation j with cable type k, 0 otherwise
x = LpVariable.dicts("connection", 
                     ((i, j, k) for i in renewable_indices 
                      for j in substations_indices 
                      for k in cable_type_indices), 
                     cat='Binary')

# Calculate distances
distances = {(i, j): np.sqrt((substations.iloc[j]['latitude'] - renewable.iloc[i]['latitude'])**2 + 
                             (substations.iloc[j]['longitude'] - renewable.iloc[i]['longitude'])**2)
             for i in renewable_indices for j in substations_indices}

# Objective function: Minimize total cost
prob += lpSum([x[i,j,k] * distances[i,j] * list(cable_types.values())[k]['cost_per_km'] 
               for i in renewable_indices 
               for j in substations_indices 
               for k in cable_type_indices])

# Constraints
# Each renewable source must connect to exactly one substation with one cable type
for i in renewable_indices:
    prob += lpSum([x[i,j,k] for j in substations_indices for k in cable_type_indices]) == 1

# Cable capacity_m constraints
for i in renewable_indices:
    for j in substations_indices:
        for k in cable_type_indices:
            prob += x[i,j,k] * renewable.iloc[i]['capacity_m'] <= list(cable_types.values())[k]['capacity_m']

# Substation capacity_m constraints (assuming each substation can handle 1000 MW)
for j in substations_indices:
    prob += lpSum([x[i,j,k] * renewable.iloc[i]['capacity_m'] 
                   for i in renewable_indices 
                   for k in cable_type_indices]) <= 1000

# Solve the problem
prob.solve()

# Extract results
connections = []
for i in renewable_indices:
    for j in substations_indices:
        for k in cable_type_indices:
            if value(x[i,j,k]) == 1:
                connections.append((i, j, k))

# Fetch Pennsylvania state boundary
pa_geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
pa_geojson = requests.get(pa_geojson_url).json()

# Filter for only Pennsylvania counties
pa_geojson['features'] = [f for f in pa_geojson['features'] if f['properties']['STATE'] == '42']

# Create the map
fig = go.Figure()

# Add Pennsylvania border
fig.add_trace(go.Choroplethmapbox(
    geojson=pa_geojson,
    locations=[f['properties']['GEO_ID'] for f in pa_geojson['features']],
    z=[0] * len(pa_geojson['features']),  # This doesn't matter for what we're doing
    featureidkey="properties.GEO_ID",
    marker=dict(opacity=0),
    showscale=False,
    hoverinfo='skip',
    marker_line_color='black',
    marker_line_width=1
))

# Add substations
fig.add_trace(go.Scattermapbox(
    lat=substations['latitude'],
    lon=substations['longitude'],
    mode='markers',
    marker=dict(size=10, color='red'),
    text=substations['name'],
    hoverinfo='text',
    name='Substations'
))

# Add renewable energy sites
fig.add_trace(go.Scattermapbox(
    lat=renewable['latitude'],
    lon=renewable['longitude'],
    mode='markers',
    marker=dict(size=10, color='green'),
    text=renewable['name'] + ' (' + renewable['capacity_m'].astype(str) + ' MW)',
    hoverinfo='text',
    name='Renewable Energy Sites'
))

# Add connections
cable_colors = ['blue', 'yellow', 'purple']  # Colors for different cable types
for i, j, k in connections:
    renewable_point = renewable.iloc[i]
    substation_point = substations.iloc[j]
    
    fig.add_trace(go.Scattermapbox(
        lat=[renewable_point['latitude'], substation_point['latitude']],
        lon=[renewable_point['longitude'], substation_point['longitude']],
        mode='lines',
        line=dict(width=2, color=cable_colors[k]),
        opacity=0.8,
        hoverinfo='none',
        showlegend=False
    ))

# Update the layout to focus on Pennsylvania
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox=dict(
        center=dict(lat=40.9699, lon=-77.7278),  # Center of Pennsylvania
        zoom=6.2  # Slightly zoomed out to show the whole state
    ),
    margin={"r":0,"t":30,"l":0,"b":0},
    height=800,
    title_text="Optimized Energy Connections in Pennsylvania",
    title_x=0.5,
    legend_title_text='Legend'
)

# Show the map
fig.show()

# Print total cost
total_cost = sum(value(x[i,j,k]) * distances[i,j] * list(cable_types.values())[k]['cost_per_km'] 
                 for i in renewable_indices 
                 for j in substations_indices 
                 for k in cable_type_indices)
print(f"Total connection cost: ${total_cost:,.2f}")