import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pulp import *
import geopandas as gpd
import json
from geodatasets import get_path
import requests

# Download GeoJSON data for Pennsylvania
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties_geojson = requests.get(url).json()

# Read the CSV files
df_substations = pd.read_csv('datasets/pa_substations.csv')
df_renewable = pd.read_csv('datasets/irena_roughly_pa_renewable_power_plants.csv')

# Filter for Pennsylvania substations
df_pa_sub = df_substations[df_substations['STATE'] == 'PA']

# Create a GeoDataFrame from all counties
counties_gdf = gpd.GeoDataFrame.from_features(counties_geojson['features'])

# Set CRS for counties_gdf (assuming it's in WGS84)
counties_gdf = counties_gdf.set_crs("EPSG:4326")

# Filter for Pennsylvania counties (FIPS codes starting with 42)
pa_counties = counties_gdf[counties_gdf['STATE'] == '42']

# Dissolve county boundaries to create state boundary
pa_state = pa_counties.dissolve(by='STATE')

# Convert renewable energy sites to GeoDataFrame
gdf_renewable = gpd.GeoDataFrame(
    df_renewable, 
    geometry=gpd.points_from_xy(df_renewable['longitude'], df_renewable['latitude']),
    crs="EPSG:4326"
)

# Perform spatial join to get only the points within Pennsylvania
df_pa_renewable = gpd.sjoin(gdf_renewable, pa_state, how="inner", predicate="within")

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
# Increase substation capacity even more (e.g., to 5000 MW or higher if needed)
substation_capacity = 1000
for j in substations_indices:
    prob += lpSum([x[i,j,k] * renewable.iloc[i]['capacity_m'] 
                   for i in renewable_indices 
                   for k in cable_type_indices]) <= substation_capacity

# Allow partial connections
for i in renewable_indices:
    prob += lpSum([x[i,j,k] for j in substations_indices for k in cable_type_indices]) <= 1

# Relax cable capacity constraints
for i in renewable_indices:
    for j in substations_indices:
        for k in cable_type_indices:
            prob += x[i,j,k] * renewable.iloc[i]['capacity_m'] <= list(cable_types.values())[k]['capacity_m'] * 10  # Multiply by 10 to relax

# Add a slack variable to allow for unconnected capacity
slack = LpVariable.dicts("slack", renewable_indices, lowBound=0)
for i in renewable_indices:
    prob += lpSum([x[i,j,k] for j in substations_indices for k in cable_type_indices]) + slack[i] == 1

# Update objective function to penalize unconnected capacity
big_M = 1e6  # A large number
prob += lpSum([x[i,j,k] * distances[i,j] * list(cable_types.values())[k]['cost_per_km'] 
               for i in renewable_indices 
               for j in substations_indices 
               for k in cable_type_indices]) + \
        lpSum([slack[i] * renewable.iloc[i]['capacity_m'] * big_M for i in renewable_indices])

# After defining the problem but before solving
print(f"Number of renewable sources: {num_renewable}")
print(f"Number of substations: {num_substations}")
print(f"Total renewable capacity: {renewable['capacity_m'].sum()} MW")
print(f"Maximum substation capacity: 2000 MW")
print(f"Largest renewable source: {renewable['capacity_m'].max()} MW")
print(f"Smallest cable capacity: {min(cable['capacity_m'] for cable in cable_types.values())} MW")

# Solve the problem
prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))  # Set a time limit of 300 seconds

# Check the solution status
print("Status:", LpStatus[prob.status])

# Extract results
if prob.status == 1:  # Optimal solution found
    connections = []
    connected_substation_indices = set()
    total_connected_capacity = 0
    total_unconnected_capacity = 0
    for i in renewable_indices:
        connected = False
        for j in substations_indices:
            for k in cable_type_indices:
                if value(x[i,j,k]) > 0.01:  # Use a small threshold
                    connections.append((i, j, k))
                    connected_substation_indices.add(j)
                    connected = True
                    total_connected_capacity += renewable.iloc[i]['capacity_m'] * value(x[i,j,k])
        if not connected:
            total_unconnected_capacity += renewable.iloc[i]['capacity_m']

    print(f"Total connected capacity: {total_connected_capacity:.2f} MW")
    print(f"Total unconnected capacity: {total_unconnected_capacity:.2f} MW")
    print(f"Number of connections made: {len(connections)}")
    print(f"Number of connected substations: {len(connected_substation_indices)}")

    # Calculate actual connection cost
    total_cost = sum(value(x[i,j,k]) * distances[i,j] * list(cable_types.values())[k]['cost_per_km'] 
                     for i in renewable_indices 
                     for j in substations_indices 
                     for k in cable_type_indices)
    print(f"Actual connection cost: ${total_cost:,.2f}")
else:
    print("No optimal solution found.")
    connections = []
    connected_substation_indices = set()

# Filter for only Pennsylvania counties (FIPS codes starting with 42)
pa_counties = {
    "type": "FeatureCollection",
    "features": [county for county in counties_geojson["features"] if county["properties"]["STATE"] == "42"]
}

# Create the map
fig = go.Figure()

# Add Pennsylvania counties
fig.add_trace(go.Choroplethmapbox(
    geojson=pa_counties,
    locations=[county["properties"]["GEO_ID"].split("US")[1] for county in pa_counties["features"]],
    z=[1] * len(pa_counties["features"]),  # Dummy data to make all counties the same color
    colorscale=[[0, "rgb(220, 220, 220)"], [1, "rgb(220, 220, 220)"]],  # Light grey color
    marker_opacity=0.7,
    marker_line_width=0.5,
    showscale=False
))

# Identify connected substations
connected_substations = set(j for (i, j, k) in connections)

# Create lists for connected and unconnected substations
connected_lats = []
connected_lons = []
connected_names = []
unconnected_lats = []
unconnected_lons = []
unconnected_names = []

for idx, substation in substations.iterrows():
    if idx in connected_substation_indices:
        connected_lats.append(substation['latitude'])
        connected_lons.append(substation['longitude'])
        connected_names.append(substation['name'])
    else:
        unconnected_lats.append(substation['latitude'])
        unconnected_lons.append(substation['longitude'])
        unconnected_names.append(substation['name'])

print(f"Number of connected substations: {len(connected_lats)}")
print(f"Number of unconnected substations: {len(unconnected_lats)}")

# Add connected substations
if connected_lats:
    fig.add_trace(go.Scattermapbox(
        lat=connected_lats,
        lon=connected_lons,
        mode='markers',
        marker=dict(size=10, color='red'),
        text=connected_names,
        hoverinfo='text',
        name='Connected Substations'
    ))

# Add unconnected substations
if unconnected_lats:
    fig.add_trace(go.Scattermapbox(
        lat=unconnected_lats,
        lon=unconnected_lons,
        mode='markers',
        marker=dict(size=10, color='grey'),  # Light red color
        text=unconnected_names,
        hoverinfo='text',
        name='Unconnected Substations'
    ))

# Print number of connected and unconnected substations for debugging
print(f"Number of connected substations: {len(connected_lats)}")
print(f"Number of unconnected substations: {len(unconnected_lats)}")
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