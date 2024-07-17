import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import osmnx as ox

# Read the CSV files
df_substations = pd.read_csv('datasets/pa_substations.csv')
df_renewable = pd.read_csv('datasets/irena_roughly_pa_renewable_power_plants.csv')

# Filter for Pennsylvania substations and known MAX_VOLT
df_pa_sub = df_substations[(df_substations['STATE'] == 'PA') & (df_substations['MAX_VOLT'] != -999999)]

# Filter for Pennsylvania renewable energy sites
df_pa_renewable = df_renewable[df_renewable['country'] == 'United States of America']
df_pa_renewable = df_pa_renewable[(df_pa_renewable['latitude'] >= 39.7) & (df_pa_renewable['latitude'] <= 42.5) &
                                  (df_pa_renewable['longitude'] >= -80.5) & (df_pa_renewable['longitude'] <= -74.7)]

# Create a map of Pennsylvania
fig, ax = plt.subplots(figsize=(16, 10))
m = Basemap(projection='lcc', resolution='i',
            lat_0=40.9699, lon_0=-77.7278,
            width=5e5, height=3e5, ax=ax)

m.shadedrelief()
m.drawcountries(color='black')
m.drawstates(color='black')

# Get major highways for Pennsylvania
highways = ox.geometries_from_place("Pennsylvania, USA", tags={"highway": ["motorway", "trunk", "primary"]})
highways = highways.to_crs(epsg=4326)  # Convert to WGS84

# Plot highways
for _, row in highways.iterrows():
    if row.geometry.geom_type == 'LineString':
        x, y = m(row.geometry.xy[0], row.geometry.xy[1])
        m.plot(x, y, 'gray', linewidth=0.5, alpha=0.7)
    elif row.geometry.geom_type == 'MultiLineString':
        for line in row.geometry:
            x, y = m(line.xy[0], line.xy[1])
            m.plot(x, y, 'gray', linewidth=0.5, alpha=0.7)

# Plot substations
x_sub, y_sub = m(df_pa_sub['LONGITUDE'].values.tolist(), df_pa_sub['LATITUDE'].values.tolist())
scatter_sub = m.scatter(x_sub, y_sub, s=20, c=df_pa_sub['MAX_VOLT'], cmap='viridis', 
                        norm=plt.Normalize(vmin=df_pa_sub['MAX_VOLT'].min(), vmax=df_pa_sub['MAX_VOLT'].max()),
                        edgecolor='black', linewidth=0.5, marker='s', label='Substations')

# Plot renewable energy sites
renewable_types = df_pa_renewable['primary_fu'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(renewable_types)))
for renewable_type, color in zip(renewable_types, colors):
    df_type = df_pa_renewable[df_pa_renewable['primary_fu'] == renewable_type]
    x_ren, y_ren = m(df_type['longitude'].values.tolist(), df_type['latitude'].values.tolist())
    m.scatter(x_ren, y_ren, s=50, c=[color], edgecolor='black', linewidth=0.5, marker='^', label=f'{renewable_type} Energy')

# Add major cities
cities = {
    'Philadelphia': (-75.1652, 39.9526),
    'Pittsburgh': (-79.9959, 40.4406),
    'Allentown': (-75.4714, 40.6084),
    'Erie': (-80.0852, 42.1292),
    'Reading': (-75.9269, 40.3356),
    'Scranton': (-75.6624, 41.4090),
    'Bethlehem': (-75.3705, 40.6259),
    'Lancaster': (-76.3055, 40.0379),
    'Harrisburg': (-76.8867, 40.2732),
    'Altoona': (-78.3947, 40.5186)
}

for city, coords in cities.items():
    x, y = m(coords[0], coords[1])
    plt.plot(x, y, 'ko', markersize=5)
    plt.text(x, y, city, fontsize=8, ha='right', va='bottom')

plt.title('Substations, Renewable Energy Sites in Pennsylvania')
cbar = plt.colorbar(scatter_sub, label='Substation MAX_VOLT')

# Create a legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to prevent clipping
plt.tight_layout()

plt.show()