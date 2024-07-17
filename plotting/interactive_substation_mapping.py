import plotly.graph_objects as go
import requests
import pandas as pd

# Download GeoJSON data for Pennsylvania
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties = requests.get(url).json()

# Filter for only Pennsylvania counties (FIPS codes starting with 42)
pa_counties = {
    "type": "FeatureCollection",
    "features": [county for county in counties["features"] if county["properties"]["STATE"] == "42"]
}

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

# Add substations
fig.add_trace(go.Scattermapbox(
    lat=df_pa_sub['LATITUDE'],
    lon=df_pa_sub['LONGITUDE'],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        opacity=0.7
    ),
    text=df_pa_sub['NAME'],
    hoverinfo='text',
    name='Substations'
))

# Add renewable energy sites
fig.add_trace(go.Scattermapbox(
    lat=df_pa_renewable['latitude'],
    lon=df_pa_renewable['longitude'],
    mode='markers',
    marker=dict(
        size=5,
        color='green',
        opacity=0.7
    ),
    text=df_pa_renewable['name'] + '<br>' + df_pa_renewable['primary_fu'],
    hoverinfo='text',
    name='Renewable Energy Sites'
))

# Update the layout to focus on Pennsylvania
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox=dict(
        center=dict(lat=40.9699, lon=-77.7278),  # Center of Pennsylvania
        zoom=6.5
    ),
    margin={"r":0,"t":30,"l":0,"b":0},
    autosize=True,
    height=800,
    title_text="Energy Infrastructure in Pennsylvania",
    title_x=0.5,
    legend_title_text='Legend'
)

# Show the map
fig.show(config={'responsive': True})