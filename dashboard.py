import hvplot.pandas
import numpy as np
import panel as pn
import pandas as pd
import folium
import holoviews as hv
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import pymap3d as pm
import geopandas as gpd
from shapely.geometry import LineString

from logparser import LogReader

# panel serve dashboard.py --show --autoreload

date_formatter = DatetimeTickFormatter(minutes='%H:%M')
plot_opts = dict(height=220, width=800)

color_map = {'GPS':'blue', 'FLP':'red', 'NLP':'green'}

# =============================================================================
# Data

filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
log = LogReader(filepath)

providers = list(set(log.fix["provider"]))
satelliteList = list(set(log.raw["prn"]))
satelliteList.sort()

# =============================================================================
# Map tab

# _polyline = []
# for prov in providers:
#     df = log.fix.loc[log.fix['provider'].isin([prov])]
#     df['latitude'].tolist()
#     _polyline.append(LineString(zip(df['longitude'].tolist(), df['latitude'].tolist())))
# fix_geo = gpd.GeoDataFrame(index=providers, geometry=_polyline, crs='epsg:4326')
# print(fix_geo['GPS'])
# map_pane = fix_geo.hvplot(frame_height=800, frame_width=800, tiles=True)
# map_pane.opts(active_tools=['wheel_zoom'])

m = folium.Map(location=[log.fix['latitude'][0],log.fix['longitude'][0]], zoom_start=16)

for prov in providers:
    points = []
    df = log.fix.loc[log.fix['provider'].isin([prov])]
    for index, row in df.iterrows():
        points.append((row['latitude'], row['longitude']))

    trajectory = folium.FeatureGroup(name=prov).add_to(m)
    trajectory.add_child(folium.PolyLine(locations=points, weight=3, color=color_map[prov]))
    #folium.PolyLine(points, color='blue').add_to(m)
folium.LayerControl().add_to(m)

map_pane = pn.pane.plot.Folium(m, height=500)

# =============================================================================
# Coordinates tab
xs = log.fix["timestamp"]

providersCheckButtons = pn.widgets.CheckButtonGroup(name='Providers', value=['GPS'], options=list(set(log.fix["provider"])), button_type='primary')
ticker_providers = pn.widgets.Select(options=providers, name='Ticker_providers')

def plotFix(providersCheckButtons):
    if providersCheckButtons:
        return log.fix.loc[log.fix['provider'].isin(providersCheckButtons)]
    else:
        return log.fix
dfi_trajectory = hvplot.bind(plotFix, providersCheckButtons).interactive()

plot_opts = dict(max_height=350, xformatter=date_formatter, xlabel="Local time", grid=True, responsive=True)
#geodeticPlot = dfi_trajectory.hvplot(x="timestamp", y=["latitude", "longitude", "altitude"], shared_axes=False, subplots=True, **plot_opts).cols(1).output()
coordinateGrid = pn.GridSpec(sizing_mode='stretch_both')
coordinateGrid[0:1, 0:1] = dfi_trajectory.hvplot(x="timestamp", y="latitude", by="provider", ylabel="Latitude [DD]", **plot_opts).output()
coordinateGrid[1:2, 0:1] = dfi_trajectory.hvplot(x="timestamp", y="longitude", by="provider", ylabel="Longitude [DD]", **plot_opts).output()
coordinateGrid[2:3, 0:1] = dfi_trajectory.hvplot(x="timestamp", y="altitude", by="provider", ylabel="Altitude [m]", **plot_opts).output()

mapGrid = pn.GridSpec(sizing_mode='stretch_both', responsive=True)
mapGrid[ 0:1, 0:1] = map_pane

# =============================================================================
# GNSS Measurement tab

meas_select = pn.widgets.Select(name='Select', options=['Cn0DbHz', 'CarrierFrequencyHz'])
svid_select = pn.widgets.Select(name='Select', options=[2,5,16,18,20,26,31,18,26,10])

checkbox_all_gps = pn.widgets.Checkbox(name='all')
checkbox_group_gps = pn.widgets.CheckBoxGroup(
    name='Checkbox Group', value=['G02'], options=[item for item in satelliteList if item.startswith('G')],
    inline=False)
checkbox_all_glo = pn.widgets.Checkbox(name='all')
checkbox_group_glo = pn.widgets.CheckBoxGroup(
    name='Checkbox Group', value=[2,5,16], options=[item for item in satelliteList if item.startswith('R')],
    inline=False)
checkbox_all_gal = pn.widgets.Checkbox(name='all')
checkbox_group_gal = pn.widgets.CheckBoxGroup(
    name='Checkbox Group', value=[2,5,16], options=[item for item in satelliteList if item.startswith('E')],
    inline=False)
checkbox_all_bei = pn.widgets.Checkbox(name='all')
checkbox_group_bei = pn.widgets.CheckBoxGroup(
    name='Checkbox Group', value=[2,5,16], options=[item for item in satelliteList if item.startswith('C')],
    inline=False)

def selectMeasurement(meas, gps_all, gps_svid):
    selected_prn = []
    if gps_all:
        checkbox_group_gps.disabled = True
        selected_prn += checkbox_group_gps.options
    else:
        checkbox_group_gps.disabled = False
        selected_prn += gps_svid

    if not selected_prn:
        return
    return log.raw.loc[log.raw['prn'].isin(selected_prn), [meas, 'prn', 'timestamp']]
dfi_raw = hvplot.bind(selectMeasurement, meas_select, checkbox_all_gps, checkbox_group_gps).interactive()

#plot_opts = dict(max_height=700, xlabel="Local time", grid=True, responsive=True)
gnssmeasGrid = pn.GridSpec(wsizing_mode='stretch_both')
gnssmeasGrid[:1, :1] = dfi_raw.hvplot(x="timestamp", by='prn', grid=True, responsive=True).output()
#gnssmeasGrid[:1, :3] = dfi_sine.hvplot(title='Sine', **plot_opts).output()

# =============================================================================
# Main

# sidebar grid

# Instantiate the template with widgets displayed in the sidebar
template = pn.template.FastGridTemplate(
    title="Mimir Dashboard",
    sidebar=[providersCheckButtons, meas_select, 
             pn.Row(pn.Column('G', checkbox_all_gps, checkbox_group_gps), 
                    pn.Column('R', checkbox_all_glo, checkbox_group_glo),
                    pn.Column('E', checkbox_all_gal, checkbox_group_gal),
                    pn.Column('C', checkbox_all_bei, checkbox_group_bei))],
    theme="dark"
)

tabs = pn.Tabs(("Map", mapGrid), ("Coordinates", coordinateGrid), ("Measurements", gnssmeasGrid), dynamic=True)

template.main[:6, :12] = tabs
template.servable()




