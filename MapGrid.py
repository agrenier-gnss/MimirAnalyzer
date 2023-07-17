import folium

from logparser import LogReader
import panel as pn

# =============================================================================
# Parameters

color_map = {'GPS':'blue', 'FLP':'red', 'NLP':'green', 'REF':'purple'}

# =============================================================================

def getMapGrid(log : LogReader):

    providers = list(set(log.fix["provider"]))

    m = folium.Map(location=[log.fix['latitude'][0],log.fix['longitude'][0]], zoom_start=16)

    for prov in providers:
        points = []
        df = log.fix.loc[log.fix['provider'].isin([prov])]
        for index, row in df.iterrows():
            points.append((row['latitude'], row['longitude']))

        trajectory = folium.FeatureGroup(name=prov).add_to(m)
        trajectory.add_child(folium.PolyLine(locations=points, weight=3, color=color_map[prov]))
    folium.LayerControl().add_to(m)

    map_grid = pn.GridSpec(sizing_mode='stretch_both', responsive=True)
    map_grid[ 0:1, 0:1] = pn.pane.plot.Folium(m, height=500)

    return map_grid

# =============================================================================

# For HvPlot and GeoPandas
# _polyline = []
# for prov in providers:
#     df = log.fix.loc[log.fix['provider'].isin([prov])]
#     df['latitude'].tolist()
#     _polyline.append(LineString(zip(df['longitude'].tolist(), df['latitude'].tolist())))
# fix_geo = gpd.GeoDataFrame(index=providers, geometry=_polyline, crs='epsg:4326')
# print(fix_geo['GPS'])
# map_pane = fix_geo.hvplot(frame_height=800, frame_width=800, tiles=True)
# map_pane.opts(active_tools=['wheel_zoom'])
