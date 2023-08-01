import hvplot.pandas
import panel as pn
import numpy as np
from bokeh.models.formatters import DatetimeTickFormatter
import pymap3d as pm
import folium

from logparser import LogReader

# =====================================================================================================================
# COORDINATE GRID

def getCoordinateGrid(log : LogReader):

    providers = list(set(log.fix["provider"]))

    # Define widgets
    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    providersCheckButtons = pn.widgets.CheckButtonGroup(
        name='Providers', 
        value=['GPS'], 
        options=list(set(log.fix["provider"])),
        button_type='primary')
    
    # Bind widget callback
    dfi_fix = hvplot.bind(plotFix, log, providersCheckButtons).interactive()

    # Format
    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         max_height=350, 
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    
    coordinate_grid = pn.GridSpec(sizing_mode='stretch_both')
    coordinate_grid[0:1, 0:1] = dfi_fix.hvplot(
         x="datetime", y="latitude", by="provider", ylabel="Latitude [DD]", **plot_opts).output()
    coordinate_grid[1:2, 0:1] = dfi_fix.hvplot(
         x="datetime", y="longitude", by="provider", ylabel="Longitude [DD]", **plot_opts).output()
    coordinate_grid[2:3, 0:1] = dfi_fix.hvplot(
         x="datetime", y="altitude", by="provider", ylabel="Altitude [m]", **plot_opts).output()
    
    return pn.Column(pn.Row(reset_button, providersCheckButtons), coordinate_grid)

# -----------------------------------------------------------------------------

def plotFix(log, providersCheckButtons):
    if providersCheckButtons:
        return log.fix.loc[log.fix['provider'].isin(providersCheckButtons)]
    else:
        return log.fix
    
# =====================================================================================================================
# DIFFERENCE GRID

def getDifferenceGrid(log : LogReader):

    reset_button = pn.widgets.Button(name='Reset', button_type='primary')

    providers = list(set(log.fix["provider"]))
    providersCheckButtons_A = pn.widgets.RadioButtonGroup(
        name='Providers A', 
        value=providers[0], 
        options=providers,
        button_type='primary')
    
    providersCheckButtons_B = pn.widgets.RadioButtonGroup(
        name='Providers B', 
        value=providers[0], 
        options=providers,
        button_type='primary')
    
    # Bind widget callback
    dfi_diff = hvplot.bind(plotDiff, log, reset_button, providersCheckButtons_A, providersCheckButtons_B).interactive()

    # Format
    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         max_height=350, 
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    
    diff_grid = pn.GridSpec(sizing_mode='stretch_both')
    diff_grid[0:1, 0:1] = dfi_diff.hvplot(
         x="datetime", y="east", ylabel="East [m]", **plot_opts).output()
    diff_grid[1:2, 0:1] = dfi_diff.hvplot(
         x="datetime", y="north", ylabel="North [m]", **plot_opts).output()
    diff_grid[2:3, 0:1] = dfi_diff.hvplot(
         x="datetime", y="up", ylabel="Up [m]", **plot_opts).output()
    
    return pn.Column(pn.Row(reset_button, providersCheckButtons_A, "V.S.", providersCheckButtons_B), diff_grid)
    

# -----------------------------------------------------------------------------

def plotDiff(log, reset_button, providersCheckButtons_A, providersCheckButtons_B):

    # Align 
    pos_A = log.fix.loc[log.fix['provider'].isin([providersCheckButtons_A]), 
                      ['datetime', 'latitude', 'longitude', 'altitude']]
    pos_B = log.fix.loc[log.fix['provider'].isin([providersCheckButtons_B]), 
                      ['datetime', 'latitude', 'longitude', 'altitude']]

    pos_A = pos_A.set_index('datetime')
    pos_B = pos_B.set_index('datetime')

    ref_enu = pos_A.iloc[0].tolist()[0:4]

    pos_A[["east", "north", "up"]] = pos_A.apply(
        lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
        axis='columns', result_type='expand')
    pos_B[["east", "north", "up"]] = pos_B.apply(
        lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
        axis='columns', result_type='expand')
    
    pos_A, pos_B = pos_A.align(pos_B)
    
    s = pos_A.interpolate(method='time') - pos_B.interpolate(method='time')

    return s

# -----------------------------------------------------------------------------

def convert2ENU(lat, lon, alt, ref):

    east, north, up = pm.geodetic2enu(lat, lon, alt, ref[0], ref[1], ref[2])

    return {"east":east, "north":north, "up":up}

# =====================================================================================================================
# GNSS MEASUREMENT GRID

GnssState_Str = {
    0 : "UNKNOWN",
    1 : "CODE_LOCK",
    2 : "BIT_SYNC",
    3 : "SUBFRAME_SYNC",        
    4 : "TOW_DECODED",          
    5 : "MSEC_AMBIGUOUS",       
    6 : "SYMBOL_SYNC",          
    7 : "GLO_STRING_SYNC",      
    8 : "GLO_TOD_DECODED",      
    9 : "BDS_D2_BIT_SYNC",      
    10 : "BDS_D2_SUBFRAME_SYNC", 
    11 : "GAL_E1BC_CODE_LOCK",   
    12 : "GAL_E1C_2ND_CODE_LOCK",
    13 : "GAL_E1B_PAGE_SYNC",    
    14 : "SBAS_SYNC",            
    15 : "TOW_KNOWN",            
    16 : "GLO_TOD_KNOWN",       
    17 : "S_2ND_CODE_LOCK"        
}

GnssStateADR_Str = {
    0 : "UNKNOWN",             
    1 : "VALID",               
    2 : "RESET",               
    3 : "CYCLE_SLIP",                 
    4 : "HALF_CYCLE_RESOLVED",        
    5 : "HALF_CYCLE_REPORTED"            
}

# -----------------------------------------------------------------------------

def getGnssMeasurementGrid(log : LogReader):

    satelliteList = list(set(log.raw["prn"]))
    satelliteList.sort()

    measurementsOptions = log.raw.columns.values.tolist()
    measurementsOptions.sort()

    # Define widgets
    meas_select = pn.widgets.Select(name='Select', value='Cn0DbHz', options=measurementsOptions)
    kind_widget = pn.widgets.Select(name='kind', value='line', options=['line', 'scatter'])
    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    _sats = [item for item in satelliteList if item.startswith('G')]
    checkbox_group_gps = pn.widgets.CheckBoxGroup(
        name='Checkbox Group', value=[_sats[0]], 
        options=_sats,
        inline=False)
    _sats = [item for item in satelliteList if item.startswith('R')]
    checkbox_group_glo = pn.widgets.CheckBoxGroup(
        name='Checkbox Group', value=[_sats[0]], 
        options=_sats,
        inline=False)
    _sats = [item for item in satelliteList if item.startswith('E')]
    checkbox_group_gal = pn.widgets.CheckBoxGroup(
        name='Checkbox Group', value=[_sats[0]], 
        options=_sats,
        inline=False)
    _sats = [item for item in satelliteList if item.startswith('C')]
    checkbox_group_bei = pn.widgets.CheckBoxGroup(
        name='Checkbox Group', value=[_sats[0]], 
        options=_sats,
        inline=False)
    
    dfi_raw = hvplot.bind(selectGnssMeasurement, reset_button, log, meas_select, 
                          checkbox_group_gps, checkbox_group_glo, 
                          checkbox_group_gal, checkbox_group_bei).interactive()

    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    gnssmeas_grid = pn.GridSpec(wsizing_mode='stretch_both')
    gnssmeas_grid[:1, :1] = dfi_raw.hvplot(x="datetime", y='measurement', by='prn', 
                                           kind=kind_widget, **plot_opts).output()

    sat_selection = pn.Row(pn.Column('G', checkbox_group_gps), pn.Column('R', checkbox_group_glo),
                           pn.Column('E', checkbox_group_gal), pn.Column('C', checkbox_group_bei))

    return pn.Column(reset_button, gnssmeas_grid), [meas_select, kind_widget, sat_selection]

# -----------------------------------------------------------------------------

# Bind plot callback
def selectGnssMeasurement(reset_button, log, meas, gps_svid, glo_svid, gal_svid, bei_svid):
    
    # Satellite selection
    selected_prn = []
    if gps_svid == [] and glo_svid == [] and gal_svid == [] and bei_svid == []:
        return
    else:
        selected_prn += gps_svid
        selected_prn += glo_svid
        selected_prn += gal_svid
        selected_prn += bei_svid

    if not selected_prn:
        return
    
    # Display state
    if meas == "State":
        df = log.raw.loc[log.raw['prn'].isin(selected_prn), ['prn', 'datetime', 'timestamp', 'State']]
        df[["State_split"]] = df.apply(lambda row: getSplitState(row['State'], bits=17, type='tracking'), axis='columns', result_type='expand')
        df = df.explode('State_split')
        df.rename(columns={"State_split":"measurement"}, inplace=True)
    elif meas == "AccumulatedDeltaRangeState":
        df = log.raw.loc[log.raw['prn'].isin(selected_prn), ['prn', 'datetime', 'timestamp', 'AccumulatedDeltaRangeState']]
        df[["State_split"]] = df.apply(lambda row: getSplitState(row['AccumulatedDeltaRangeState'], bits=5, type='phase'), axis='columns', result_type='expand')
        df = df.explode('State_split')
        df.rename(columns={"State_split":"measurement"}, inplace=True)
    else:
        df = log.raw.loc[log.raw['prn'].isin(selected_prn), [meas, 'prn', 'timestamp', 'datetime']]
        df.rename(columns={meas:"measurement"}, inplace=True)
    return df

# -----------------------------------------------------------------------------

def getSplitState(state, bits=1, type='tracking'):
    
    # Split to bit array
    out = [1 if state & (1 << (bits-1-n)) else np.nan for n in range(bits)]

    # Align state on a seperate integer to plot
    out = [out[i] * (bits-i) for i in range(bits)]

    # Clean list from nan
    if type in 'tracking':
        out = [GnssState_Str[x] for x in out if str(x) != 'nan']
    elif type in 'phase':
        out = [GnssStateADR_Str[x] for x in out if str(x) != 'nan']

    return {"State_split":out}

# =====================================================================================================================
# HEALTH MEASUREMENT GRID

def getHealthGrid(log : LogReader):

    measurementsOptions = log.health.columns.values.tolist()
    measurementsOptions.sort()

    # Define widgets
    meas_select = pn.widgets.Select(name='Select', value='value_0', options=measurementsOptions)
    kind_widget = pn.widgets.Select(name='kind', value='line', options=['line', 'scatter'])
    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    healthCheckButton = pn.widgets.RadioButtonGroup(name='type', 
                                                          value='ECG', 
                                                          options=['ECG', 'PPG'], 
                                                          button_type='primary')
    
    dfi_raw = hvplot.bind(selectHealthMeasurement, reset_button, log, meas_select, healthCheckButton).interactive()

    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    meas_grid = pn.GridSpec(wsizing_mode='stretch_both')
    meas_grid[:1, :1] = dfi_raw.hvplot(x="datetime", y='measurement', kind=kind_widget, **plot_opts).output()
    #gnssmeasGrid[:1, :3] = dfi_sine.hvplot(title='Sine', **plot_opts).output()

    return pn.Column(reset_button, meas_grid), [healthCheckButton, meas_select, kind_widget]

# -----------------------------------------------------------------------------

# Bind plot callback
def selectHealthMeasurement(reset_button, log, meas, healthCheckButton):
    
    # Satellite selection
    print(log.health)
    df = log.health.loc[log.health['sensor'].isin([healthCheckButton]), [meas, 'timestamp', 'datetime']]
    df.rename(columns={meas:"measurement"}, inplace=True)
    
    return df

# =====================================================================================================================
# IMU MEASUREMENTS GRID

def getImuMeasurementGrid(log : LogReader):

    measurementsOptions = log.motion.columns.values.tolist()
    measurementsOptions.sort()

    # Define widgets
    meas_select = pn.widgets.Select(name='Select', value='x', options=measurementsOptions)
    kind_widget = pn.widgets.Select(name='kind', value='line', options=['line', 'scatter'])
    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    imuCheckButton = pn.widgets.RadioButtonGroup(name='type', 
                                                value='ACC', 
                                                options=['ACC', 'GYR', 'MAG'], 
                                                button_type='primary')
    
    dfi_raw = hvplot.bind(selectImuMeasurement, reset_button, log, meas_select, imuCheckButton).interactive()

    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    meas_grid = pn.GridSpec(wsizing_mode='stretch_both')
    meas_grid[:1, :1] = dfi_raw.hvplot(x="datetime", y='measurement', kind=kind_widget, **plot_opts).output()
    #gnssmeasGrid[:1, :3] = dfi_sine.hvplot(title='Sine', **plot_opts).output()

    return pn.Column(reset_button, meas_grid), [imuCheckButton, meas_select, kind_widget]

# -----------------------------------------------------------------------------

# Bind plot callback
def selectImuMeasurement(reset_button, log, meas, healthCheckButton):
    
    # Satellite selection
    df = log.motion.loc[log.motion['sensor'].isin([healthCheckButton]), [meas, 'timestamp', 'datetime']]
    df.rename(columns={meas:"measurement"}, inplace=True)
    
    return df

# =====================================================================================================================
# MAP GRID

color_map = {'GPS':'blue', 'gps':'blue', 'FLP':'red', 'NLP':'green', 'REF':'purple'}

# -----------------------------------------------------------------------------

def getMapGrid(log : LogReader):

    providers = list(set(log.fix["provider"]))

    m = folium.Map(location=[log.fix['latitude'][0],log.fix['longitude'][0]], zoom_start=16)

    for prov in providers:
        trajectory = folium.FeatureGroup(name=prov).add_to(m)
        points = []
        df = log.fix.loc[log.fix['provider'].isin([prov])]
        for index, row in df.iterrows():
            points.append((row['latitude'], row['longitude']))
            trajectory.add_child(folium.CircleMarker(location=points[-1], radius=1, color=color_map[prov]))
        trajectory.add_child(folium.PolyLine(locations=points, weight=3, color=color_map[prov]))
    folium.LayerControl().add_to(m)

    map_grid = pn.GridSpec(sizing_mode='stretch_both', responsive=True)
    map_grid[ 0:1, 0:1] = pn.pane.plot.Folium(m, height=500)

    return map_grid

# -----------------------------------------------------------------------------

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
