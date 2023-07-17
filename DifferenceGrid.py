import hvplot.pandas
import pandas as pd
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter
import pymap3d as pm

from logparser import LogReader, PosReader

# =============================================================================

def getDifferenceGrid(log : LogReader):

    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    
    # Bind widget callback
    dfi_diff = hvplot.bind(plotDiff, log, reset_button).interactive()

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
    
    return pn.Column(reset_button, diff_grid)
    

# =============================================================================

def plotDiff(log, reset_button):

    # Align 
    gps = log.fix.loc[log.fix['provider'].isin(['GPS']), ['datetime', 'latitude', 'longitude', 'altitude']]
    ref = log.fix.loc[log.fix['provider'].isin(['REF']), ['datetime', 'latitude', 'longitude', 'altitude']]

    gps = gps.set_index('datetime')
    ref = ref.set_index('datetime')

    ref_enu = ref.iloc[0].tolist()[0:4]

    gps[["east", "north", "up"]] = gps.apply(
        lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
        axis='columns', result_type='expand')
    ref[["east", "north", "up"]] = ref.apply(
        lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
        axis='columns', result_type='expand')
    
    gps, ref = gps.align(ref)
    
    s = gps.interpolate(method='time') - ref.interpolate(method='time')

    # s['latitude'] = s['latitude'].apply(lambda x: x*43.5e3)
    # s['longitude'] = s['longitude'].apply(lambda x: x*43.5e3)

    return s

# =============================================================================

def convert2ENU(lat, lon, alt, ref):

    east, north, up = pm.geodetic2enu(lat, lon, alt, ref[0], ref[1], ref[2])

    return {"east":east, "north":north, "up":up}

# =============================================================================

if __name__ == "__main__":
    
    filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
    log = LogReader(filepath)

    filepath = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
    ref = PosReader(filepath)

    _df = pd.concat([log.fix, ref.pos], ignore_index=True)
    log.fix = _df 

    plotDiff(log, 0)