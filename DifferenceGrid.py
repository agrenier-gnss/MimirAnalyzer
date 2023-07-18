import hvplot.pandas
import pandas as pd
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter
import pymap3d as pm

from logparser import LogReader, PosReader

# =============================================================================

def getDifferenceGrid(log : LogReader):

    reset_button = pn.widgets.Button(name='Reset', button_type='primary')

    providersCheckButtons_A = pn.widgets.RadioButtonGroup(
        name='Providers A', 
        value='GPS', 
        options=list(set(log.fix["provider"])),
        button_type='primary')
    
    providersCheckButtons_B = pn.widgets.RadioButtonGroup(
        name='Providers B', 
        value='FLP', 
        options=list(set(log.fix["provider"])),
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
    

# =============================================================================

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