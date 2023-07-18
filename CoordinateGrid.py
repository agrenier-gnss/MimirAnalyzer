import hvplot.pandas
import pandas as pd
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter

from logparser import LogReader, PosReader

# =============================================================================

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
    
    #geodeticPlot = dfi_trajectory.hvplot(x="timestamp", y=["latitude", "longitude", "altitude"], shared_axes=False, subplots=True, **plot_opts).cols(1).output()
    coordinate_grid = pn.GridSpec(sizing_mode='stretch_both')
    coordinate_grid[0:1, 0:1] = dfi_fix.hvplot(
         x="datetime", y="latitude", by="provider", ylabel="Latitude [DD]", **plot_opts).output()
    coordinate_grid[1:2, 0:1] = dfi_fix.hvplot(
         x="datetime", y="longitude", by="provider", ylabel="Longitude [DD]", **plot_opts).output()
    coordinate_grid[2:3, 0:1] = dfi_fix.hvplot(
         x="datetime", y="altitude", by="provider", ylabel="Altitude [m]", **plot_opts).output()
    
    return pn.Column(pn.Row(reset_button, providersCheckButtons), coordinate_grid)

# =============================================================================

def plotFix(log, providersCheckButtons):
    if providersCheckButtons:
        return log.fix.loc[log.fix['provider'].isin(providersCheckButtons)]
    else:
        return log.fix
    
