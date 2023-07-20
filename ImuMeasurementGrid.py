import hvplot.pandas
from logparser import LogReader
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter
import numpy as np
import math

# =============================================================================

def getImuMeasurementGrid(log : LogReader):

    measurementsOptions = log.motion.columns.values.tolist()
    measurementsOptions.sort()

    # Define widgets
    meas_select = pn.widgets.Select(name='Select', value='x', options=measurementsOptions)
    kind_widget = pn.widgets.Select(name='kind', value='line', options=['line', 'scatter'])
    reset_button = pn.widgets.Button(name='Reset', button_type='primary')
    healthCheckButton = pn.widgets.RadioButtonGroup(name='type', 
                                                          value='ACC', 
                                                          options=['ACC', 'GYR', 'MAG'], 
                                                          button_type='primary')
    
    dfi_raw = hvplot.bind(selectMeasurement, reset_button, log, meas_select, healthCheckButton).interactive()

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

# =============================================================================

# Bind plot callback
def selectMeasurement(reset_button, log, meas, healthCheckButton):
    
    # Satellite selection
    df = log.motion.loc[log.motion['sensor'].isin([healthCheckButton]), [meas, 'timestamp', 'datetime']]
    df.rename(columns={meas:"measurement"}, inplace=True)
    
    return df

# =============================================================================