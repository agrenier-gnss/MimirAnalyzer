import hvplot.pandas
from logparser import LogReader
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter

# =============================================================================

measurementsOptions = ["TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
                "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
                "HardwareClockDiscontinuityCount","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
                "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
                "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
                "AccumulatedDeltaRangeUncertaintyMeters","CarrierCycles","CarrierPhase",\
                "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","AgcDb", "BasebandCn0DbHz",\
                "FullInterSignalBiasNanos","FullInterSignalBiasUncertaintyNanos","SatelliteInterSignalBiasNanos",\
                "SatelliteInterSignalBiasUncertaintyNanos"]

meas_select = pn.widgets.Select(name='Select', value='Cn0DbHz', options=measurementsOptions)

# =============================================================================

def getGnssMeasurementGrid(log : LogReader):

    satelliteList = list(set(log.raw["prn"]))
    satelliteList.sort()

    # Define widgets
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
    
    # Bind plot callback
    def selectMeasurement(log, meas, gps_svid, glo_svid, gal_svid, bei_svid):
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
        return log.raw.loc[log.raw['prn'].isin(selected_prn), [meas, 'prn', 'timestamp', 'datetime']]
    
    dfi_raw = hvplot.bind(selectMeasurement, log, meas_select, 
                          checkbox_group_gps, checkbox_group_glo, 
                          checkbox_group_gal, checkbox_group_bei).interactive()

    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    gnssmeas_grid = pn.GridSpec(wsizing_mode='stretch_both')
    gnssmeas_grid[:1, :1] = dfi_raw.hvplot(x="datetime", by='prn', **plot_opts).output()
    #gnssmeasGrid[:1, :3] = dfi_sine.hvplot(title='Sine', **plot_opts).output()

    sat_selection = pn.Row(pn.Column('G', checkbox_group_gps), pn.Column('R', checkbox_group_glo),
                           pn.Column('E', checkbox_group_gal), pn.Column('C', checkbox_group_bei))

    return gnssmeas_grid, [meas_select, sat_selection]

# =============================================================================

# checkbox_all_glo = pn.widgets.Checkbox(name='all')
# checkbox_group_glo = pn.widgets.CheckBoxGroup(
#     name='Checkbox Group', value=[2,5,16], 
#     options=[item for item in satelliteList if item.startswith('R')],
#     inline=False)
# checkbox_all_gal = pn.widgets.Checkbox(name='all')
# checkbox_group_gal = pn.widgets.CheckBoxGroup(
#     name='Checkbox Group', value=[2,5,16], 
#     options=[item for item in satelliteList if item.startswith('E')],
#     inline=False)
# checkbox_all_bei = pn.widgets.Checkbox(name='all')
# checkbox_group_bei = pn.widgets.CheckBoxGroup(
#     name='Checkbox Group', value=[2,5,16], 
#     options=[item for item in satelliteList if item.startswith('C')],
#     inline=False)