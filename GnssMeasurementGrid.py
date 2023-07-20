import hvplot.pandas
from logparser import LogReader
import panel as pn
from bokeh.models.formatters import DatetimeTickFormatter
import numpy as np
from misc import  *

# =============================================================================

# measurementsOptions = ["TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
#                 "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
#                 "HardwareClockDiscontinuityCount","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
#                 "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
#                 "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
#                 "AccumulatedDeltaRangeUncertaintyMeters","CarrierCycles","CarrierPhase",\
#                 "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","AgcDb", "BasebandCn0DbHz",\
#                 "FullInterSignalBiasNanos","FullInterSignalBiasUncertaintyNanos","SatelliteInterSignalBiasNanos",\
#                 "SatelliteInterSignalBiasUncertaintyNanos", "Pseudorange", "Svid", "CarrierFrequencyHz"]
# measurementsOptions.sort()

# meas_select = pn.widgets.Select(name='Select', value='Cn0DbHz', options=measurementsOptions)

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

# =============================================================================

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
    
    dfi_raw = hvplot.bind(selectMeasurement, reset_button, log, meas_select, 
                          checkbox_group_gps, checkbox_group_glo, 
                          checkbox_group_gal, checkbox_group_bei).interactive()

    date_formatter = DatetimeTickFormatter(minutes='%H:%M')
    plot_opts = dict(
         xformatter=date_formatter, 
         xlabel="Local time", 
         grid=True, 
         responsive=True)
    gnssmeas_grid = pn.GridSpec(wsizing_mode='stretch_both')
    gnssmeas_grid[:1, :1] = dfi_raw.hvplot(x="datetime", y='measurement', by='prn', kind=kind_widget, **plot_opts).output()
    #gnssmeasGrid[:1, :3] = dfi_sine.hvplot(title='Sine', **plot_opts).output()

    sat_selection = pn.Row(pn.Column('G', checkbox_group_gps), pn.Column('R', checkbox_group_glo),
                           pn.Column('E', checkbox_group_gal), pn.Column('C', checkbox_group_bei))

    return pn.Column(reset_button, gnssmeas_grid), [meas_select, kind_widget, sat_selection]

# =============================================================================

# Bind plot callback
def selectMeasurement(reset_button, log, meas, gps_svid, glo_svid, gal_svid, bei_svid):
    
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

# =============================================================================

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

# =============================================================================

if __name__=="__main__":

    filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
    log = LogReader(filepath)

    selectMeasurement(0, log, 'State', ['G02-L1'], [],[],[])