
import numpy as np
import pymap3d as pm
from datetime import timedelta
# ======================================================================================================================

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

# ======================================================================================================================

def getLogDictionnary(device_name, filepath, mode):
    return {
        'device_name' : device_name,
        'filepath'    : filepath,
        'mode'        : mode
    }

# ----------------------------------------------------------------------------------------------------------------------

def getSystemStr(letter):
    match letter:
        case 'G':
            return "GPS"
        case 'E':
            return "Galileo"
        case 'R':
            return "GLONASS"
        case 'C':
            return "BeiDou"
        case 'I': 
            return "IRNSS"
        case 'J':
            return "QZSS"
        case 'S':
            return "SBAS"

# ======================================================================================================================
# Coordinate conversions

def convert2ENU(lat, lon, alt, ref):
    east, north, up = pm.geodetic2enu(lat, lon, alt, ref[0], ref[1], ref[2])
    return {"east":east, "north":north, "up":up}

# ----------------------------------------------------------------------------------------------------------------------

def getHorizontalError(east, north):
    error = np.sqrt(north**2 + east**2)
    return {"2D_error":error}

# ----------------------------------------------------------------------------------------------------------------------

def get3DError(east, north, up):
    error = np.sqrt(north**2 + east**2 + up**2)
    return {"2D_error":error}

# ======================================================================================================================

def filterPercentile(df, data_name, percentile):
    q = df[data_name].quantile(percentile)
    df = df[df[data_name].abs() < q]
    return df

def removeFirstEntries(df, seconds=30):
    df = df.set_index('datetime')
    df = df[df.index[0] + timedelta(seconds=30):]

    return df

def filterValues(df, data_name, value):
    df.dropna(subset=[data_name], how='all', inplace=True)
    size_before_filter = len(df)
    df = df[df[data_name].abs() < value]
    size_after_filter = len(df)
    diff = size_before_filter - size_after_filter
    print(f"Rows removed {diff} ({diff / (size_before_filter)*100:.3f}%)")

    return df

# ======================================================================================================================

def getSplitState(state, bits=1, type='tracking'):
    
    # Split to bit array
    states = [1 if state & (1 << (bits-1-n)) else np.nan for n in range(bits)]

    # Align state on a seperate integer to plot
    states = [states[i] * (bits-i) for i in range(bits)]

    # Clean list from nan
    if type in 'tracking':
        states = [GnssState_Str[x] for x in states if str(x) != 'nan']
    elif type in 'phase':
        states = [GnssStateADR_Str[x] for x in states if str(x) != 'nan']

    out = {}
    if type in 'tracking':
        for _state in list(GnssState_Str.values()):
            if _state in states:
                out[f"TRACK_{_state}"] = True
            else:
                out[f"TRACK_{_state}"] = False
    elif type in 'phase':
        for _state in list(GnssStateADR_Str.values()):
            if _state in states:
                out[f"ADR_{_state}"] = True
            else:
                out[f"ADR_{_state}"] = False

    return out

