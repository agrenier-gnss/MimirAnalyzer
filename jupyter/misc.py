
import numpy as np
import pymap3d as pm
from datetime import timedelta

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
    out = [1 if state & (1 << (bits-1-n)) else np.nan for n in range(bits)]

    # Align state on a seperate integer to plot
    out = [out[i] * (bits-i) for i in range(bits)]

    # Clean list from nan
    if type in 'tracking':
        out = [GnssState_Str[x] for x in out if str(x) != 'nan']
    elif type in 'phase':
        out = [GnssStateADR_Str[x] for x in out if str(x) != 'nan']

    return {"State_split":out}

