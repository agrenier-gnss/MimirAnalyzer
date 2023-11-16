
import numpy as np
import pymap3d as pm
from datetime import timedelta
# ======================================================================================================================

GnssState_Str = {
    0  : "TRACK_UNKNOWN",
    1  : "TRACK_CODE_LOCK",
    2  : "TRACK_BIT_SYNC",
    3  : "TRACK_SUBFRAME_SYNC",        
    4  : "TRACK_TOW_DECODED",          
    5  : "TRACK_MSEC_AMBIGUOUS",       
    6  : "TRACK_SYMBOL_SYNC",          
    7  : "TRACK_GLO_STRING_SYNC",      
    8  : "TRACK_GLO_TOD_DECODED",      
    9  : "TRACK_BDS_D2_BIT_SYNC",      
    10 : "TRACK_BDS_D2_SUBFRAME_SYNC", 
    11 : "TRACK_GAL_E1BC_CODE_LOCK",   
    12 : "TRACK_GAL_E1C_2ND_CODE_LOCK",
    13 : "TRACK_GAL_E1B_PAGE_SYNC",    
    14 : "TRACK_SBAS_SYNC",            
    15 : "TRACK_TOW_KNOWN",            
    16 : "TRACK_GLO_TOD_KNOWN",       
    17 : "TRACK_S_2ND_CODE_LOCK"        
}

GnssStateADR_Str = {
    0 : "ADR_UNKNOWN",             
    1 : "ADR_VALID",               
    2 : "ADR_RESET",               
    3 : "ADR_CYCLE_SLIP",                 
    4 : "ADR_HALF_CYCLE_RESOLVED",        
    5 : "ADR_HALF_CYCLE_REPORTED"            
}

# List of PRN GPS satellites with L5 enabled (Block 2F and on-ward)
GPS_SAT_L5_ENABLED = [25,1,24,27,30,6,9,3,26,8,10,32,4,18,23,14,11,28]

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

def getSystemLetter(self, system:int):
        match system:
            case GnssSystems.GPS:
                return 'G'
            case GnssSystems.SBAS:
                return 'S'
            case GnssSystems.GLONASS:
                return 'R'
            case GnssSystems.QZSS:
                return 'J'
            case GnssSystems.BEIDOU:
                return 'C'
            case GnssSystems.GALILEO:
                return 'E'
            case GnssSystems.IRNSS:
                return 'I'
            case _:
                return 'U'

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

# ----------------------------------------------------------------------------------------------------------------------

def get2DRMSE(east, north):
    error = np.sqrt(np.mean(east**2 + north**2))
    return error

# ======================================================================================================================

def filterPercentile(df, data_name, percentile):
    size_before_filter = len(df)
    q = df[data_name].quantile(percentile)
    df = df[df[data_name].abs() < q]
    size_after_filter = len(df)
    diff = size_before_filter - size_after_filter
    print(f"Rows removed {diff} ({diff / (size_before_filter)*100:.3f}%)")
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
    print(f"Rows removed {diff: <4} ({diff / (size_before_filter)*100:.3f}% - {100 - diff / (size_before_filter)*100: >8.2f}%)")

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
                out[f"{_state}"] = True
            else:
                out[f"{_state}"] = False
    elif type in 'phase':
        for _state in list(GnssStateADR_Str.values()):
            if _state in states:
                out[f"{_state}"] = True
            else:
                out[f"{_state}"] = False

    return out

# ======================================================================================================================

def fixfile(filepath_in, filepath_out, mode):
    string_to_add = ","

    with open(filepath_in, 'r') as f:
        file_lines = []
        for line in f:
            line_split = line.strip().split(",")
            if line_split[0] == 'Raw':
                if mode == 'old':
                    line_split.insert(30, '')
                    line_split.append('\n')
                elif mode == 'new':
                    line_split.append('\n')
                file_lines.append(','.join(line_split))
            else:
                file_lines.append(line)

    with open(filepath_out, 'w') as f:
        f.writelines(file_lines) 

    return 

if __name__ == "__main__":

    import os

    #path = ".data/2023_Dataset_Hervanta/S2_dynamic_campus/raw"
    # dirs = os.listdir(".data/2023_Dataset_Hervanta/S2_dynamic_campus/raw")
    # files = [('log_GooglePixel7_20230801110405.txt', 'old'), 
    #          ('log_GooglePixelWatch_20230801110404.txt', 'old'), 
    #          ('log_OnePlusNord2_20230811103018.txt', 'new'), 
    #          ('log_SamsungA52_20230811101903.txt', 'new'), 
    #          ('log_Xiaomi11T_20230801111451.txt', 'old')]

    path = ".data/2023_Dataset_Hervanta/S4_dynamic_lake/raw"
    files = [('log_GooglePixel7_20230811150244.txt', 'new'),
             ('log_GooglePixelWatch_20230811150346.txt', 'new'),
             ('log_OnePlusNord2_20230811150159.txt', 'new'),
             ('log_SamsungA52_20230811150240.txt', 'new'),
             ('log_Xiaomi11_20230811150208.txt', 'new')]

    for mfile in files:
        fixfile(f"{path}/{mfile[0]}", f"{path}/{mfile[0][:-4]}_modified.txt", mode=mfile[1])

