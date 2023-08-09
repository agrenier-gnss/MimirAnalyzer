
import numpy as np
import pymap3d as pm

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