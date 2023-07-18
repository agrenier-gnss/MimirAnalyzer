
from datetime import datetime
import numpy as np

# =====================================================================================================================
class GnssSystems:
    GPS = 1
    GLONASS = 3
    BEIDOU = 5
    GALILEO = 6

STATE_TOW_DECODED = 8

# =====================================================================================================================

# Value should be fix by the first value received and keep for the rest of the pseudorange conversion
GNSS_TIME_BIAS_NANOS = np.nan

# =====================================================================================================================

def getFixDictionnary(line):
    try: 
        mdict = {
            "provider" : line[1],
            "timestamp": float(line[8])/1e3,
            'datetime' : np.datetime64(int(line[8]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
            "latitude" : float(line[2]),
            "longitude": float(line[3]),
            "altitude" : float(line[4]) if line[4] != "" else float("nan"),
        }
    except ValueError:
        return

    return mdict

# =====================================================================================================================

def getRawDictionnary(line):
    try:
        values = [float(value) if value != "" else float("nan") for value in line[1:-2]]
        
        keys = ["timestamp", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
                "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
                "HardwareClockDiscontinuityCount","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
                "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
                "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
                "AccumulatedDeltaRangeUncertaintyMeters","CarrierFrequencyHz","CarrierCycles","CarrierPhase",\
                "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","ConstellationType","AgcDb", "BasebandCn0DbHz",\
                "FullInterSignalBiasNanos","FullInterSignalBiasUncertaintyNanos","SatelliteInterSignalBiasNanos",\
                "SatelliteInterSignalBiasUncertaintyNanos"]
        
        mdict = {}
        for i in range(len(keys)):
            if keys[i] == "Svid":
                mdict[keys[i]] = int(values[i])
            elif keys[i] == "ConstellationType":
                mdict[keys[i]] = int(values[i])
            elif keys[i] == "timestamp":
                mdict[keys[i]] = float(line[1])/1e3,
                mdict['datetime'] = np.datetime64(int(line[1]), 'ms')
            else:
                mdict[keys[i]] = values[i]
            i += 1
        
        mdict["CodeType"] = line[-2]
        mdict["ChipsetElapsedRealtimeNanos"] = float(line[-1])
        mdict["prn"] = f"{getSystemLetter(mdict['ConstellationType'])}{mdict['Svid']:02d}"

    except ValueError:

        return

    return mdict

# =====================================================================================================================

def getPosDictionnary(line):
    try:
        leapseconds = np.timedelta64(18, 's')
        datetime = np.datetime64(f'{line[0:4]}-{line[5:7]}-{line[8:10]}T{line[11:23]}') - leapseconds
        timestamp = (datetime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        mdict = {
            "provider" : 'REF',
            "datetime" : datetime,
            "timestamp": timestamp, 
            "latitude" : float(line[24:38]),
            "longitude": float(line[39:53]),
            "altitude" : float(line[54:64]),
        }
    except ValueError:
        return

    return mdict
    

# =====================================================================================================================

def getSystemLetter(system:int):
    match system:
        case GnssSystems.GPS:
            return 'G'
        case GnssSystems.GLONASS:
            return 'R'
        case GnssSystems.BEIDOU:
            return 'C'
        case GnssSystems.GALILEO:
            return 'E'
        case _:
            return 'U'
        
# =====================================================================================================================


def getPseudoranges(raw : dict):

    # Check if tracking state correct
    state = raw["State"]
    if not (state & STATE_TOW_DECODED):
        pseudorange = np.nan
        return

    # Retrieve transmitted time 
    t_tx = raw["ReceivedSvTimeNanos"]

    # Retrieve received time
    if np.isnan(GNSS_TIME_BIAS_NANOS):
        GNSS_TIME_BIAS_NANOS = raw["FullBiasNanos"] + raw["BiasNanos"]
        print(GNSS_TIME_BIAS_NANOS)
    
    t_rx_gnss = raw["TimeNanos"] + raw["TimeOffsetNanos"] - GNSS_TIME_BIAS_NANOS

    # Transform receive time from GNSS to constellation time
    match(raw["ConstellationType"]):
        case GnssSystems.GPS:
            return 'G'
        case GnssSystems.GLONASS:
            return 'R'
        case GnssSystems.BEIDOU:
            return 'C'
        case GnssSystems.GALILEO:
            return
        case _:
            return
            
    return