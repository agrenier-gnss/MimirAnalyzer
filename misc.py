
from datetime import datetime
import numpy as np

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
        case 1:
            return 'G'
        case 3: 
            return 'R'
        case 5:
            return 'C'
        case 6:
            return 'E'
        case _:
            return 'U'