
from datetime import datetime

# =====================================================================================================================

def getFixDictionnary(line):
    try: 
        mdict = {
            "provider" : line[1],
            "timestamp": datetime.fromtimestamp(int(line[8])/1e3),
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
                mdict[keys[i]] = int(line[1]),
            else:
                mdict[keys[i]] = values[i]
            i += 1
        
        mdict["CodeType"] = line[-2]
        mdict["ChipsetElapsedRealtimeNanos"] = float(line[-1])
        mdict["prn"] = f"{getSystemLetter(mdict['ConstellationType'])}{mdict['Svid']:02d}"

    except ValueError:
        return

    return mdict

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