
import numpy as np
import pandas as pd

import georinex as gr

# =====================================================================================================================

class GnssSystems:
    GPS      = 1
    SBAS     = 2
    GLONASS  = 3
    QZSS     = 4
    BEIDOU   = 5
    GALILEO  = 6
    IRNSS    = 7

STATE_UNKNOWN               = int(0x00000000) # 0
STATE_CODE_LOCK             = int(0x00000001) # 1
STATE_BIT_SYNC              = int(0x00000002) # 2 
STATE_SUBFRAME_SYNC         = int(0x00000004) # 3
STATE_TOW_DECODED           = int(0x00000008) # 4
STATE_MSEC_AMBIGUOUS        = int(0x00000010) # 5
STATE_SYMBOL_SYNC           = int(0x00000020) # 6
STATE_GLO_STRING_SYNC       = int(0x00000040) # 7
STATE_GLO_TOD_DECODED       = int(0x00000080) # 8
STATE_BDS_D2_BIT_SYNC       = int(0x00000100) # 9
STATE_BDS_D2_SUBFRAME_SYNC  = int(0x00000200) # 10
STATE_GAL_E1BC_CODE_LOCK    = int(0x00000400) # 11
STATE_GAL_E1C_2ND_CODE_LOCK = int(0x00000800) # 12
STATE_GAL_E1B_PAGE_SYNC     = int(0x00001000) # 13
STATE_SBAS_SYNC             = int(0x00002000) # 14
STATE_TOW_KNOWN             = int(0x00004000) # 15
STATE_GLO_TOD_KNOWN         = int(0x00008000) # 16
STATE_2ND_CODE_LOCK         = int(0x00010000) # 17

ADR_STATE_UNKNOWN             = int(0x00000000) # 0
ADR_STATE_VALID               = int(0x00000001) # 1
ADR_STATE_RESET               = int(0x00000002) # 2
ADR_STATE_CYCLE_SLIP          = int(0x00000004) # 3
ADR_STATE_HALF_CYCLE_RESOLVED = int(0x00000008) # 4
ADR_STATE_HALF_CYCLE_REPORTED = int(0x00000010) # 5

L1_FREQUENCY_HZ = int(1575.42e6)
L5_FREQUENCY_HZ = int(1176.45e6)

SPEED_OF_LIGHT = 299792458 # [m/s]

LEAP_SECONDS = 19 # As in 2023

SECONDS_IN_DAY = int(86400)
SECONDS_IN_WEEK = int(604800)
NANOSECONDS_IN_WEEK = int(SECONDS_IN_WEEK * 1e9)
NANOSECONDS_IN_DAY = int(SECONDS_IN_DAY * 1e9)
NANOSECONDS_IN_100MS = int(1e8)

keys_logger = ["Raw", "timestamp", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
                "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
                "HardwareClockDiscontinuityCount","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
                "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
                "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
                "AccumulatedDeltaRangeUncertaintyMeters","CarrierFrequencyHz","CarrierCycles","CarrierPhase",\
                "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","ConstellationType","AgcDb", "BasebandCn0DbHz",\
                "FullInterSignalBiasNanos","FullInterSignalBiasUncertaintyNanos","SatelliteInterSignalBiasNanos",\
                "SatelliteInterSignalBiasUncertaintyNanos", "CodeType", "ChipsetElapsedRealtimeNanos"]

keys_mimir = ["Raw", "timestamp", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
                "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
                "HardwareClockDiscontinuityCount","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
                "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
                "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
                "AccumulatedDeltaRangeUncertaintyMeters","CarrierFrequencyHz","CarrierCycles","CarrierPhase",\
                "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","ConstellationType","AgcDb", "BasebandCn0DbHz",\
                "FullInterSignalBiasNanos","FullInterSignalBiasUncertaintyNanos","SatelliteInterSignalBiasNanos",\
                "SatelliteInterSignalBiasUncertaintyNanos", "CodeType", "ChipsetElapsedRealtimeNanos"]

keys_old   = ["Raw", "timestamp", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", \
                "BiasUncertaintyNanos", "DriftNanosPerSecond","DriftUncertaintyNanosPerSecond",\
                "HardwareClockDiscontinuityCount","Svid","TimeOffsetNanos","State","ReceivedSvTimeNanos", \
                "ReceivedSvTimeUncertaintyNanos","Cn0DbHz","PseudorangeRateMetersPerSecond",\
                "PseudorangeRateUncertaintyMetersPerSecond","AccumulatedDeltaRangeState","AccumulatedDeltaRangeMeters",\
                "AccumulatedDeltaRangeUncertaintyMeters","CarrierFrequencyHz","CarrierCycles","CarrierPhase",\
                "CarrierPhaseUncertainty","MultipathIndicator","SnrInDb","ConstellationType","AgcDb"]

# =====================================================================================================================


class LogReader():

    def __init__(self, manufacturer, device, filepath:str, specifiedTags=[], mode='logger'):
        
        self.manufacturer = manufacturer
        self.device = device
        self.specifiedTags = specifiedTags
        self.mode = mode

        self.load(filepath)

        self.GnssClockBias = np.nan
        self.pseudorangesDict = {}

        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def load(self, filepath:str):

        self.resetGlobal()
        
        fix = []
        raw = []
        health = []
        motion = []
        env = []

        with open(filepath) as file:
            i = 0
            for line in file:
                line = file.readline().strip().split(",")

                if self.specifiedTags and line[0] not in self.specifiedTags:
                    continue

                match line[0]:
                    case "Raw":
                        mdict = self.getRawDictionnary(line, self.mode)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Raw'")
                        else:
                            raw.append(mdict)
                    
                    case "Fix":
                        mdict = self.getFixDictionnary(line, self.mode)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Fix'")
                        else:
                            fix.append(mdict)

                    case "ECG":
                        mdict = self.getHealthDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'ECG'")
                        else:
                            health.append(mdict)

                    case "PPG":
                        mdict = self.getHealthDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'PPG'")
                        else:
                            health.append(mdict)

                    case "ACC":
                        mdict = self.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'ACC'")
                        else:
                            motion.append(mdict)
                    
                    case "GYR":
                        mdict = self.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'GYR'")
                        else:
                            motion.append(mdict)
                    
                    case "MAG":
                        mdict = self.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'MAG'")
                        else:
                            motion.append(mdict)

                    case "PSR":
                        mdict = self.getEnvironmentDict(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'MAG'")
                        else:
                            env.append(mdict)
                    
                i += 1
        
        # Convert to dataframes
        self.fix = pd.DataFrame(fix)
        self.fix.set_index('datetime', inplace=True)
        self.raw = pd.DataFrame(raw)
        self.raw.set_index('datetime', inplace=True)
        self.health = pd.DataFrame(health)
        self.motion = pd.DataFrame(motion)
        self.env = pd.DataFrame(env)

        # Compute some additional entries
        #self.raw = self.raw.sort_values(by=['prn', 'TimeNanos'])
        dt = self.raw.groupby('prn')['TimeNanos'].diff().values * 1e-9

        doppler = self.raw.groupby('prn')['PseudorangeRateMetersPerSecond'].diff().div(dt, axis=0,)
        self.raw['DopplerError'] = doppler
        
        phases = self.raw.groupby('prn')['AccumulatedDeltaRangeMeters'].diff().div(dt, axis=0,)
        self.raw['PhaseVelocity'] = phases
        self.raw['PhaseError'] = self.raw.groupby('prn')['PhaseVelocity'].diff().div(dt, axis=0,)

        self.raw['PhaseMinusDoppler'] = self.raw['PhaseVelocity'] - self.raw['PseudorangeRateMetersPerSecond']

        #self.raw.replace([np.inf, -np.inf], np.nan, inplace=True)

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def getFixDictionnary(self, line, mode='logger'):
        try: 
            if mode in ['logger', 'mimir']:
                mdict = {
                    "provider" : line[1].upper(),
                    "timestamp": float(line[8])/1e3,
                    'datetime' : np.datetime64(int(line[8]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
                    "latitude" : float(line[2]),
                    "longitude": float(line[3]),
                    "altitude" : float(line[4]) if line[4] != "" else float("nan"),
                }
            elif mode in 'old':
                mdict = {
                    "provider" : line[1].upper(),
                    "timestamp": float(line[7])/1e3,
                    'datetime' : np.datetime64(int(line[7]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
                    "latitude" : float(line[2]),
                    "longitude": float(line[3]),
                    "altitude" : float(line[4]) if line[4] != "" else float("nan"),
                }

        except ValueError:
            return

        return mdict
    
    # -----------------------------------------------------------------------------------------------------------------

    def getRawDictionnary(self, line, mode='logger'):

        if mode in 'mimir':
            keys = keys_mimir
        elif mode in 'old':
            keys = keys_old
        else:
            keys = keys_logger

        try:
            mdict = {}
            i = 0
            for key in keys:
                if line[i] == '':
                    mdict[key] = float("nan")
                else:
                    match key:
                        case "Raw":
                            pass
                        case "timestamp":
                            mdict["timestamp"] = float(line[i])/1e3
                            mdict["datetime"]  = np.datetime64(int(line[i]), 'ms')
                        case "Svid" | "ConstellationType" | "State" | "AccumulatedDeltaRangeState":
                            mdict[key] = int(line[i])
                        case "CodeType":
                            mdict["CodeType"] = line[i]
                        case _:
                            mdict[key] = float(line[i])
                i += 1
            
            prn  = f"{self.getSystemLetter(mdict['ConstellationType'])}{mdict['Svid']:02d}"
            freq = f"{self.getFrequencyLabel(mdict['CarrierFrequencyHz'])}"
            mdict["prn"] = f"{prn}-{freq}"
            mdict["system"] = f"{self.getSystemLetter(mdict['ConstellationType'])}" 
            mdict["frequency"] = freq
            mdict["Pseudorange"], mdict["PseudorangeVelocity"], mdict["PseudorangeAcceleration"] = self.getPseudoranges(mdict)

        except ValueError as e:
            return

        return mdict
    
    # -----------------------------------------------------------------------------------------------------------------

    def getHealthDictionnary(self, line):

        try: 
            mdict = {
                "sensor"         : line[0],
                "timestamp"      : float(line[1])/1e3,
                "datetime"       : np.datetime64(int(line[1]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
                "elapsedRealtime": int(line[2]),
                "accuracy"       : int(line[3])
            }

            i = 0
            for value in line[4:12]:
                mdict[f'value_{i}'] = float(value)
                i += 1
            
        except ValueError:
            return
        
        return mdict

    # -----------------------------------------------------------------------------------------------------------------

    def getMotionDictionnary(self, line):

        try: 
            mdict = {
                "sensor"         : line[0],
                "timestamp"      : float(line[1])/1e3,
                "datetime"       : np.datetime64(int(line[1]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
                "elapsedRealtime": int(line[2]),
                "x"              : float(line[3]),
                "y"              : float(line[4]),
                "z"              : float(line[5])
            }
            
        except ValueError:
            return
        
        return mdict

    # -----------------------------------------------------------------------------------------------------------------

    def getEnvironmentDict(self, line):

        try: 
            mdict = {
                "sensor"         : line[0],
                "timestamp"      : float(line[1])/1e3,
                "datetime"       : np.datetime64(int(line[1]), 'ms'), # datetime.fromtimestamp(float(line[8])/1e3),
                "elapsedRealtime": int(line[2]),
                "value"          : float(line[3])
            }

            
        except ValueError:
            return
        
        return mdict
    
    # -----------------------------------------------------------------------------------------------------------------

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
            
    # -----------------------------------------------------------------------------------------------------------------

    def getFrequencyLabel(self, freq:float):

        if abs(freq - L1_FREQUENCY_HZ) < 1e8:
            return 'L1'
        elif abs(freq - L5_FREQUENCY_HZ) < 1e6:
            return 'L5'
        else:
            print(f'unknown frequecy {freq}')
            return 'LX'

    # -----------------------------------------------------------------------------------------------------------------

    def getPseudoranges(self, raw : dict):

        pseudorange = np.nan
        pseudorangeVelocity = np.nan
        pseudorangeAcceleration = np.nan

        # Check if tracking state correct
        state = raw["State"]
        if not (state & STATE_TOW_DECODED) \
            and not (state & STATE_TOW_KNOWN) \
            and not (state & STATE_GLO_TOD_DECODED)\
            and not (state & STATE_GLO_TOD_KNOWN):
            return pseudorange, pseudorangeVelocity, pseudorangeAcceleration
        
        # if not (state & STATE_2ND_CODE_LOCK)\
        #     and not (state & STATE_GAL_E1C_2ND_CODE_LOCK):
        #     return pseudorange, pseudorangeVelocity, pseudorangeAcceleration

        # Retrieve transmitted time 
        t_tx = raw["ReceivedSvTimeNanos"]

        # Retrieve received time
        if np.isnan(self.GnssClockBias):
            self.GnssClockBias = raw["FullBiasNanos"] + raw["BiasNanos"]
        
        t_rx_gnss = raw["TimeNanos"] + raw["TimeOffsetNanos"] - self.GnssClockBias

        # Transform receive time from GNSS to constellation time
        match(raw["ConstellationType"]):
            case GnssSystems.GPS:
                t_rx = t_rx_gnss % NANOSECONDS_IN_WEEK
            case GnssSystems.GLONASS:
                if (state & STATE_GLO_TOD_DECODED):
                    t_rx = t_rx_gnss % NANOSECONDS_IN_DAY + (3*3600 - LEAP_SECONDS)*1e9
                else:
                    return pseudorange, pseudorangeVelocity, pseudorangeAcceleration
            case GnssSystems.BEIDOU:
                t_rx = (t_rx_gnss % NANOSECONDS_IN_WEEK) - 14*1e9
            case GnssSystems.GALILEO:
                if (state & STATE_GAL_E1C_2ND_CODE_LOCK):
                    #t_rx = t_rx_gnss % NANOSECONDS_IN_100MS
                    t_rx = t_rx_gnss % NANOSECONDS_IN_WEEK
                else:
                    t_rx = t_rx_gnss % NANOSECONDS_IN_WEEK
            case _:
                return pseudorange, pseudorangeVelocity, pseudorangeAcceleration
            
        # Build pseudorange
        pseudorange = (t_rx - t_tx) / 1e9 * SPEED_OF_LIGHT

        # Pseudorange velocity
        pseudorangeVelocity = np.nan
        if raw['prn'] in self.pseudorangesDict:
            dt = abs(raw["timestamp"] - self.pseudorangesDict[raw['prn']]['timestamp'])
            pseudorangeVelocity = pseudorange - self.pseudorangesDict[raw['prn']]['pseudorange']
            pseudorangeVelocity /= dt

            if not np.isnan(self.pseudorangesDict[raw['prn']]['pseudorangeVelocity']):
                pseudorangeAcceleration = pseudorangeVelocity - self.pseudorangesDict[raw['prn']]['pseudorangeVelocity']
                pseudorangeAcceleration /= dt
        
        self.pseudorangesDict[raw['prn']] = {'timestamp':raw["timestamp"], 
                                        'pseudorange':pseudorange, 
                                        'pseudorangeVelocity':pseudorangeVelocity,
                                        'pseudorangeAcceleration':pseudorangeAcceleration}
                
        return pseudorange, pseudorangeVelocity, pseudorangeAcceleration
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def resetGlobal(self):
        self.GnssClockBias = np.nan
        self.pseudorangesDict = {}
    
# =====================================================================================================================

class PosReader():

    def __init__(self, filepath:str):

        self.df = []

        self.load(filepath)

        return
    
    def load(self, filepath:str):
        
        pos = []
        with open(filepath) as file:
            i = 0
            for line in file:
                if line[0] == '%':
                    continue

                mdict = self.getPosDictionnary(line)
                if mdict is None:
                    print(f"Warning: Line {i} skipped with invalid values for 'Pos'")
                else:
                    pos.append(mdict)
            
                i += 1
        
        # Convert to dataframes
        self.df = pd.DataFrame(pos)
        self.df.set_index('datetime', inplace=True)
        
        # print(self.pos)

        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def getPosDictionnary(self, line):
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

class RinexReader():
        
    def __init__(self, filepath:str, tlim, meas, sampling):

        self.df = []
        
        # load file
        obs = gr.load(filepath, tlim=tlim, meas=meas)
        self.df = obs.to_dataframe().dropna(how='all').reset_index().set_index('time')

        # errors 
        #dt = self.df.groupby('sv')['time'].diff().values

        for meas in meas:
            match(meas[0]):
                case 'C' | 'L': 
                    self.df[f"{meas}_rate"] = self.df.groupby('sv')[meas].diff().div(sampling, axis=0,)
                    self.df[f"{meas}_error"] = self.df.groupby('sv')[f"{meas}_rate"].diff().div(sampling, axis=0,)
                case 'D':
                    self.df[f"{meas}_error"] = self.df.groupby('sv')[meas].diff().div(sampling, axis=0,)

        return
    
# =====================================================================================================================

if __name__ == "__main__":
    
    #filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
    #filepath = "./.data/log_old_20230414152332.txt"
    # filepath = "./.data/log_mimir_20230715122058.txt"
    # log = LogReader(filepath)

    # filepath = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
    # log = PosReader(filepath)

    filepath = './.data/2023_Dataset_Hervanta/S2_dynamic_campus/_reference/rover/NMND17420010S_2023-08-01_08-14-05.23O'
    rinex = RinexReader(filepath, tlim=None, meas=['C1C', 'C5Q', 'C2I', 'C5P'], sampling=1)
