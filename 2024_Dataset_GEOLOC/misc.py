
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import gps_time
import pymap3d as pm
import scipy 

import matplotlib
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import pandas as pd

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from logparser import LogReader, PosReader, RinexReader, MatReader
import logparser as lp
# ======================================================================================================================

LEAP_SECONDS = 18

PALETTE_COLOR = {"TEXTING": "#779ECB", "SWINGING": "#FC8EAC", "POCKET":"#50C878"}

PALETTE_COLOR_DEVICE = {"UA (HYB)":"#1f77b4", 
                        "GP7 (GPS)": "#ff7f0e", 
                        "GPW (GPS)": "#2ca02c", 
                        "GPW (FUSED)":"#d62728",
                        "SW6 (GPS)" : "#9467bd", 
                        "A52 (GPS)" : "#8c564b",
                        "AWINDA"    : "#e377c2"}

# ======================================================================================================================
# Survey, acquisition, indoor 
indoor_time = pd.DataFrame.from_dict({
    0 : ['S3', 'A1', ("2024-03-14T09:30:45", "2024-03-14T09:38:00")], 
    1 : ['S3', 'A2', ("2024-03-14T09:49:15", "2024-03-14T09:56:30")], 
    2 : ['S3', 'A3', ("2024-03-14T10:04:00", "2024-03-14T10:10:30")],
    3 : ['S3', 'A4', ("2024-03-14T10:17:00", "2024-03-14T10:24:00")],
    }, orient='index', columns=['survey', 'acquisition', 'time'])
time_format = "%Y-%m-%dT%H:%M:%S"

survey_mode = pd.DataFrame.from_dict({
    0  : ['survey', 'acquisition',       'UA',     'U1',      'A52',      'GP7',      'GPW',      'SW6'], 
    1  : ['S1',              'A1', 'SWINGING',       '',   'POCKET', 'SWINGING', 'SWINGING',         ''], 
    2  : ['S1',              'A2',  'TEXTING',       '',   'POCKET',  'TEXTING',  'TEXTING',         ''], 
    3  : ['S1',              'A3', 'SWINGING',       '', 'SWINGING',   'POCKET', 'SWINGING',         ''], 
    4  : ['S1',              'A4',  'TEXTING',       '',  'TEXTING',   'POCKET',  'TEXTING',         ''], 
    5  : ['S1',              'A5', 'SWINGING',       '', 'SWINGING',   'POCKET',         '', 'SWINGING'], 
    6  : ['S1',              'A6',  'TEXTING',       '',  'TEXTING',   'POCKET',         '',  'TEXTING'], 
    7  : ['S1',              'A7', 'SWINGING', 'POCKET',   'POCKET', 'SWINGING', 'SWINGING', 'SWINGING'], 
    8  : ['S1',              'A8',  'TEXTING', 'POCKET',   'POCKET',  'TEXTING',  'TEXTING',  'TEXTING'], 
    9  : ['S1',              'A9', 'SWINGING', 'POCKET', 'SWINGING',   'POCKET', 'SWINGING', 'SWINGING'], 
    10 : ['S1',             'A10',  'TEXTING', 'POCKET',  'TEXTING',   'POCKET',  'TEXTING',  'TEXTING'],
    11 : ['S3',              'A1', 'SWINGING', 'POCKET',   'POCKET', 'SWINGING', 'SWINGING',         ''], 
    12 : ['S3',              'A2',  'TEXTING', 'POCKET',   'POCKET',  'TEXTING',  'TEXTING',  'TEXTING'], 
    13 : ['S3',              'A3', 'SWINGING', 'POCKET', 'SWINGING',   'POCKET', 'SWINGING', 'SWINGING'], 
    14 : ['S3',              'A4',  'TEXTING', 'POCKET',  'TEXTING',   'POCKET',  'TEXTING',  'TEXTING'], 
    15 : ['S3',              'A5', 'SWINGING', 'POCKET',         '',         '',         '',         ''], 
    16 : ['S3',              'A6',  'TEXTING', 'POCKET',  'TEXTING',   'POCKET',  'TEXTING',  'TEXTING'], 
    17 : ['S4',              'A1',  'TEXTING', 'POCKET',   'POCKET',  'TEXTING',  'TEXTING',  'TEXTING'], 
    18 : ['S4',              'A2',  'TEXTING', 'POCKET',  'TEXTING',   'POCKET',  'TEXTING',  'TEXTING'],
    }, orient='index')
survey_mode = survey_mode.rename(columns=survey_mode.iloc[0])
survey_mode = survey_mode.drop(survey_mode.index[0])
survey_mode = survey_mode.reset_index(drop=True)

gps_week = pd.DataFrame.from_dict({
    0  : ['survey', 'acquisition', 'week'], 
    1  : ['S1',              'A1',   2305], 
    2  : ['S1',              'A2',   2305], 
    3  : ['S1',              'A3',   2305], 
    4  : ['S1',              'A4',   2305], 
    5  : ['S1',              'A5',   2305], 
    6  : ['S1',              'A6',   2305], 
    7  : ['S1',              'A7',   2306], 
    8  : ['S1',              'A8',   2306], 
    9  : ['S1',              'A9',   2306], 
    10 : ['S1',             'A10',   2306],
    11 : ['S3',              'A1',   2305], 
    12 : ['S3',              'A2',   2305], 
    13 : ['S3',              'A3',   2305], 
    14 : ['S3',              'A4',   2305], 
    15 : ['S3',              'A5',   2305], 
    16 : ['S3',              'A6',   2305], 
    17 : ['S4',              'A1',   2305], 
    18 : ['S4',              'A2',   2305],
    }, orient='index')
gps_week = gps_week.rename(columns=gps_week.iloc[0])
gps_week = gps_week.drop(gps_week.index[0])
gps_week = gps_week.reset_index(drop=True)

# ======================================================================================================================

def load_raw(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    log_dict = {}

    # Android devices
    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                filepath = f"{folder_path}/{acq}/{device}/Raw.csv"
                if not os.path.isfile(filepath):
                    #print(f"File not found for {acq} {device}")
                    continue

                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue
                log = LogReader(manufacturer="", device="", acronym=device, specifiedTags='Raw', mode="old", filepath=filepath)
                
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
                    stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
                    log.raw = log.raw.loc[(log.raw['timestamp'] > start_time) & (log.raw['timestamp'] < stop_time)]
                
                log.raw['mode'] = _mode
                log.raw['acquisition'] = acq
                
                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.raw], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.raw


    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def load_rinex(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    log_dict = {}

    measurements_RINEX = sum([[f"{y}{x}" for y in ['S']] for x in ['1C', '2L', '2S', '2C', '2I', '7Q', '7I']], []) # '2L', '2S', '2C', '2I', '7Q', '7I'
    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                # Check if correct mode
                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue

                filepath = f"{folder_path}/{acq}/{device}/gnss.rnx"
                if not os.path.isfile(filepath):
                    #print(f"File not found for {acq} {device}")
                    continue

                # Filter by indoor time only
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) \
                                            & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    log = RinexReader(device, filepath, tlim=[times[0], times[1]], meas=measurements_RINEX, sampling=0.2)
                else:
                    log = RinexReader(device, filepath, tlim=[], meas=measurements_RINEX, sampling=0.2)

                log.df['mode'] = _mode
                
                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.df], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.df

    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def load_mat(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    log_dict = {}

    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                # Check if correct mode
                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue
                
                filepath = f"{folder_path}/{acq}/{device}/gnss.mat"
                if not os.path.isfile(filepath):
                    #print(f"File not found for {acq} {device}")
                    continue

                log = MatReader(device, filepath)

                # Convert time
                log.df.dropna(inplace=True)
                week = gps_week.loc[(gps_week['survey'] == survey) & (gps_week['acquisition'] == acq_list[0])]['week'].iloc[0]
                log.df['datetime'] = log.df.apply(lambda x: gps_time.GPSTime(week, x['time']-LEAP_SECONDS).to_datetime().timestamp(), axis=1) 
                
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    start_time = datetime.strptime(times[0], time_format).timestamp()
                    stop_time = datetime.strptime(times[1], time_format).timestamp()
                    log.df = log.df.loc[(log.df['datetime'] > start_time) & (log.df['datetime'] < stop_time)]

                log.df['mode'] = _mode

                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.df], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.df

    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def load_mat_cn0(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    pd.options.mode.chained_assignment = None  # default='warn'

    log_dict = {}

    columns = ['tow']
    columns += [f'G{i:02d}' for i in range(1,33)]
    columns += [f'R{i:02d}' for i in range(1,27)]
    columns += [f'E{i:02d}' for i in range(1,37)]
                

    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                # Check if correct mode
                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue
                
                filepath = f"{folder_path}/{acq}/{device}/gnss.mat"
                if not os.path.isfile(filepath):
                    #print(f"File not found for {acq} {device}")
                    continue

                mat = scipy.io.loadmat(filepath)

                df = pd.DataFrame()
                # L1 
                _df = np.concatenate((mat['GNSS']['time'][0][0], mat['GNSS']['snr_1'][0][0]), axis=1)
                _df = pd.DataFrame(_df, columns=columns)
                _df = _df.melt(id_vars=['tow'], var_name='prn', value_name='snr').dropna()
                _df['frequency'] = 'L1'
                df = pd.concat([df, _df],  axis=0)
                df
                # L2/L5
                _df = np.concatenate((mat['GNSS']['time'][0][0], mat['GNSS']['snr_2'][0][0]), axis=1)
                _df = pd.DataFrame(_df, columns=columns)
                _df = _df.melt(id_vars=['tow'], var_name='prn', value_name='snr').dropna()
                _df['frequency'] = 'L1'
                _df['frequency'].loc[_df["prn"].str.contains('G')] = 'L2'
                _df['frequency'].loc[_df["prn"].str.contains('R')] = 'L2'
                _df['frequency'].loc[_df["prn"].str.contains('E')] = 'L5'
                log = pd.concat([df, _df],  axis=0)

                # Convert time
                log.dropna(inplace=True)
                week = gps_week.loc[(gps_week['survey'] == survey) & (gps_week['acquisition'] == acq_list[0])]['week'].iloc[0]
                log['datetime'] = log.apply(lambda x: gps_time.GPSTime(week, x['tow']-LEAP_SECONDS).to_datetime().timestamp(), axis=1) 
                
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    start_time = datetime.strptime(times[0], time_format).timestamp()
                    stop_time = datetime.strptime(times[1], time_format).timestamp()
                    log = log.loc[(log['datetime'] > start_time) & (log['datetime'] < stop_time)]

                log['mode']= _mode

                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log

    return log_dict


# ----------------------------------------------------------------------------------------------------------------------

def load_fix(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    log_dict = {}

    # Android devices
    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                filepath = f"{folder_path}/{acq}/{device}/Fix.csv"
                if not os.path.isfile(filepath):
                    #print(f"File not found for {acq} {device}")
                    continue

                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue
                log = LogReader(manufacturer="", device="", acronym=device, specifiedTags='Fix', mode="old", filepath=filepath)
                
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
                    stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
                    log.fix = log.fix.loc[(log.fix['timestamp'] > start_time) & (log.fix['timestamp'] < stop_time)]
                
                log.fix['mode'] = _mode
                log.fix['acquisition'] = acq

                log.fix.reset_index(inplace=True)
                
                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.fix], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.fix


    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def load_light(folder_path, acq_list, device_list, mode=['TEXTING', 'SWINGING', 'POCKET'], indoor_only=False, survey=''):

    log_dict = {}

    # Android devices
    for acq in acq_list:
        for device in device_list:
            for _mode in mode:
                if not checkAcquisitionMode(survey, acq, device, _mode):
                    continue
                
                processing_types = ['Hyb', 'Light', 'Pdr']
                for _type in processing_types:
                    filepath = f"{folder_path}/{acq}/{device}/pos_{_type.lower()}.csv"
                    if not os.path.isfile(filepath):
                        #print(f"File not found for {acq} {device}")
                        continue
                    
                    log = pd.read_csv(filepath)

                    log.rename(columns={f'{_type}_TOW':'tow',
                                        f'{_type}_lat':'latitude', 
                                        f'{_type}_lon':'longitude', 
                                        f'{_type}_alt':'altitude'}, inplace=True)

                    # Convert time
                    log.dropna(inplace=True)
                    week = gps_week.loc[(gps_week['survey'] == survey) & (gps_week['acquisition'] == acq_list[0])]['week'].iloc[0]
                    log['datetime'] = log.apply(lambda x: gps_time.GPSTime(week, x['tow']-LEAP_SECONDS).to_datetime(), axis=1) 

                    # if indoor_only:
                    #     times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    #     start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
                    #     stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
                    #     log.fix = log.fix.loc[(log.fix['timestamp'] > start_time) & (log.fix['timestamp'] < stop_time)]
                    
                    log['provider'] = _type.upper()
                    log['mode'] = _mode
                    log['acquisition'] = acq
                    
                    if device in log_dict:
                        log_dict[device] = pd.concat([log_dict[device], log], ignore_index=True, sort=False)
                    else:
                        log_dict[device] = log


    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def load_awinda(folder_path, acq_list, device_list, survey=''):

    log_dict = {}

    # Android devices
    for acq in acq_list:
        for device in device_list:
                
            filepath = f"{folder_path}/{acq}/{device}/pos_awinda_60hz.csv"
            if not os.path.isfile(filepath):
                #print(f"File not found for {acq} {device}")
                continue
            
            log = pd.read_csv(filepath)

            log.rename(columns={f'Awinda_TOW':'tow',
                                f'Awinda_lat':'latitude', 
                                f'Awinda_lon':'longitude', 
                                f'Awinda_alt':'altitude'}, inplace=True)

            # Convert time
            log.dropna(inplace=True)
            week = gps_week.loc[(gps_week['survey'] == survey) & (gps_week['acquisition'] == acq_list[0])]['week'].iloc[0]
            log['datetime'] = log.apply(lambda x: gps_time.GPSTime(week, x['tow']-LEAP_SECONDS).to_datetime(), axis=1) 

            # if indoor_only:
            #     times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
            #     start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
            #     stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
            #     log.fix = log.fix.loc[(log.fix['timestamp'] > start_time) & (log.fix['timestamp'] < stop_time)]
            
            log['provider'] = "AWINDA"
            #log['mode'] = _mode
            log['acquisition'] = acq
            
            if device in log_dict:
                log_dict[device] = pd.concat([log_dict[device], log], ignore_index=True, sort=False)
            else:
                log_dict[device] = log


    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def getENUErrors(log_dict, device_android, device_uliss, acq_list, provider_uliss, provider_android):

    pd.options.mode.chained_assignment = None  # default='warn'

    df_diff = pd.DataFrame()
    for acq in acq_list:
        log_uliss = log_dict[device_uliss].loc[(log_dict[device_uliss]['provider'] == provider_uliss) 
                                       & (log_dict[device_uliss]['acquisition'] == acq)]
        log_uliss.set_index('datetime', inplace=True)

        ref_enu = [log_uliss['latitude'].mean(), log_uliss['longitude'].mean(), log_uliss['altitude'].mean()]
        log_uliss[["east", "north", "up"]] = log_uliss.apply(
            lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
            axis='columns', result_type='expand')

        for device in device_android:
            for provider in provider_android[device]:
                log_android = log_dict[device].loc[(log_dict[device]['provider'] == provider) 
                                                & (log_dict[device]['acquisition'] == acq)]
                
                if log_android.empty:
                    continue

                log_android = log_android.iloc[5:]
                log_android.set_index('datetime', inplace=True)
                
                log_android[["east", "north", "up"]] = log_android.apply(
                    lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
                    axis='columns', result_type='expand')

                pos_A, pos_B = log_uliss[["east", "north", "up"]].align(log_android[["east", "north", "up"]])
                log_diff = pos_B.interpolate(method='time') - pos_A.interpolate(method='time')
                log_diff.dropna(how='all', inplace=True)

                log_diff[["2D_error"]] = log_diff.apply(
                    lambda row: getHorizontalError(row['east'], row['north']), 
                    axis='columns', result_type='expand')
                
                log_diff[["3D_error"]] = log_diff.apply(
                    lambda row: get3DError(row['east'], row['north'], row['up']), 
                    axis='columns', result_type='expand')
                
                log_diff["Device"] = f"{device} ({provider})"
                df_diff = pd.concat([df_diff, log_diff])

    return df_diff

# ----------------------------------------------------------------------------------------------------------------------

def selectMode(log_dict_list, mode):

    log_dict = {}
    for _log_dict in log_dict_list:
        log_dict.update(_log_dict)

    for device, log in log_dict.items():
        log_dict[device] = log[log['mode'].isin(mode)]

    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def selectValidSatellites(df, check_phase=False):

    # Check bit flags
    df = df[(df['ConstellationType'] == lp.GnssSystems.GPS) 
                & (df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) 
                    & ((df['State'] & lp.STATE_TOW_DECODED == lp.STATE_TOW_DECODED) 
                    |  (df['State'] & lp.STATE_TOW_KNOWN == lp.STATE_TOW_KNOWN)) |
            (df['ConstellationType'] == lp.GnssSystems.GALILEO) 
                & ((df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) | 
                   (df['State'] & lp.STATE_GAL_E1BC_CODE_LOCK == lp.STATE_GAL_E1BC_CODE_LOCK) |
                   (df['State'] & lp.STATE_GAL_E1C_2ND_CODE_LOCK == lp.STATE_GAL_E1C_2ND_CODE_LOCK)) 
                    & ((df['State'] & lp.STATE_TOW_DECODED == lp.STATE_TOW_DECODED) 
                    |  (df['State'] & lp.STATE_TOW_KNOWN == lp.STATE_TOW_KNOWN)) |
            (df['ConstellationType'] == lp.GnssSystems.GLONASS) 
                & (df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) 
                    & ((df['State'] & lp.STATE_GLO_TOD_DECODED == lp.STATE_GLO_TOD_DECODED) \
                    |  (df['State'] & lp.STATE_GLO_TOD_KNOWN == lp.STATE_GLO_TOD_KNOWN))]

    return df


# ----------------------------------------------------------------------------------------------------------------------

def selectValidSatellitesPhases(df, check_phase=False):

    # Check bit flags
    df = df[(df['AccumulatedDeltaRangeState'] & lp.ADR_STATE_VALID == lp.ADR_STATE_VALID) 
          & (df['AccumulatedDeltaRangeState'] & lp.ADR_STATE_CYCLE_SLIP != lp.ADR_STATE_CYCLE_SLIP) 
          & (df['AccumulatedDeltaRangeState'] & lp.ADR_STATE_RESET != lp.ADR_STATE_RESET)]

    return df

# ----------------------------------------------------------------------------------------------------------------------

def checkAcquisitionMode(survey, acquisition, device, mode):

    _mode = survey_mode.loc[(survey_mode['survey'] == survey) & (survey_mode['acquisition'] == acquisition)][device].iloc[0]

    if _mode == mode:
        return True
    else:
        return False

# ----------------------------------------------------------------------------------------------------------------------

def plotBoxPlotCN0PerFrequency(log_dict, device_android, device_uliss):
    pd.options.mode.chained_assignment = None  # default='warn'
    # suppose df1 and df2 are dataframes, each with the same 10 columns
    df = pd.DataFrame()
    for device, log in log_dict.items():
        if device in device_android:
            _df = log.reset_index()[['Cn0DbHz', 'frequency']]
        if device in device_uliss:
            _df = log[['snr', 'frequency']]
            _df.rename(columns={'snr':'Cn0DbHz'}, inplace=True)
        _df['device'] = device

        df = pd.concat([df, _df], axis=0)
    df.rename(columns={'frequency':'Frequency'}, inplace=True)

    plt.figure(figsize=(4,3))
    sns.boxplot(data=df, x='device', y='Cn0DbHz', hue='Frequency', 
                order = device_android + device_uliss, hue_order=['L1', 'L2', 'L5'], whis=(0, 100), gap=.1, legend=False)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("C/N0 [dB-Hz]")

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotBoxPlotCN0PerMode(log_dict, device_android, device_uliss, mode=['TEXTING', 'SWINGING', 'POCKET']):

    pd.options.mode.chained_assignment = None  # default='warn'
    # suppose df1 and df2 are dataframes, each with the same 10 columns
    df = pd.DataFrame()
    for device, log in log_dict.items():
        if device in device_android:
            _df = log.reset_index()[['Cn0DbHz', 'mode']]
        if device in device_uliss:
            _df = log[['snr', 'mode']]
            _df.rename(columns={'snr':'Cn0DbHz'}, inplace=True)
        _df['device'] = device

        df = pd.concat([df, _df], axis=0)

    plt.figure(figsize=(4,3))
    ax = sns.boxplot(data=df, x='device', y='Cn0DbHz', hue='mode', order = device_android + device_uliss, 
                hue_order=mode, whis=(0, 100), gap=.1, legend=True,
                palette=PALETTE_COLOR)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("C/N0 [dB-Hz]")

    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotBoxPlotSignalsPerMode(log_dict, device_android, device_uliss, mode=['TEXTING', 'SWINGING', 'POCKET']):

    df = pd.DataFrame()
    for device, log in log_dict.items():
        if device in device_android:
            _df = log[['TimeNanos', 'prn', 'acquisition', 'mode']]
            _df = _df.groupby(['acquisition', 'TimeNanos', 'mode']).nunique()
            _df = _df.reset_index().drop(columns=['acquisition', 'TimeNanos'])
        
        if device in device_uliss:
            _df = log[['num_sat', 'mode']]
            _df.rename(columns={'num_sat':'prn'}, inplace=True)

        _df['device'] = device
        df = pd.concat([df, _df], axis=0)

    plt.figure(figsize=(4,3))
    sns.boxplot(data=df, x='device', y='prn', hue='mode', order = device_android + device_uliss, 
                hue_order=mode, whis=(0, 100), gap=.1, legend=False,
                palette=PALETTE_COLOR)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("Number of signals")

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotBarSignalsPerMode(log_dict, device_android, device_uliss, mode=['TEXTING', 'SWINGING', 'POCKET']):

    df = pd.DataFrame()
    for device, log in log_dict.items():
        if device in device_android:
            _df = log[['TimeNanos', 'prn', 'acquisition', 'mode']]
            _df = _df.groupby(['acquisition', 'TimeNanos', 'mode']).nunique()
            _df = _df.reset_index().drop(columns=['acquisition', 'TimeNanos'])
        
        if device in device_uliss:
            _df = log[['num_sat', 'mode']]
            _df.rename(columns={'num_sat':'prn'}, inplace=True)

        _df['device'] = device
        df = pd.concat([df, _df], axis=0)

    plt.figure(figsize=(4,3))
    ax = sns.barplot(data=df, x='device', y='prn', hue='mode', order = device_android + device_uliss, 
                hue_order=mode, errorbar='sd', legend=True,
                palette=PALETTE_COLOR)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("Number of signals")

    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)



    return 

# ----------------------------------------------------------------------------------------------------------------------

def plotBarSatellitesPerMode(log_dict, device_android, device_uliss):

    df = pd.DataFrame()
    for device, log in log_dict.items():
        if device in device_android:
            _df = log[['TimeNanos', 'sv', 'acquisition', 'mode']]
            _df = _df.groupby(['acquisition', 'TimeNanos', 'mode']).nunique()
            _df = _df.reset_index().drop(columns=['acquisition', 'TimeNanos'])
        
        if device in device_uliss:
            continue

        _df['device'] = device
        df = pd.concat([df, _df], axis=0)

    plt.figure(figsize=(4,3))
    sns.barplot(data=df, x='device', y='sv', hue='mode', order = device_android + device_uliss, 
                hue_order=['TEXTING', 'SWINGING', 'POCKET'], errorbar='sd', legend=False)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("Number of signals")

    return 

# ----------------------------------------------------------------------------------------------------------------------

def plotMap(locations, extent, scale, marker='', markersize=1, figsize=(8,4)):
    """
    Taken from: https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    Mapping New York City Open Street Map (OSM) with Cartopy
    This code uses a spoofing algorithm to avoid bounceback from OSM servers
    
    """

    matplotlib.rcParams.update({'font.size': 10})

    def image_spoof(self, tile): # this function pretends not to be a Python script
        url = self._image_url(tile) # get the url of the street map API
        req = Request(url) # start request
        req.add_header('User-agent','Anaconda 3') # add user agent to request
        fh = urlopen(req) 
        im_data = io.BytesIO(fh.read()) # get image
        fh.close() # close url
        img = Image.open(im_data) # open image with PIL
        img = img.convert(self.desired_tile_form) # set image format
        return img, self.tileextent(tile), 'lower' # reformat for cartopy

    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map

    fig = plt.figure(figsize=figsize) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_extent(extent) # set extents

    ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification

    # Polylines
    for label, loc in locations.items():
        ax1.plot(loc['longitude'].to_list(), loc['latitude'].to_list(),
                 linewidth=2, marker=marker, markersize=markersize, transform=ccrs.Geodetic(), label=label)
    
    # Grid
    # gl = ax1.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # gl.top_labels = False
    # gl.right_labels = False

    ax1.set_xticks(np.linspace(extent[0],extent[1],5),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    # ax1.xaxis.set_tick_params(labelsize=14)
    # ax1.yaxis.set_tick_params(labelsize=14)

    plt.legend()
    plt.grid(False)

    matplotlib.rcParams.update({'font.size': 12})

# ----------------------------------------------------------------------------------------------------------------------

def plotEN(log_dict, device_android, device_uliss, acq, provider_uliss, provider_android):
    
    pd.options.mode.chained_assignment = None  # default='warn'

    log_uliss = log_dict[device_uliss].loc[(log_dict[device_uliss]['provider'] == provider_uliss) 
                                    & (log_dict[device_uliss]['acquisition'] == acq)]
    log_uliss.set_index('datetime', inplace=True)

    ref_enu = [log_uliss['latitude'].mean(), log_uliss['longitude'].mean(), log_uliss['altitude'].mean()]
    log_uliss[["east", "north", "up"]] = log_uliss.apply(
        lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
        axis='columns', result_type='expand')
    
    plt.figure(figsize=(8,6))
    plt.plot(log_uliss['east'], log_uliss['north'])

    for device in device_android:
        log_android = log_dict[device].loc[(log_dict[device]['provider'] == provider_android) 
                                            & (log_dict[device]['acquisition'] == acq)]

        log_android = log_android.iloc[5:]
        log_android.set_index('datetime', inplace=True)
        
        log_android[["east", "north", "up"]] = log_android.apply(
            lambda row: convert2ENU(row['latitude'], row['longitude'], row['altitude'], ref_enu), 
            axis='columns', result_type='expand')
        
        plt.plot(log_android['east'], log_android['north'])

    plt.grid()
    plt.axis('equal')

    pd.options.mode.chained_assignment = 'warn' # default='warn'

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotECDF(log_diff):

    device_order = [*PALETTE_COLOR_DEVICE][1:]

    plt.figure(figsize=(6,4))
    sns.ecdfplot(log_diff, x='2D_error', stat='proportion', hue='Device', hue_order=device_order,
                 palette=PALETTE_COLOR_DEVICE)
    plt.grid()
    plt.xlim((0,50))
    plt.xlabel("Horizontal error [m]")

    return 

# ----------------------------------------------------------------------------------------------------------------------

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