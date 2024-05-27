
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from logparser import LogReader, PosReader, RinexReader

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
    1  : ['S1',             'A01', 'SWINGING',       '',   'POCKET', 'SWINGING', 'SWINGING',         ''], 
    2  : ['S1',             'A02',  'TEXTING',       '',   'POCKET',  'TEXTING',  'TEXTING',         ''], 
    3  : ['S1',             'A03', 'SWINGING',       '', 'SWINGING',   'POCKET', 'SWINGING',         ''], 
    4  : ['S1',             'A04',  'TEXTING',       '',  'TEXTING',   'POCKET',  'TEXTING',         ''], 
    5  : ['S1',             'A05', 'SWINGING',       '', 'SWINGING',   'POCKET',         '', 'SWINGING'], 
    6  : ['S1',             'A06',  'TEXTING',       '',  'TEXTING',   'POCKET',         '',  'TEXTING'], 
    7  : ['S1',             'A07', 'SWINGING', 'POCKET',   'POCKET', 'SWINGING', 'SWINGING', 'SWINGING'], 
    8  : ['S1',             'A08',  'TEXTING', 'POCKET',   'POCKET',  'TEXTING',  'TEXTING',  'TEXTING'], 
    9  : ['S1',             'A09', 'SWINGING', 'POCKET', 'SWINGING',   'POCKET', 'SWINGING', 'SWINGING'], 
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

# ======================================================================================================================

def load_files(folder_path, acq_list, device_android, device_uliss, mode=None, indoor_only=False, survey=''):
    log_dict = {}
    # Android devices
    for acq in acq_list:
        for device in device_android:
            filepath = f"{folder_path}/{acq}/{device}/Raw.csv"
            if not os.path.isfile(filepath):
                #print(f"File not found for {acq} {device}")
                continue

            if mode:
                _mode = survey_mode.loc[(survey_mode['survey'] == survey) & (survey_mode['acquisition'] == acq)][device].iloc[0]
                if mode not in _mode:
                    continue
            log = LogReader(manufacturer="", device="", acronym=device, specifiedTags='Raw', mode="old", filepath=filepath)
            
            if indoor_only:
                times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
                stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
                log.raw = log.raw.loc[(log.raw['timestamp'] > start_time) & (log.raw['timestamp'] < stop_time)]
            
            if device in log_dict:
                log_dict[device] = pd.concat([log_dict[device], log.raw], ignore_index=True, sort=False)
            else:
                log_dict[device] = log.raw
    
    # ULISS devices
    measurements_RINEX = sum([[f"{y}{x}" for y in ['S']] for x in ['1C', '2L', '2S', '2C', '2I', '7Q', '7I']], []) # '2L', '2S', '2C', '2I', '7Q', '7I'
    for acq in acq_list:
        for device in device_uliss:
            filepath = f"{folder_path}/{acq}/{device}/gnss.rnx"
            if not os.path.isfile(filepath):
                #print(f"File not found for {acq} {device}")
                continue

            if mode:
                _mode = survey_mode.loc[(survey_mode['survey'] == survey) & (survey_mode['acquisition'] == acq)][device].iloc[0]
                if mode not in _mode:
                    continue

            if indoor_only:
                times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                log = RinexReader(device, filepath, tlim=[times[0], times[1]], meas=measurements_RINEX, sampling=0.2)
            else:
                log = RinexReader(device, filepath, tlim=[], meas=measurements_RINEX, sampling=0.2)
            if device in log_dict:
                log_dict[device] = pd.concat([log_dict[device], log.df], ignore_index=True, sort=False)
            else:
                log_dict[device] = log.df
    
    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def plotBoxPlotCN0(log_dict, device_android, device_uliss):
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
                order = device_android + device_uliss, hue_order=['L1', 'L2', 'L5'], whis=(0, 100), gap=.1)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("C/N0 [dB-Hz]")

    return
