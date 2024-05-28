
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import pandas as pd
import datetime

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from logparser import LogReader, PosReader, RinexReader, MatReader
import logparser as lp

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

                log.df['mode'] = _mode

                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.df], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.df

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

                #log.fix = log.fix.loc[log.fix['provider'].isin(['FUSED'])]
                
                if indoor_only:
                    times = indoor_time.loc[(indoor_time['survey'] == survey) & (indoor_time['acquisition'] == acq)]['time'].iloc[0]
                    start_time = datetime.strptime(times[0], time_format).timestamp() + 3600*2
                    stop_time = datetime.strptime(times[1], time_format).timestamp() + 3600*2
                    log.fix = log.fix.loc[(log.fix['timestamp'] > start_time) & (log.fix['timestamp'] < stop_time)]
                
                log.fix['mode'] = _mode
                log.fix['acquisition'] = acq
                
                if device in log_dict:
                    log_dict[device] = pd.concat([log_dict[device], log.fix], ignore_index=True, sort=False)
                else:
                    log_dict[device] = log.fix


    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

def selectMode(log_dict_list, mode):

    log_dict = {}
    for _log_dict in log_dict_list:
        log_dict.update(_log_dict)

    for device, log in log_dict.items():
        log_dict[device] = log[log['mode'].isin(mode)]

    return log_dict

# ----------------------------------------------------------------------------------------------------------------------

# def selectValidSatellites(df, check_phase=False):

#     # Check bit flags
#     df = df[(df['ConstellationType'] == lp.GnssSystems.GPS) 
#                 & (df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) 
#                     & ((df['State'] & lp.STATE_TOW_DECODED == lp.STATE_TOW_DECODED) 
#                     |  (df['State'] & lp.STATE_TOW_KNOWN == lp.STATE_TOW_KNOWN)) |
#             (df['ConstellationType'] == lp.GnssSystems.GALILEO) 
#                 & (df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) 
#                     & ((df['State'] & lp.STATE_TOW_DECODED == lp.STATE_TOW_DECODED) 
#                     |  (df['State'] & lp.STATE_TOW_KNOWN == lp.STATE_TOW_KNOWN)) |
#             (df['ConstellationType'] == lp.GnssSystems.GLONASS) 
#                 & (df['State'] & lp.STATE_CODE_LOCK == lp.STATE_CODE_LOCK) 
#                     & ((df['State'] & lp.STATE_GLO_TOD_DECODED == lp.STATE_GLO_TOD_DECODED) \
#                     |  (df['State'] & lp.STATE_GLO_TOD_KNOWN == lp.STATE_GLO_TOD_KNOWN))]

#     return df


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
                order = device_android + device_uliss, hue_order=['L1', 'L2', 'L5'], whis=(0, 100), gap=.1)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("C/N0 [dB-Hz]")

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotBoxPlotCN0PerMode(log_dict, device_android, device_uliss):
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
    sns.boxplot(data=df, x='device', y='Cn0DbHz', hue='mode', order = device_android + device_uliss, 
                hue_order=['TEXTING', 'SWINGING', 'POCKET'], whis=(0, 100), gap=.1)
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("C/N0 [dB-Hz]")

    return

# ----------------------------------------------------------------------------------------------------------------------

def plotBarSignalsPerMode(log_dict, device_android, device_uliss):

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
    sns.barplot(data=df, x='device', y='prn', hue='mode', order = device_android + device_uliss, 
                hue_order=['TEXTING', 'SWINGING', 'POCKET'], errorbar='sd')
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("Number of signals")


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
                hue_order=['TEXTING', 'SWINGING', 'POCKET'], errorbar='sd')
    plt.ylim((0, 60))
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Device")
    plt.ylabel("Number of signals")

    return 

# ----------------------------------------------------------------------------------------------------------------------

def plotMap(locations, scale, marker='', markersize=1):
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

    fig = plt.figure(figsize=(8,8)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    #ax1.set_extent(extent) # set extents

    ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification

    # Polylines
    for label, loc in locations.items():
        ax1.plot(loc['longitude'].to_list(), loc['latitude'].to_list(),
                 linewidth=2, marker=marker, markersize=markersize, transform=ccrs.Geodetic(), label=label)
    
    # Grid
    # gl = ax1.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # gl.top_labels = False
    # gl.right_labels = False

    #ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    #ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    # ax1.xaxis.set_tick_params(labelsize=14)
    # ax1.yaxis.set_tick_params(labelsize=14)

    plt.legend()
    plt.grid(False)

    matplotlib.rcParams.update({'font.size': 12})