
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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


import seaborn as sns

import misc

plt.style.use('plot_style.mplstyle')
        
# ======================================================================================================================
# Position plots

def plotENU(logs, lim, yticks, xticks, mode='reference'):

    # Params
    minor_ticks_east = yticks[0]
    major_ticks_east = yticks[1]
    minor_ticks_north = yticks[2]
    major_ticks_north = yticks[3]
    minor_ticks_up = yticks[4]
    major_ticks_up = yticks[5]
    ylim_east = lim[0]
    ylim_north = lim[1]
    ylim_up = lim[2]

    # Init
    fig, axs = plt.subplots(3, figsize=(6,6), sharex=True)
    plt.suptitle('East / North / Up errors')
    for log in logs:
        if mode == 'reference':
            df = log.fix.loc[log.fix['provider'].isin(['GPS']), ["east", "north", "up"]]
            df.index = [idx - df.index[0] for idx in df.index]
        elif mode == 'difference':
            df = log.diff
            df.index = [idx - df.index[0] for idx in df.index]
        axs[0].plot(df.index.seconds.tolist(), df['east'].tolist(), label=f"{log.device}")
        axs[1].plot(df.index.seconds.tolist(), df['north'].tolist(), label=f"{log.device}")
        axs[2].plot(df.index.seconds.tolist(), df['up'].tolist(), label=f"{log.device}")
        
    # East
    axs[0].set_ylabel("East [m]")
    axs[0].set_ylim(-ylim_east, ylim_east)
    axs[0].yaxis.set_major_locator(MultipleLocator(major_ticks_east))
    axs[0].yaxis.set_major_formatter('{x:.0f}')
    axs[0].yaxis.set_minor_locator(MultipleLocator(minor_ticks_east))

    # East
    axs[1].set_ylabel("North [m]")
    axs[1].set_ylim(-ylim_north, ylim_north)
    axs[1].yaxis.set_major_locator(MultipleLocator(major_ticks_north))
    axs[1].yaxis.set_major_formatter('{x:.0f}')
    axs[1].yaxis.set_minor_locator(MultipleLocator(minor_ticks_north))

    # Set ticks up
    axs[2].set_ylabel("Up [m]")
    axs[2].yaxis.set_major_locator(MultipleLocator(major_ticks_up))
    axs[2].yaxis.set_major_formatter('{x:.0f}')
    axs[2].yaxis.set_minor_locator(MultipleLocator(minor_ticks_up))
    axs[2].set_ylim(-ylim_up, ylim_up)

    
    # X Axis formatter
    axs[2].xaxis.set_minor_locator(MultipleLocator(xticks[0]))
    axs[2].xaxis.set_major_locator(MultipleLocator(xticks[1]))

    def timeTicks(x, pos):                                                                                                                                                                                                                                                         
        d = datetime.timedelta(seconds=x)                                                                                                                                                                                                                                          
        return str(d)                                                                                                                                                                                                                                                              
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)                                                                                                                                                                                                                         
    axs[2].xaxis.set_major_formatter(formatter)

    plt.xlabel('Duration')

    axs[0].margins(x=0)
    axs[1].margins(x=0)
    axs[2].margins(x=0)
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(labels, loc='upper right', ncols=len(labels), bbox_to_anchor=(1.0, 0.95), framealpha=1.0) 

    #fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

def plotEN(logs, lim, ticks):
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs = plt.subplots(1, figsize=(6,6))
    fig.suptitle('East/North errors')
    i = 0
    for log in logs:
        df = log.fix.loc[log.fix['provider'].isin(['GPS']), ["east", "north", "up"]]
        df.plot(x='east', y='north', kind='scatter', label=f"{log.manufacturer} {log.device}",
                color=colors[i], s=6, zorder=3, ax=axs)
        i+=1
    
    plt.grid(zorder=0)
    plt.axis('square')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
        
    fig.tight_layout()

    minor_ticks = ticks[0]
    major_ticks = ticks[1]

    axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
    axs.yaxis.set_major_formatter('{x:.0f}')
    axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))
    axs.xaxis.set_major_locator(MultipleLocator(major_ticks))
    axs.xaxis.set_major_formatter('{x:.0f}')
    axs.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

# ----------------------------------------------------------------------------------------------------------------------

def plotHistENU(logs):

    lim = 5
    nb_bins = 51

    for log in logs:
        fig, axs = plt.subplots(1, figsize=(6,4))
        fig.suptitle(f"Histogram ENU errors ({log.manufacturer} {log.device})")
        pos = log.fix.loc[log.fix['provider'].isin(['GPS']), ["east", "north", "up"]]

        bins = np.linspace(-lim, lim, nb_bins)
        
        bottom = np.zeros(nb_bins-1)
        for label in ['east', 'north', 'up']:
            hist, edges = np.histogram(pos[label], density=True, bins=bins)
            unity_density = hist / hist.sum()
            axs.bar(x=edges[:-1], height=unity_density, align='edge', 
                    width= 0.9 * (bins[1] - bins[0]), label=label.capitalize(), zorder=3, bottom=bottom)
            bottom += unity_density

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

def plotStatisticsENU(logs, lim, ticks, mode='violin'):

    for log in logs:
        
        minor_ticks = ticks[0]
        major_ticks = ticks[1]

        fig, axs = plt.subplots(1, figsize=(4,4))

        axs.set_title(f"{log.manufacturer} {log.device}")
        
        pos = log.diff[["east", "north", "up"]].dropna()
        data = [pos['east'], pos['north'], pos['up']]

        if mode == 'violin':
            axs.violinplot(data, showmeans=False, showmedians=True)
        elif mode == 'box':
            axs.boxplot(data, showmeans=True, showfliers=False)
        
        axs.set_xticks([y + 1 for y in range(len(data))], labels=['East', 'North', 'Up'])

        axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
        axs.yaxis.set_major_formatter('{x:.0f}')
        axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))

        axs.set_ylabel("Error [m]")

        plt.ylim(-lim, lim)

        handles, labels = axs.get_legend_handles_labels()
        fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

def plotMap(locations, extent, scale, marker='', markersize=1):
    """
    Taken from: https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    Mapping New York City Open Street Map (OSM) with Cartopy
    This code uses a spoofing algorithm to avoid bounceback from OSM servers
    
    """

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

    fig = plt.figure(figsize=(6,6)) # open matplotlib figure
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

    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    # ax1.xaxis.set_tick_params(labelsize=14)
    # ax1.yaxis.set_tick_params(labelsize=14)

    plt.legend()
    plt.grid(False)


# ======================================================================================================================
# Measurement plots

def plotHistPerSystem(logs, systems, data_name, ticks, lim, absolute=False):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]
    xlim = lim[0]
    ylim = lim[1]
    nb_bins = 31
    bins = np.linspace(0, xlim, nb_bins)

    for log in logs:
        fig, axs = plt.subplots(1, figsize=(6,4))
        fig.suptitle(f"Histogram pseudorange errors ({log.manufacturer} {log.device})")

        sats = list(set(log.raw["prn"]))
        sats.sort()

        # Find total absolute sum
        _sats = [item for item in sats if item.startswith(systems)]
        df = log.raw.loc[log.raw['prn'].isin(_sats), [data_name]]
        hist, edges = np.histogram(df[data_name], density=False, bins=bins)
        total_sum = hist.sum()

        bottom = np.zeros(nb_bins-1)
        for sys in systems:
            _sats = [item for item in sats if item.startswith(sys)]
            df = log.raw.loc[log.raw['prn'].isin(_sats), [data_name]]

            if absolute:
                df[data_name] = df[data_name].abs()
                bins = np.linspace(0, xlim, nb_bins)
            else:
                bins = np.linspace(-xlim, xlim, nb_bins)
            hist, edges = np.histogram(df[data_name], density=False, bins=bins)
            unity_density = hist / hist.sum() / len(systems)
            axs.bar(x=edges[:-1], height=unity_density, align='center', 
                    width= 0.9 * (bins[1] - bins[0]), label=sys, zorder=3, bottom=bottom)
            bottom += unity_density
        
        axs.xaxis.set_major_locator(MultipleLocator(major_ticks))
        axs.xaxis.set_major_formatter('{x:.0f}')
        axs.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

        plt.ylim(0, ylim)

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

# Plot statistics data in box
def plotStatisticsDataBox(logs, data_name, ylabel, systems, frequencies, lim, ticks, mode='raw'):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]

    for log in logs:

        fig, axs = plt.subplots(1, figsize=(8,5))
        fig.suptitle(f"{log.manufacturer} {log.device}")
        if mode == 'ref':
            fig.suptitle(f"{log.manufacturer} {log.device} (Reference)")
            df = log.ref.df
        elif mode == 'raw':
            fig.suptitle(f"{log.manufacturer} {log.device}")
            df = log.raw

        sats = list(set(df["prn"]))
        sats.sort()
        
        data = []
        labels = []
        for sys in systems:
            _sats = [item for item in sats if item.startswith(sys)]
            if sys == 'R':
                __sats = _sats
                _df = df.loc[df['prn'].isin(__sats), [data_name]]

                _data = _df[data_name]
                data.append(_data[~np.isnan(_data)])
                labels.append(f"{misc.getSystemStr(sys)}")
            else:
                for freq in frequencies:
                    __sats = [item for item in _sats if freq in item]
                    _df = df.loc[df['prn'].isin(__sats), [data_name]]

                    _data = _df[data_name]
                    _data = _data[~np.isnan(_data)].tolist()
                    
                    if len(_data) != 0:
                        data.append(_data)
                    else:
                        data.append([float('nan'), float('nan')])
                    labels.append(f"{misc.getSystemStr(sys)}-{freq}")
        
        meanpointprops = dict(markeredgecolor='#87bc45', markerfacecolor='#87bc45')
        axs.boxplot(data, showmeans=True, meanprops=meanpointprops, showfliers=False)
        axs.set_xticks([y + 1 for y in range(len(data))], labels=labels)
        axs.set_ylabel(ylabel)

        axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
        axs.yaxis.set_major_formatter('{x:.1f}')
        axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))
        
        plt.ylim(-lim, lim)
        axs.set_axisbelow(True)
        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

# Plot statistics data in violin
def plotStatisticsDataViolin(logs, data_name, ylabel, systems, frequencies, lim, ticks, mode='raw'):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]

    for log in logs:

        fig, axs = plt.subplots(1, figsize=(6,5))

        if mode == 'ref':
            fig.suptitle(f"{log.manufacturer} {log.device} (Reference)")
            df = log.ref.df
        elif mode == 'raw':
            fig.suptitle(f"{log.manufacturer} {log.device}")
            df = log.raw

        sats = list(set(df["prn"]))
        sats.sort()
        
        labels = []
        for sys in systems:
            labels.append(f"{misc.getSystemStr(sys)}")

        _sats = [item for item in sats if item.startswith(systems)]
        _sats = [item for item in _sats if item.endswith(tuple([freq[-1] for freq in frequencies]))]
        
        _df = df.loc[df['prn'].isin(_sats), ['prn', 'system', 'frequency', data_name]]

        _df.reset_index(drop=True, inplace=True)

        # Correction for mono-frequencies
        for sys in systems:
            _frequencies = list(set(_df.loc[_df['system'].isin([sys])]['frequency']))
            new_row = {'system':f'{sys}', 'frequency':'L1', data_name:float('nan')}
            
            if 'L1' not in _frequencies:
                new_row = [f'{sys}00-L1', f'{sys}', 'L1', float('nan')]
                _df.loc[len(_df.index)] = new_row
                _df.loc[len(_df.index)] = new_row
            elif 'L5' not in _frequencies:
                new_row = [f'{sys}00-L5', f'{sys}', 'L5', float('nan')]
                _df.loc[len(_df.index)] = new_row
                _df.loc[len(_df.index)] = new_row

        sns.violinplot(ax=axs, data=_df, x='system', y=data_name, hue='frequency', 
                       order=systems, hue_order=frequencies, legend=False,
                       split=True, orient='v', palette=['#27aeef', '#ef9b20'] , zorder=3)
        plt.setp(axs.collections, alpha=.7)

        axs.set_xticks([y for y in range(len(labels))], labels=labels)
        axs.set_xlabel('')
        axs.set_ylabel(ylabel)
        axs.legend(handles=axs.legend_.legend_handles, labels=frequencies)

        axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
        axs.yaxis.set_major_formatter('{x:.0f}')
        axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))

        plt.ylim(0, lim)
        axs.set_axisbelow(True)
        # handles, labels = axs.get_legend_handles_labels()
        # fig.legend(handles, labels=frequencies)
        fig.tight_layout()

        # Reference graph

# ----------------------------------------------------------------------------------------------------------------------

# Plot measurement
def plotMeasurement(log, data_name, sat):

    minor_ticks = 0.2
    major_ticks = 1
    lim = 300

    df = log.raw.loc[log.raw['prn'].isin(sat), ['datetime', 'prn', data_name]]

    fig, axs = plt.subplots(1, figsize=(6,5))
    fig.suptitle(f"{data_name} errors ({log.manufacturer} {log.device})")

    df.groupby('prn')[data_name].plot(x='datetime', y=data_name, ax=axs)

    # axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
    # axs.yaxis.set_major_formatter('{x:.0f}')
    # axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))

    #plt.ylim(-lim, lim)

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()

# ======================================================================================================================
# Visibility

def plotTotalSignalsPerEpochs(logs, lim, ticks, mode='signal'):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]

    if mode == 'signal':
        column = 'prn'
    elif mode == 'satellite':
        column = 'sv'

    fig, axs = plt.subplots(1, figsize=(6,5))
    fig.suptitle(f"Total {mode}s seen per epoch")

    for log in logs:

        df = log.raw[['TimeNanos', column]]
        df = df.groupby('TimeNanos').nunique()
        #df.plot(y='prn', label=log.device, style='o', ms=2, ax=axs)
        time = np.array(df.index.tolist()) * 1e-9
        axs.scatter(time - time[0], df[column].tolist(), label=f"{log.manufacturer} {log.device}", marker='.')

    axs.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
    axs.xaxis.set_major_locator(MultipleLocator(major_ticks))
    def timeTicks(x, pos):                                                                                                                                                                                                                                                         
        d = datetime.timedelta(seconds=x)                                                                                                                                                                                                                                          
        return str(d)                                                                                                                                                                                                                                                              
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)                                                                                                                                                                                                                         
    axs.xaxis.set_major_formatter(formatter)
    axs.set_axisbelow(True)
    axs.legend()
    axs.set_ylim(lim[0], lim[1])
    axs.set_xlabel("Duration")
    fig.tight_layout()

    #axs.margins(x=0)


# ----------------------------------------------------------------------------------------------------------------------

def statsSatellitesPerSystem(logs):

    systems = ['G', 'R', 'E', 'C', 'I', 'S', 'J']
    
    df_stats = pd.DataFrame()
    devices = []
    for log in logs:
        devices.append(log.device)
    df_stats['device'] = devices

    for sys in systems:
        _bars_dev = []
        _bars_ref = []
        for log in logs:
            sat_dev = log.raw[log.raw.prn.str.match(rf'{sys}[0-9]{{2,3}}-L.')]['prn'].nunique()
            sat_ref = log.ref.df[log.ref.df.prn.str.match(rf'{sys}[0-9]{{2,3}}-L.')]['prn'].nunique()
            _bars_dev.append(sat_dev)
            _bars_ref.append(sat_ref)
        df_stats[f"{sys}"] = _bars_dev
        df_stats[f"{sys}_ref"] = _bars_ref
        
    return df_stats


# ----------------------------------------------------------------------------------------------------------------------

def plotBarSatellitesPerSystem(logs):

    width = 0.18

    systems = ['G', 'R', 'E', 'C', 'I', 'S', 'J']
    frequencies = ['L1', 'L5']

    bars_dev = []
    bars_ref = []
    for sys in systems:
        _bars_dev = []
        _bars_ref = []
        for log in logs:
            sat_dev = log.raw[log.raw.prn.str.match(rf'{sys}[0-9]{{2,3}}-L.')]['prn'].nunique()
            sat_ref = log.ref.df[log.ref.df.prn.str.match(rf'{sys}[0-9]{{2,3}}-L.')]['prn'].nunique()
            _bars_dev.append(sat_dev)
            _bars_ref.append(sat_ref)
            
        bars_dev.append(_bars_dev)
        bars_ref.append(_bars_ref)

    # Plot
    fig, axs = plt.subplots(1, figsize=(6,4))
    fig.suptitle(f"Visible satellites per constellations")
    x = np.arange(len(logs))

    colors_ref = ["#82c8f0", "#f5a5c8", "#ffdca5", "#7dcdbe", "#4e008e", "#c3b9d7", "#cf286f"]
    colors_dev = colors_ref
    for i in range(len(bars_dev)):
        if i > 0:
            bottom_dev = np.sum(np.array(bars_dev[:i]), axis=0)
            bottom_ref = np.sum(np.array(bars_ref[:i]), axis=0)
        else:
            bottom_dev = np.zeros(len(logs))
            bottom_ref = np.zeros(len(logs))

        axs.bar(x-0.1, bars_ref[i], width, bottom=bottom_ref, label=misc.getSystemStr(systems[i]), color=colors_ref[i])
        axs.bar(x+0.1, bars_dev[i], width, bottom=bottom_dev, color=colors_dev[i])
    
    devices = []
    for log in logs:
        devices.append(log.device)
    axs.set_xticks(x, devices)
    axs.set_axisbelow(True)

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, systems, loc='upper right', ncols=len(labels), bbox_to_anchor=(1.0, 0.95), framealpha=1.0) 

    #fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.tight_layout()

    return 

# ----------------------------------------------------------------------------------------------------------------------

def statsSatellitesPerFrequency(logs):

    frequencies = ['L1', 'L5']
    
    df_stats = pd.DataFrame()
    devices = []
    for log in logs:
        devices.append(log.device)
    df_stats['device'] = devices

    for freq in frequencies:
        _bars_dev = []
        _bars_ref = []
        for log in logs:
            sat_dev = log.raw[log.raw.prn.str.contains(rf'.[0-9]{{2,3}}-{freq}')]['prn'].nunique()
            sat_ref = log.ref.df[log.ref.df.prn.str.contains(rf'.[0-9]{{2,3}}-{freq}')]['prn'].nunique()
            _bars_dev.append(sat_dev)
            _bars_ref.append(sat_ref)
        df_stats[f"{freq}"] = _bars_dev
        df_stats[f"{freq}_ref"] = _bars_ref
        
    return df_stats

# ----------------------------------------------------------------------------------------------------------------------

def plotBarSatellitesPerFrequency(logs):

    width = 0.18

    #systems = ['G', 'R', 'E', 'C', 'I', 'S', 'J']
    frequencies = ['L1', 'L5']

    fig, axs = plt.subplots(1, figsize=(6,4))
    fig.suptitle(f"Visible satellites per constellations")
    x = np.arange(len(logs))

    bars_dev = []
    bars_ref = []
    for freq in frequencies:
        _bars_dev = []
        _bars_ref = []
        for log in logs:
            sat_dev = log.raw[log.raw.prn.str.contains(rf'.[0-9]{{2,3}}-{freq}')]['prn'].nunique()
            sat_ref = log.ref.df[log.ref.df.prn.str.contains(rf'.[0-9]{{2,3}}-{freq}')]['prn'].nunique()
            _bars_dev.append(sat_dev)
            _bars_ref.append(sat_ref)
        bars_dev.append(_bars_dev)
        bars_ref.append(_bars_ref)
    
    colors_ref = ['tab:blue', 'tab:red']
    colors_dev = ['#98BCE9', '#EB5957']
    for i in range(len(bars_dev)):
        if i > 0:
            bottom_dev = np.sum(np.array(bars_dev[:i]), axis=0)
            bottom_ref = np.sum(np.array(bars_ref[:i]), axis=0)
        else:
            bottom_dev = np.zeros(len(logs))
            bottom_ref = np.zeros(len(logs))

        axs.bar(x-0.1, bars_ref[i], width, bottom=bottom_ref, label=frequencies[i], color=colors_ref[i])
        axs.bar(x+0.1, bars_dev[i], width, bottom=bottom_dev, color=colors_dev[i])
    
    xticks = []
    for log in logs:
        xticks.append(log.device)
    axs.set_xticks(x, xticks)
    axs.set_axisbelow(True)
    axs.legend()
    fig.tight_layout()
