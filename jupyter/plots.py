
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import pandas as pd

import seaborn as sns

import misc

plt.style.use('plot_style.mplstyle')
        
# ======================================================================================================================
# Position plots

def plotENU(logs, lim, ticks):

    # Params
    minor_ticks_east = ticks[0]
    major_ticks_east = ticks[1]
    minor_ticks_north = ticks[2]
    major_ticks_north = ticks[3]
    minor_ticks_up = ticks[4]
    major_ticks_up = ticks[5]
    ylim_east = lim[0]
    ylim_north = lim[1]
    ylim_up = lim[2]

    # Init
    fig, axs = plt.subplots(3, figsize=(8,6))
    plt.suptitle('East / North / Up errors')
    for log in logs:
        df = log.fix.loc[log.fix['provider'].isin(['GPS']), ["east", "north", "up"]]
        df.index = [idx - df.index[0] for idx in df.index]
        axs[0].plot(df['east'], label=f"{log.manufacturer} {log.device}")
        axs[1].plot(df['north'], label=f"{log.manufacturer} {log.device}")
        axs[2].plot(df['up'], label=f"{log.manufacturer} {log.device}")
        
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
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left",
            mode="expand", ncol=len(labels))

    fig.tight_layout(rect=[0, 0.03, 1, 1])

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
        
        pos = log.fix.loc[log.fix['provider'].isin(['GPS']), ["east", "north", "up"]]
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

def plotMap(locations, extent, scale):
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

    fig = plt.figure(figsize=(8,8)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_extent(extent) # set extents

    ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification

    # Polylines
    for label, loc in locations.items():
        ax1.plot(loc['longitude'].to_list(), loc['latitude'].to_list(),
                 linewidth=2, transform=ccrs.Geodetic(), label=label)
    
    # Grid
    # gl = ax1.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # gl.top_labels = False
    # gl.right_labels = False

    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.3f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.3f',degree_symbol='') # format lats
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
def plotStatisticsDataBox(logs, data_name, ylabel, systems, frequencies, lim, ticks):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]

    for log in logs:

        sats = list(set(log.raw["prn"]))
        sats.sort()
        
        data = []
        labels = []
        for sys in systems:
            _sats = [item for item in sats if item.startswith(sys)]
            if sys == 'R':
                __sats = _sats
                df = log.raw.loc[log.raw['prn'].isin(__sats), [data_name]]

                # Filter if neeeded
                # q = df[data_name].quantile(0.99)
                # df[data_name] = df[df[data_name].abs() < q]

                _data = df[data_name]
                data.append(_data[~np.isnan(_data)])
                labels.append(f"{misc.getSystemStr(sys)}")
            else:
                for freq in frequencies:
                    __sats = [item for item in _sats if item.endswith(freq[-1])]
                    df = log.raw.loc[log.raw['prn'].isin(__sats), [data_name]]

                    # # Filter if neeeded
                    # q = df[data_name].quantile(0.99)
                    # df[data_name] = df[df[data_name].abs() < q]

                    _data = df[data_name]
                    _data = _data[~np.isnan(_data)].tolist()
                    
                    if len(_data) != 0:
                        data.append(_data)
                    else:
                        data.append([float('nan'), float('nan')])
                    labels.append(f"{misc.getSystemStr(sys)}-{freq}")
        
        fig, axs = plt.subplots(1, figsize=(8,5))
        fig.suptitle(f"{log.manufacturer} {log.device}")
        axs.boxplot(data, showmeans=True, showfliers=False)
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
def plotStatisticsDataViolin(logs, data_name, ylabel, systems, frequencies, lim, ticks):

    minor_ticks = ticks[0]
    major_ticks = ticks[1]
    percentile = 0.999

    for log in logs:

        sats = list(set(log.raw["prn"]))
        sats.sort()
        
        labels = []
        for sys in systems:
            labels.append(f"{misc.getSystemStr(sys)}")

        _sats = [item for item in sats if item.startswith(systems)]
        _sats = [item for item in _sats if item.endswith(tuple([freq[-1] for freq in frequencies]))]
        
        df = log.raw.loc[log.raw['prn'].isin(_sats), ['prn', 'system', 'frequency', data_name]]
        
        # Filter if neeeded
        q = df[data_name].quantile(percentile)
        df = df[df[data_name].abs() < q]

        df.reset_index(drop=True, inplace=True)

        # Correction for mono-frequencies
        for sys in systems:
            _frequencies = list(set(df.loc[df['system'].isin([sys])]['frequency']))
            new_row = {'system':f'{sys}', 'frequency':'L1', data_name:float('nan')}
            
            if 'L1' not in _frequencies:
                new_row = [f'{sys}00-L1', f'{sys}', 'L1', float('nan')]
                df.loc[len(df.index)] = new_row
                df.loc[len(df.index)] = new_row
            elif 'L5' not in _frequencies:
                new_row = [f'{sys}00-L5', f'{sys}', 'L5', float('nan')]
                df.loc[len(df.index)] = new_row
                df.loc[len(df.index)] = new_row
        
        fig, axs = plt.subplots(1, figsize=(6,5))
        fig.suptitle(f"{log.manufacturer} {log.device}")

        sns.violinplot(ax=axs, data=df, x='system', y=data_name, hue='frequency', 
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