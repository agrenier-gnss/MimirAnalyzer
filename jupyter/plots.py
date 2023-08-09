
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image

import misc

plt.style.use('plot_style.mplstyle')
        
# ======================================================================================================================
# Position plots

def plotENU(logs):

    # Params
    minor_ticks = 0.5
    major_ticks = 2
    minor_ticks_up = 1
    major_ticks_up = 5
    ylim_east = 5
    ylim_north = ylim_east
    ylim_up = 15

    # Init
    fig, axs = plt.subplots(3, figsize=(8,6))
    plt.suptitle('East / North / Up errors')
    for log in logs:
        df = log['content'].fix.loc[log['content'].fix['provider'].isin(['GPS']), ["datetime", "east", "north", "up"]]
        df['east'].plot(ax=axs[0], label=log['device_name'])
        df['north'].plot(ax=axs[1], label=log['device_name'])
        df['up'].plot(ax=axs[2], label=log['device_name'])
        
    # East
    axs[0].set_ylabel("East [m]")
    axs[0].set_ylim(-ylim_east, ylim_east)
    axs[0].yaxis.set_major_locator(MultipleLocator(major_ticks))
    axs[0].yaxis.set_major_formatter('{x:.0f}')
    axs[0].yaxis.set_minor_locator(MultipleLocator(minor_ticks))

    # East
    axs[1].set_ylabel("North [m]")
    axs[1].set_ylim(-ylim_north, ylim_north)
    axs[1].yaxis.set_major_locator(MultipleLocator(major_ticks))
    axs[1].yaxis.set_major_formatter('{x:.0f}')
    axs[1].yaxis.set_minor_locator(MultipleLocator(minor_ticks))

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
        df = log['content'].fix.loc[log['content'].fix['provider'].isin(['GPS']), ["datetime", "east", "north", "up"]]
        df.plot(x='east', y='north', kind='scatter', label=log['device_name'], color=colors[i], s=6, zorder=3, ax=axs)
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
        fig.suptitle(f"Histogram ENU errors ({log['device_name']})")
        pos = log['content'].fix.loc[log['content'].fix['provider'].isin(['GPS']), ["east", "north", "up"]]

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

def plotStatisticsENU(logs, mode='violin'):

    for log in logs:
        
        minor_ticks = 0.2
        major_ticks = 1
        lim = 5

        fig, axs = plt.subplots(1, figsize=(4,4))

        fig.suptitle(f"Statistics of ENU errors ({log['device_name']})")
        
        pos = log['content'].fix.loc[log['content'].fix['provider'].isin(['GPS']), ["east", "north", "up"]]
        data = [pos['east'], pos['north'], pos['up']]

        if mode == 'violin':
            axs.violinplot(data, showmeans=False, showmedians=True)
        elif mode == 'box':
            axs.boxplot(data, showmeans=True, showfliers=False)
        
        axs.set_xticks([y + 1 for y in range(len(data))], labels=['East', 'North', 'Up'])

        axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
        axs.yaxis.set_major_formatter('{x:.0f}')
        axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))

        plt.ylim(-lim, lim)

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
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
    for loc in locations:
        ax1.plot(loc['longitude'].to_list(), loc['latitude'].to_list(), color='blue', linewidth=2, transform=ccrs.Geodetic())
    
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

    plt.grid(False)


# ======================================================================================================================
# Measurement plots

def plotHistPerSystem(logs, systems, data_name, absolute=False):

    minor_ticks = 0.1
    major_ticks = 1
    lim = 60
    ylim = 0.6
    nb_bins = 31
    bins = np.linspace(0, lim, nb_bins)

    for log in logs:
        fig, axs = plt.subplots(1, figsize=(6,4))
        fig.suptitle(f"Histogram pseudorange errors ({log['device_name']})")

        sats = list(set(log['content'].raw["prn"]))
        sats.sort()

        # Find total absolute sum
        _sats = [item for item in sats if item.startswith(systems)]
        df = log['content'].raw.loc[log['content'].raw['prn'].isin(_sats), [data_name]]
        hist, edges = np.histogram(df[data_name], density=False, bins=bins)
        total_sum = hist.sum()

        bottom = np.zeros(nb_bins-1)
        for sys in systems:
            _sats = [item for item in sats if item.startswith(sys)]
            df = log['content'].raw.loc[log['content'].raw['prn'].isin(_sats), [data_name]]

            if absolute:
                df[data_name] = df[data_name].abs()
                bins = np.linspace(0, lim, nb_bins)
            else:
                bins = np.linspace(-lim, lim, nb_bins)
            hist, edges = np.histogram(df[data_name], density=False, bins=bins)
            unity_density = hist / hist.sum() / len(systems)
            axs.bar(x=edges[:-1], height=unity_density, align='center', 
                    width= 0.9 * (bins[1] - bins[0]), label=sys, zorder=3, bottom=bottom)
            bottom += unity_density
        
        # axs.xaxis.set_major_locator(MultipleLocator(major_ticks))
        # axs.xaxis.set_major_formatter('{x:.0f}')
        # axs.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

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

        sats = list(set(log['content'].raw["prn"]))
        sats.sort()
        
        data = []
        labels = []
        for sys in systems:
            _sats = [item for item in sats if item.startswith(sys)]
            if sys == 'R':
                __sats = _sats
                df = log['content'].raw.loc[log['content'].raw['prn'].isin(__sats), [data_name]]

                # Filter if neeeded
                # q = df[data_name].quantile(0.99)
                # df[data_name] = df[df[data_name].abs() < q]

                _data = df[data_name]
                data.append(_data[~np.isnan(_data)])
                labels.append(f"{misc.getSystemStr(sys)}")
            else:
                for freq in frequencies:
                    __sats = [item for item in _sats if item.endswith(freq[-1])]
                    df = log['content'].raw.loc[log['content'].raw['prn'].isin(__sats), [data_name]]

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
        fig.suptitle(f"{log['device_name']}")
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

# Plot measurement
def plotMeasurement(log, data_name, sat):

    minor_ticks = 0.2
    major_ticks = 1
    lim = 300

    df = log['content'].raw.loc[log['content'].raw['prn'].isin(sat), ['datetime', 'prn', data_name]]

    fig, axs = plt.subplots(1, figsize=(6,5))
    fig.suptitle(f"{data_name} errors ({log['device_name']})")

    df.groupby('prn')[data_name].plot(x='datatime', y=data_name, ax=axs)

    # axs.yaxis.set_major_locator(MultipleLocator(major_ticks))
    # axs.yaxis.set_major_formatter('{x:.0f}')
    # axs.yaxis.set_minor_locator(MultipleLocator(minor_ticks))

    #plt.ylim(-lim, lim)

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()