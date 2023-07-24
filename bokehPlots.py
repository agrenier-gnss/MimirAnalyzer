
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, BoxZoomTool, LinearAxis, Range1d, Span, Label
from bokeh.plotting import figure

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import numpy as np

# =====================================================================================================================

def plotLineTimeRange(ts, data_name):

    source = ColumnDataSource(ts)

    p = figure(height=300, width=1200,
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(ts.index[100], ts.index[1000]))

    p.line('datetime', data_name, source=source)
    p.yaxis.axis_label = data_name

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1200, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None, toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line('datetime', data_name, source=source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.add_tools(BoxZoomTool(dimensions="width"))
    select.toolbar.active_multi = 'auto'

    return column(p, select)

# =====================================================================================================================

def plotHist(ts, data_name):
    
    source = ColumnDataSource(ts)
    bins = np.linspace(0, np.max(ts[data_name]), 50)
    hist, edges = np.histogram(ts[data_name], density=True, bins=bins)
    unity_density = hist / hist.sum()

    values = ts[data_name].values.tolist()
    values = [x for x in values if str(x) != 'nan']

    p = figure(height=300, width=600, y_range=(0, 0.4), x_range=(0, 100), background_fill_color="#efefef")
    p.quad(top=unity_density, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="skyblue", line_color="white")
    p.yaxis.axis_label = data_name

    p95 = np.percentile(values, 95)
    vline = Span(location=p95, dimension='height', line_color='red', line_width=1)
    p.add_layout(vline)
    my_label = Label(x=p95, y=(p.height-100), y_units='screen', text=f'95% - {p95:.3f}m', 
                     text_color='red',text_font_style='italic', text_font_size='8pt')
    p.add_layout(my_label)

    p99 = np.percentile(values, 99)
    vline = Span(location=p99, dimension='height', line_color='red', line_width=1)
    p.add_layout(vline)
    my_label = Label(x=p99, y=(p.height-50), y_units='screen', text=f'99% - {p99:.3f}m', 
                     text_color='red',text_font_style='italic',text_font_size='8pt')
    p.add_layout(my_label)


    p.extra_y_ranges['cumsum'] = Range1d(0, 1.1)
    p.line(edges[:-1], unity_density.cumsum(), color="navy", y_range_name="cumsum")

    ax2 = LinearAxis(y_range_name="cumsum", axis_label="Cumulative probability")
    p.add_layout(ax2, 'right')
    

    return p

# =====================================================================================================================

def plotEN_pyplot(ts, lim):

    #plt.style.use('seaborn-darkgrid')

    minor_ticks = 0.1
    major_ticks = 1

    plt.figure(figsize=(8,8))
    ts.plot(x='east', y='north', kind='scatter', s=6, zorder=3)
    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
    # ax.xaxis.set_major_formatter('{x:.0f}')
    # ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
    # ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
    # ax.yaxis.set_major_formatter('{x:.0f}')
    # ax.yaxis.set_minor_locator(MultipleLocator(minor_ticks))
    plt.axis('square')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title("North / East")
    plt.grid(zorder=0)
    plt.show()
    
    return 

# =====================================================================================================================

def plot_hist_pyplot(ts, data_name, lim, title):

    minor_ticks = 0.05
    major_ticks = 0.2

    ts[data_name] = ts[data_name].abs()
    
    bins = np.linspace(0, lim, 50)
    hist, edges = np.histogram(ts[data_name], density=True, bins=bins)
    unity_density = hist / hist.sum()

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.bar(x=edges[:-1], height=unity_density, align='edge', width= 0.9 * (bins[1] - bins[0]), zorder=3)
    ax2.stairs(values=unity_density.cumsum(), edges=edges[:], color='red', zorder=4)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
    ax1.grid(zorder=0)
    plt.title(title)
    plt.show()
    
    return 

# =====================================================================================================================
