
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, BoxZoomTool, LinearAxis, Range1d, Span, Label
from bokeh.plotting import figure

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
    
    p = figure(height=300, width=600, y_range=(0, 0.4), background_fill_color="#efefef")
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
