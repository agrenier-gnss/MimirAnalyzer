import hvplot.pandas
import numpy as np
import panel as pn
import pandas as pd
import folium
import holoviews as hv
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import pymap3d as pm
import geopandas as gpd
from shapely.geometry import LineString

from logparser import LogReader, PosReader
from MapGrid import getMapGrid
from CoordinateGrid import getCoordinateGrid
from DifferenceGrid import getDifferenceGrid
from GnssMeasurementGrid import getGnssMeasurementGrid

# panel serve main_dashboard.py --show --autoreload

# =============================================================================
# Data

filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
log = LogReader(filepath)

filepath_ref = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
ref = PosReader(filepath_ref)

_df = pd.concat([log.fix, ref.pos], ignore_index=True)
log.fix = _df 
print(log.fix)

# =============================================================================
# Tabs

sidebarlist = []
# Map tab
map_grid = getMapGrid(log)

# Coordinates tab
coordinate_grid = getCoordinateGrid(log)

# Difference tab
difference_grid = getDifferenceGrid(log)

# GNSS Measurement tab
gnssmeas_grid, gnssmeas_widgets = getGnssMeasurementGrid(log)
sidebarlist.extend(gnssmeas_widgets)

# =============================================================================
# Main

# Instantiate the template with widgets displayed in the sidebar
template = pn.template.FastGridTemplate(
    title="Mimir Dashboard",
    sidebar=sidebarlist,
    theme="dark"
)

tabs = pn.Tabs(
    ("Map", map_grid), 
    ("Coordinates", coordinate_grid),
    ("Differences", difference_grid),
    ("Measurements", gnssmeas_grid), dynamic=True)

template.main[:6, :12] = tabs
template.servable()




