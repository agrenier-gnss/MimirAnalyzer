import hvplot.pandas
import panel as pn
import pandas as pd

from logparser import LogReader, PosReader
from dashboard.grids import getMapGrid, getCoordinateGrid, getDifferenceGrid, getGnssMeasurementGrid
from dashboard.grids import getImuMeasurementGrid, getHealthGrid

# panel serve main_dashboard.py --show --autoreload

# =============================================================================
# Data

filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
#filepath = "./.data/log_old_20230414152332.txt"
#filepath = "./.data/log_mimir_20230715122058.txt"\
filepath = '.data/static/gnss_log_GooglePixel7_2023_02_17_09_55_01.txt'
log = LogReader(filepath)

healthEnabled = False
imuEnabled = False
referenceEnabled = True
if referenceEnabled:
    filepath_ref = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
    ref = PosReader(filepath_ref)

    _df = pd.concat([log.fix, ref.pos], ignore_index=True)
    log.fix = _df 

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

# IMU Measurement tab
if imuEnabled:
    imu_grid, imu_widgets = getImuMeasurementGrid(log)
    sidebarlist.extend(imu_widgets)
else:
    imu_grid = 'IMU data not enabled.'

# Health tab
if healthEnabled:
    health_grid, health_widgets = getHealthGrid(log)
    sidebarlist.extend(health_widgets)
else:
    health_grid = 'Health data not enabled.'

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
    ("Measurements", gnssmeas_grid),
    ("IMU", imu_grid),
    ("Health", health_grid), dynamic=True)

template.main[:6, :12] = tabs
template.servable()




