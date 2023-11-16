import hvplot.pandas
import panel as pn
import pandas as pd

from logparser import LogReader, PosReader
from dashboard.grids import getMapGrid, getCoordinateGrid, getDifferenceGrid, getGnssMeasurementGrid
from dashboard.grids import getImuMeasurementGrid, getHealthGrid

# Command to launch the panel server
# panel serve main_dashboard.py --show --autoreload

# =============================================================================
# Data

#filepath = ".data/health/log_mimir_20230211195331.txt"
filepath = ".data/health/log_mimir_20230520212608.txt"
#filepath = ".data/health/log_mimir_20231108092720.txt"
#filepath = ".data/health/log_mimir_20231108093047.txt"
#filepath = ".data/health/log_mimir_20230715122058.txt"
#filepath = ".data/health/log_mimir_20231108095114.txt"

log = LogReader('test', 'test', 'test', filepath, mode='mimir', specifiedTags= ["ECG", "PPG", "GAL"])

# =============================================================================
# Tabs

sidebarlist = []

health_grid, health_widgets = getHealthGrid(log)
sidebarlist.extend(health_widgets)

# =============================================================================
# Main

# Instantiate the template with widgets displayed in the sidebar
template = pn.template.FastGridTemplate(
    title="Mimir Dashboard",
    sidebar=sidebarlist,
    theme="dark"
)

tabs = pn.Tabs(("Health", health_grid), dynamic=True)

template.main[:6, :12] = tabs
template.servable()




