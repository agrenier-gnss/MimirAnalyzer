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

#filepath     = "./example_data/log_mimir_GooglePixel7_20230801110405_trimmed.txt"

#filepath = "./.data/log_old_20230414152332.txt"
#filepath = "./.data/log_mimir_20230715122058.txt"\
filepath = '.data/static/gnss_log_GooglePixel7_2023_02_17_09_55_01.txt'
#filepath = "./.data/dynamic_campus/log_mimir_GooglePixel7_20230801110405.txt"

# filepath = "./.data/2023_Dataset_Hervanta/3_dynamic_campus/Xiaomi_11T/log_Xiaomi11T_20230801111451.txt"
# filepath = "./.data/2023_Dataset_Hervanta/3_dynamic_campus/Google_Pixel7/log_GooglePixel7_20230801110405.txt"
filepath = "./.data/2023_Dataset_Hervanta/3_dynamic_campus/Google_PixelWatch/log_GooglePixelWatch_20230801110404.txt"

#filepath = "./.data/dynamic_campus/log_mimir_SamsungA52_Backpack_20230811102920.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_GooglePixelWatch_20230811141922.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_GooglePixelWatch_20230811150346.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_GooglePixel7_20230811141817.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_GooglePixel7_20230811150244.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_OnePlusNord2_20230811141819.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_OnePlusNord2_20230811150159.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_SamsungA52_20230811141745.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_SamsungA52_20230811150240.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_Xiaomi11_20230811141828.txt"
# filepath = "./.data/dynamic_hervanta/log_mimir_Xiaomi11_20230811150208.txt"

# filepath="./.data/2023_Dataset_Hervanta/S1_static/raw/log_GooglePixel7_20230217095501.txt"

# filepath="./.data/2023_Dataset_Hervanta/S2_dynamic_campus/raw/log_GooglePixel7_20230801110405_modified.txt"
filepath="/mnt/c/Users/Antoine/OneDrive - TUNI.fi/Data/Android/_tests/log_mimir_20240205122020.txt"

#filepath="/mnt/c/Users/Antoine/OneDrive - TUNI.fi/Data/log_mimir_20230814145154.txt"

filepath="/mnt/c/Users/vmangr/OneDrive - TUNI.fi/Shared/GEOLOC/dataset/20230313/open_sky_iffstar/GP7/log_mimir_20240313111400.txt"

filepath="/mnt/c/Users/vmangr/OneDrive - TUNI.fi/Shared/GEOLOC/dataset/.archive/20240314/A52/log_mimir_20240314114638.txt"

filepath="/mnt/c/Users/vmangr/Documents/Datasets/AndroidDevices/dataset_GEOLOC_20240319/dataset/.archive/20240313/SW6/log_mimir_20231114215734.txt"

filepath="/mnt/c/Users/vmangr/OneDrive - TUNI.fi/Shared/GEOLOC/dataset/S1/A08/GP7/log_mimir_20240313111400.txt"

log = LogReader('test', 'test', 'test', filepath, mode='old')

referenceEnabled = False
filepath_ref = "./example_data/NMND18410025C_2023-04-14_13-03-45.pos"
if referenceEnabled:
    ref = PosReader(filepath_ref)

    _df = pd.concat([log.fix, ref.pos], ignore_index=True)
    log.fix = _df 

healthEnabled = False
imuEnabled = False

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




