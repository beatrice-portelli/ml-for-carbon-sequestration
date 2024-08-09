import collections
Config = collections.namedtuple('Config', ('name', 'features'))

# TARGETS

TARGET_CARBON_STORAGE = "CS"
TARGET_CARBON_SEQUESTRATION = "CSE"

# sets of indices

vegetation_indexes = [
    'NDVI_max',  'NDVI_mean',  'NDVI_median',  'NDVI_stdDev',
    'NDII_max',  'NDII_mean',  'NDII_median',  'NDII_stdDev',
    'GNDVI_max', 'GNDVI_mean', 'GNDVI_median', 'GNDVI_stdDev',
    'EVI_max',   'EVI_mean',   'EVI_median',   'EVI_stdDev',
]

chm = [
    'chm_mean', 'chm_median', 'chm_stdev', 'chm_max',
]

climatic = [
    'pr_avg_JJA_median', 'pr_avg_MAM_median',
    'tas_avg_JJA_median', 'tas_max_JJA_median', 'tas_min_JJA_median',
    'tas_avg_MAM_median', 'tas_max_MAM_median', 'tas_min_MAM_median'
]

elevation_slope_aspect = [
    'elevation_mean', 'elevation_median', 'elevation_stdev', 'elevation_max',
    'slope_percentage_mean', 'slope_percentage_median', 'slope_percentage_stdev', 'slope_percentage_max',
    'aspect_degree_mean', 'aspect_degree_median', 'aspect_degree_stdev', 'aspect_degree_max',
]

# INPUT CONFIGS

CONFIG_sat          = Config("Conf1", vegetation_indexes + elevation_slope_aspect)
CONFIG_sat_clim     = Config("Conf2", vegetation_indexes + elevation_slope_aspect + climatic)
CONFIG_sat_chm      = Config("Conf3", vegetation_indexes + elevation_slope_aspect + chm)
CONFIG_sat_clim_chm = Config("Conf4", vegetation_indexes + elevation_slope_aspect + chm + climatic)

vegetation_indexes = ["NDVI", "NDII", "GNDVI", "EVI"]
elevation_slope_aspect = ["ELE", "SLO", "ASP"]
climatic = ["PREC_spring", "PREC_summer", "TEMP_spring", "TEMP_summer"]
chm = ["CHM"]

DL_CONFIG_sat          = Config("Conf1", vegetation_indexes + elevation_slope_aspect)
DL_CONFIG_sat_clim     = Config("Conf2", vegetation_indexes + elevation_slope_aspect + climatic)
DL_CONFIG_sat_chm      = Config("Conf3", vegetation_indexes + elevation_slope_aspect + chm)
DL_CONFIG_sat_clim_chm = Config("Conf4", vegetation_indexes + elevation_slope_aspect + chm + climatic)

# https://gisgeography.com/sentinel-2-bands-combinations/
DL_CONFIG_sat_348      = Config("Conf5_colorinfrared_rev", ["SAT_3_green", "SAT_4_red", "SAT_8_nir"])
DL_CONFIG_sat_234      = Config("Conf5_realcolors_rev", ["SAT_2_blue", "SAT_3_green", "SAT_4_red"])
DL_CONFIG_sat_843      = Config("Conf5_colorinfrared", ["SAT_8_nir", "SAT_4_red", "SAT_3_green"])
DL_CONFIG_sat_432      = Config("Conf5_realcolors", ["SAT_4_red", "SAT_3_green", "SAT_2_blue"])
