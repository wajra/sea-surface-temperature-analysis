from datetime import timedelta
import datetime
import re
import glob
import numpy as np
import os.path
from netCDF4 import Dataset, num2date
from pandas import DataFrame
from geometry_operations import get_bound_limits
from geometry_operations import bounding_box_contents
from geometry_operations import get_overlapping_coords
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from daily_computations import assign_monthly_climatologies
from daily_computations import assign_maximum_sst_values
from daily_computations import compute_daily_climatology
from daily_computations import build_daily_sst_index
from daily_computations import get_lat_lons


def write_dhw_index(coral_reef_polygon, sst_source='JPL_GHRSST', mmm_sst_source='JPL_GHRSST',
                    write_location='D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/'):
    specific_lats, specific_lons = get_lat_lons(data_source=sst_source)
    coral_reef_overlapping_coords = get_overlapping_coords(specific_lats, specific_lons, coral_reef_polygon)



if __name__ == '__main__':
    noaa_df = build_daily_sst_index(data_source='NOAA_OI_SST')
    ghrsst_df = build_daily_sst_index(data_source='JPL_GHRSST')
    # First, we need to select a coral reef site
    coral_reef_name = 'Great Basses'
    noaa_lats, noaa_lons = get_lat_lons(data_source='NOAA_OI_SST')
    ghrsst_lats, ghrsst_lons = get_lat_lons(data_source='JPL_GHRSST')
    # Then we will simply make an index

    current_date = '1998-05-20'
    date_object = datetime.datetime.strptime(current_date, '%Y-%m-%d')

    start_date = date_object - timedelta(days=84)
    print('We go from {0} to {1}'.format(start_date, date_object))

    td_object = date_object - start_date
    print(td_object)

    start_date_string = start_date.strftime('%Y-%m-%d')
    print(start_date_string)

    dhw_df = noaa_df[start_date_string: current_date]

    coral_reef_bounds = get_bound_limits()
    sample_coral_reef_bounds = coral_reef_bounds[coral_reef_name]
    sample_coral_reef = bounding_box_contents(sample_coral_reef_bounds)
    sample_coral_reef_coords = get_overlapping_coords(ghrsst_lats, ghrsst_lons, sample_coral_reef)

    sample_coral_reef_mean_mmm_sst = assign_maximum_sst_values(data_source='JPL_GHRSST')[sample_coral_reef_coords[:, 0],
                                                                                         sample_coral_reef_coords[:, 1]
    ].mean()

    sample_coral_reef_coords_noaa = get_overlapping_coords(noaa_lats, noaa_lons, sample_coral_reef)

    sample_coral_reef_mean_mmm_sst_nooa = assign_maximum_sst_values(data_source='NOAA_OI_SST')[sample_coral_reef_coords_noaa[:, 0],
    sample_coral_reef_coords_noaa[:, 1]].mean()

    print('Mean MMM SST from JPL: {0}'.format(sample_coral_reef_mean_mmm_sst))
    print('Mean MMM SST from NOAA: {0}'.format(sample_coral_reef_mean_mmm_sst_nooa))
    hotspot_dict = {'date': [],
                    'hotspot_level': []}
    for date_index, row in noaa_df.sort_index().iterrows():
        mean_temp = row['sst'][sample_coral_reef_coords_noaa[:, 0], sample_coral_reef_coords_noaa[:, 1]].mean()
        if mean_temp - sample_coral_reef_mean_mmm_sst >= 1.0:
            hotspot_dict['date'].append(date_index)
            hotspot_dict['hotspot_level'].append(mean_temp - sample_coral_reef_mean_mmm_sst)

    hotspot_df = DataFrame.from_dict(hotspot_dict)
    hotspot_df.index = hotspot_df['date']
    del(hotspot_df['date'])
    print(hotspot_df)

    print(hotspot_df[start_date_string:current_date].hotspot_level.sum())

    dhw_value = hotspot_df[start_date_string:current_date].hotspot_level.sum()/7

    print(dhw_value)

    hotspot_df.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/{0}_hotspot_history_noaa.xlsx'.format('_'.join(coral_reef_name.lower().split(' '))))

    hotspot_dict = {'date': [],
                    'hotspot_level': []}
    for date_index, row in ghrsst_df.sort_index().iterrows():
        dataset = Dataset(row['file_name'])
        sst_vals = dataset.variables['analysed_sst'][0]
        mean_temp = sst_vals[sample_coral_reef_coords[:, 0], sample_coral_reef_coords[:, 1]].mean()
        if mean_temp - sample_coral_reef_mean_mmm_sst >= 1.0:
            hotspot_dict['date'].append(date_index)
            hotspot_dict['hotspot_level'].append(mean_temp - sample_coral_reef_mean_mmm_sst)

    hotspot_df = DataFrame.from_dict(hotspot_dict)
    hotspot_df.index = hotspot_df['date']
    del(hotspot_df['date'])
    hotspot_df.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/{0}_hotspot_history_ghrsst.xlsx'.format('_'.join(coral_reef_name.lower().split(' '))))

    vankalai_2016_end_date_string = '2016-05-14'
    vankalai_2016_end_date = datetime.datetime.strptime(vankalai_2016_end_date_string, '%Y-%m-%d')

    vankalai_2016_start_date = vankalai_2016_end_date - timedelta(days=84)
    vankalai_2016_start_date_string = vankalai_2016_start_date.strftime('%Y-%m-%d')

    print(vankalai_2016_start_date_string)
    print(vankalai_2016_end_date_string)

    ghrsst_dhw_value = hotspot_df[vankalai_2016_start_date_string: vankalai_2016_end_date_string].hotspot_level.sum()/7
    print(ghrsst_dhw_value)

