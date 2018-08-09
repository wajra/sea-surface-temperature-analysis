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


if __name__ == '__main__':
    noaa_df = build_daily_sst_index(data_source='NOAA_OI_SST')
    counter = 0

    noaa_lats, noaa_lons = get_lat_lons(data_source='NOAA_OI_SST')

    coral_reef_bounds = get_bound_limits()

    noaa_mmm_sst_values = assign_maximum_sst_values(data_source='NOAA_OI_SST')

    coral_reef_max_dhw = {'coral_reef_name': [],
                          'year': [],
                          'max_dhw': [],
                          'max_hotspot': []}

    # Now we will calculate DHW for 1998 and 2016
    bleaching_years = {'1982': ['1982-01-01', '1982-12-31'], '1998': ['1998-01-01', '1998-12-31'],
                       '2010': ['2010-01-01', '2010-12-31'], '2016': ['2016-01-01', '2016-12-31']}

    """
    Adams Bridge Reef
    Silavathurai Reef
    Vellankulam Reef
    Kandakuliya Reef
    Talawila Reef
    Delft Reef

    """

    for coral_reef_name in ['Iranativu Reef', 'Hikkaduwa Reef', 'Kalpitiya Bar Reef', 'Analaitivu Reef',
                            'Palaitivu Reef', 'Vankalai Reef', 'Unawatuna Reef', 'Nainativu Reef', 'Punkudutivu Reef',
                            'Beruwala Reef', 'Weligama Reef', 'Great Basses', 'Adams Bridge Reef', 'Silavathurai Reef',
                            'Vellankulam Reef', 'Kandakuliya Reef', 'Talawila Reef', 'Delft Reef']:
        hotspot_dict = {'date': [],
                        'hotspot_level': [],
                        'sst': []}

        consecutive_dict = {'date': [], 'status': []}

        coral_reef_bound = coral_reef_bounds[coral_reef_name]
        coral_reef_coords = get_overlapping_coords(noaa_lats, noaa_lons, coral_reef_bound)
        overall_mmm_sst = noaa_mmm_sst_values[coral_reef_coords[:, 0], coral_reef_coords[:, 1]].mean()
        print('Currrent Coral Reef: {0}'.format(coral_reef_name))
        # We will first print every Hotspot value throughout history and write them to a file
        for daily_index, row in noaa_df.sort_index().iterrows():
            current_sst = row['sst'][coral_reef_coords[:, 0], coral_reef_coords[:, 1]].mean()
            if current_sst - overall_mmm_sst >= 1.0:
                hotspot_dict['date'].append(daily_index)
                hotspot_dict['hotspot_level'].append(current_sst - overall_mmm_sst)
                hotspot_dict['sst'].append(current_sst - 273.15)

                consecutive_dict['date'].append(daily_index)
                consecutive_dict['status'].append(current_sst - overall_mmm_sst)
            else:
                consecutive_dict['date'].append(daily_index)
                consecutive_dict['status'].append(current_sst - overall_mmm_sst)
        # Put the data into a dataframe
        hotspot_df = DataFrame.from_dict(hotspot_dict)
        hotspot_df.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/overall_hotspot_history_{0}.xlsx'.format(coral_reef_name))

        consecutive_df = DataFrame.from_dict(consecutive_dict)
        consecutive_df.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/CONSECUTIVE_HOTSPOT_VALUES/hotspot_status_{0}.xlsx'.format(coral_reef_name))


        # Then we'll set the index of the hotspot_df to datetime row
        hotspot_df.index = hotspot_df['date']
        del hotspot_df['date']

        for bleaching_year in sorted(bleaching_years.keys()):
            dhw_value_dict = {'date': [], 'dhw': []}
            period_start = bleaching_years[bleaching_year][0]
            period_end = bleaching_years[bleaching_year][1]

            period_start_date = datetime.datetime.strptime(period_start, '%Y-%m-%d')
            period_end_date = datetime.datetime.strptime(period_end, '%Y-%m-%d')

            current_date = period_start_date

            while (period_end_date - current_date).days >= 0:
                dhw_start_date = current_date - timedelta(days=84)

                dhw_start_date_string = dhw_start_date.strftime('%Y-%m-%d')
                current_date_string = current_date.strftime('%Y-%m-%d')

                # Then we get the total hotspot value over 84 days
                dhw_value = hotspot_df[dhw_start_date_string: current_date_string].hotspot_level.sum()/7
                dhw_value_dict['date'].append(current_date)
                dhw_value_dict['dhw'].append(dhw_value)

                current_date += timedelta(days=1)

            dhw_df = DataFrame.from_dict(dhw_value_dict)
            # Getting the maximum DHW value
            max_dhw = dhw_df['dhw'].max()
            max_hotspot = hotspot_df[period_start: period_end].hotspot_level.max()
            print(max_dhw)
            print(max_hotspot)
            coral_reef_max_dhw['coral_reef_name'].append(coral_reef_name)
            coral_reef_max_dhw['year'].append(bleaching_year)
            coral_reef_max_dhw['max_dhw'].append(round(max_dhw, 3))
            coral_reef_max_dhw['max_hotspot'].append(round(max_hotspot, 3))
            dhw_df.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/dhw_history_{0}_{1}.xlsx'.format(coral_reef_name,
                                                                                                                       bleaching_year))


    for daily_index, row in noaa_df.sort_index().iterrows():
        current_sst = row['sst'][15, 11]
        print(current_sst)
        break

    non_overlapping_reefs = {'Trincomalee Reefs': (14, 13), 'Pigeon Islands Reef': (15, 12),
                             'Thennady Bay Reef': (11, 14), 'Batticaloa Reef': (10, 15), 'Little Basses': (5, 14)}

    for coral_reef_name in sorted(non_overlapping_reefs.keys()):
        print(coral_reef_name)
        hotspot_dict = {'date': [],
                        'hotspot_level': [],
                        'sst': []}
        consecutive_dict = {'date': [], 'status': []}
        coral_reef_coords = non_overlapping_reefs[coral_reef_name]
        overall_mmm_sst = noaa_mmm_sst_values[coral_reef_coords[0], coral_reef_coords[1]]
        print(overall_mmm_sst)
        for daily_index, row in noaa_df.sort_index().iterrows():
            current_sst = row['sst'][coral_reef_coords[0], coral_reef_coords[1]]
            if current_sst - overall_mmm_sst >= 1.0:
                hotspot_dict['date'].append(daily_index)
                hotspot_dict['hotspot_level'].append(current_sst - overall_mmm_sst)
                hotspot_dict['sst'].append(current_sst - 273.15)
                consecutive_dict['date'].append(daily_index)
                consecutive_dict['status'].append(current_sst - overall_mmm_sst)
            else:
                consecutive_dict['date'].append(daily_index)
                consecutive_dict['status'].append(current_sst - overall_mmm_sst)

        hotspot_df = DataFrame.from_dict(hotspot_dict)
        hotspot_df.to_excel(
            'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/overall_hotspot_history_{0}.xlsx'.format(
                coral_reef_name))
        consecutive_df = DataFrame.from_dict(consecutive_dict)
        consecutive_df.to_excel(
            'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/CONSECUTIVE_HOTSPOT_VALUES/hotspot_status_{0}.xlsx'.format(
                coral_reef_name))
        # Then we'll set the index of the hotspot_df to datetime row
        hotspot_df.index = hotspot_df['date']
        del hotspot_df['date']

        for bleaching_year in sorted(bleaching_years.keys()):
            dhw_value_dict = {'date': [], 'dhw': []}
            period_start = bleaching_years[bleaching_year][0]
            period_end = bleaching_years[bleaching_year][1]

            period_start_date = datetime.datetime.strptime(period_start, '%Y-%m-%d')
            period_end_date = datetime.datetime.strptime(period_end, '%Y-%m-%d')

            current_date = period_start_date

            while (period_end_date - current_date).days >= 0:
                dhw_start_date = current_date - timedelta(days=84)

                dhw_start_date_string = dhw_start_date.strftime('%Y-%m-%d')
                current_date_string = current_date.strftime('%Y-%m-%d')

                # Then we get the total hotspot value over 84 days
                dhw_value = hotspot_df[dhw_start_date_string: current_date_string].hotspot_level.sum() / 7
                dhw_value_dict['date'].append(current_date)
                dhw_value_dict['dhw'].append(dhw_value)

                current_date += timedelta(days=1)

            dhw_df = DataFrame.from_dict(dhw_value_dict)
            # Getting the maximum DHW value
            max_dhw = dhw_df['dhw'].max()
            max_hotspot = hotspot_df[period_start: period_end].hotspot_level.max()
            print(max_dhw)
            coral_reef_max_dhw['coral_reef_name'].append(coral_reef_name)
            coral_reef_max_dhw['year'].append(bleaching_year)
            coral_reef_max_dhw['max_dhw'].append(round(max_dhw, 3))
            coral_reef_max_dhw['max_hotspot'].append(round(max_hotspot, 3))
            dhw_df.to_excel(
                'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/dhw_history_{0}_{1}.xlsx'.format(
                    coral_reef_name, bleaching_year))

    print(coral_reef_max_dhw)
    coral_reef_max_dhw_df = DataFrame.from_dict(coral_reef_max_dhw)
    coral_reef_max_dhw_df.to_excel(
        'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/all_coral_reef_histories_rounded.xlsx')