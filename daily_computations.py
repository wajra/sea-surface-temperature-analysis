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

"""
So....
Now we have everything we need to perform daily computations. We will start off with the following
1. Deriving Daily Mean Climatologies
2. Computing Daily Anomalies
3. Computing Bleaching Hotspots
"""

mm_sst_read_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/JPL_GHRSST/MM_SST',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/NOAA_OI_SST/MM_SST'}

daily_sst_read_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/JPL_TEMP_DATASET_1',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/NOAA_OI_SST_DATA'}

mmm_sst_read_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/JPL_GHRSST/MMM_SST/JPL_GHRSST_MMM_SST',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/NOAA_OI_SST/MMM_SST/NOAA_OI_SST_MMM_SST'}

sample_nc_files = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/SAMPLE_DATASETS/JPL_GHRSST_SAMPLE.nc',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/SAMPLE_DATASETS/NOAA_OI_SST_SAMPLE.nc'}


def assign_monthly_climatologies(data_source = 'JPL_GHRSST'):
    monthly_climatologies = []
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            read_location = mm_sst_read_locations[data_source]
            climatology_files = glob.glob(read_location + '/MM_SST_JPL_GHRSST*')
            for file in climatology_files:
                monthly_climatologies.append(np.load(file))
            return monthly_climatologies
        elif data_source == 'NOAA_OI_SST':
            read_location = mm_sst_read_locations[data_source]
            climatology_files = glob.glob(read_location + '/MM_SST_NOAA_OI*')
            for file in climatology_files:
                monthly_climatologies.append(np.load(file))
            return monthly_climatologies
    else:
        return None

def assign_maximum_sst_values(data_source = 'JPL_GHRSST'):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            mmm_sst_array = np.load(mmm_sst_read_locations[data_source])
            return mmm_sst_array
        elif data_source == 'NOAA_OI_SST':
            mmm_sst_array = np.load(mmm_sst_read_locations[data_source])
            return mmm_sst_array
    else:
        return None


def compute_daily_climatology(data_source='JPL_GHRSST', date=datetime.date(2004, 6, 18)):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            monthly_climatologies = assign_monthly_climatologies('JPL_GHRSST')
            current_date_month = date.month
            current_date_year = date.year
            month_A_index = None
            month_B_index = None
            """
            for month_index, climatology_month in enumerate(monthly_climatologies, start=1):
                if date >= datetime.date(date.year, month_index, 15) and date <= datetime.date(date.year, month_index+1, 15):
                    month_A_index = month_index
                    break
            """
            date_A = None
            date_B = None
            if datetime.date(current_date_year, 12, 15) <= date < datetime.date(current_date_year+1, 1, 1):
                date_A = datetime.date(current_date_year, 12, 15)
                date_B = datetime.date(current_date_year+1, 1, 15)
                month_A_index = 12
                month_B_index = 1
            elif datetime.date(current_date_year, 1, 1) <= date < datetime.date(current_date_year, 1, 15):
                date_A = datetime.date(current_date_year-1, 12, 15)
                date_B = datetime.date(current_date_year, 1, 15)
                month_A_index = 12
                month_B_index = 1
            else:
                for month_index, climatology_month in enumerate(monthly_climatologies, start=1):
                    if datetime.date(current_date_year, month_index, 15) <= date < datetime.date(current_date_year, month_index+1, 15):
                        date_A = datetime.date(current_date_year, month_index, 15)
                        date_B = datetime.date(current_date_year, month_index+1, 15)
                        month_A_index = month_index
                        month_B_index = month_index + 1
                        break
            date_A_B_gap = date_B - date_A
            print('Gap between B and A: {0}'.format(date_A_B_gap.days))
            date_A_current_date_gap = date - date_A
            print(date_A_current_date_gap)
            print('Gap between current date and A: {0}'.format(date_A_current_date_gap.days))
            day_fraction = date_A_current_date_gap.days/date_A_B_gap.days
            print(day_fraction)
            date_A_sst = monthly_climatologies[month_A_index - 1]
            date_B_sst = monthly_climatologies[month_B_index - 1]
            print(month_A_index)
            print(month_B_index)
            daily_sst_climatology = day_fraction * (date_B_sst - date_A_sst) + date_A_sst
            return daily_sst_climatology
        elif data_source == 'NOAA_OI_SST':
            monthly_climatologies = assign_monthly_climatologies('NOAA_OI_SST')
            current_date_month = date.month
            current_date_year = date.year
            month_A_index = None
            month_B_index = None
            """
            for month_index, climatology_month in enumerate(monthly_climatologies, start=1):
                if date >= datetime.date(date.year, month_index, 15) and date <= datetime.date(date.year, month_index+1, 15):
                    month_A_index = month_index
                    break
            """
            date_A = None
            date_B = None
            if datetime.date(current_date_year, 12, 15) <= date < datetime.date(current_date_year + 1, 1, 1):
                date_A = datetime.date(current_date_year, 12, 15)
                date_B = datetime.date(current_date_year + 1, 1, 15)
                month_A_index = 12
                month_B_index = 1
            elif datetime.date(current_date_year, 1, 1) <= date < datetime.date(current_date_year, 1, 15):
                date_A = datetime.date(current_date_year - 1, 12, 15)
                date_B = datetime.date(current_date_year, 1, 15)
                month_A_index = 12
                month_B_index = 1
            else:
                for month_index, climatology_month in enumerate(monthly_climatologies, start=1):
                    if datetime.date(current_date_year, month_index, 15) <= date < datetime.date(current_date_year,
                                                                                                 month_index + 1, 15):
                        date_A = datetime.date(current_date_year, month_index, 15)
                        date_B = datetime.date(current_date_year, month_index + 1, 15)
                        month_A_index = month_index
                        month_B_index = month_index + 1
                        break
            date_A_B_gap = date_B - date_A
            print('Gap between B and A: {0}'.format(date_A_B_gap.days))
            date_A_current_date_gap = date - date_A
            print(date_A_current_date_gap)
            print('Gap between current date and A: {0}'.format(date_A_current_date_gap.days))
            day_fraction = date_A_current_date_gap.days / date_A_B_gap.days
            print(day_fraction)
            date_A_sst = monthly_climatologies[month_A_index - 1]
            date_B_sst = monthly_climatologies[month_B_index - 1]
            print(month_A_index)
            print(month_B_index)
            daily_sst_climatology = day_fraction * (date_B_sst - date_A_sst) + date_A_sst
            return daily_sst_climatology
        else:
            return None


def compute_daily_anomaly(data_source='JPL_GHRSST', date=datetime.date(2004, 6, 18)):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            daily_sst_climatology = compute_daily_climatology(date)


def get_specific_daily_sst(data_source='JPL_GHRSST', date=datetime.date(2004, 6, 18)):
    return None


# Using this method we should build an index to retrieve the files that we need instantly
# For now, let's just do this for JPL_GHRSST values
def build_daily_sst_index(data_source='JPL_GHRSST'):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            read_location = daily_sst_read_locations[data_source]
            if os.path.exists(read_location):
                print('We can proceed')
                all_data = {'date': [], 'file_name': []}
                all_nc_files = glob.glob(read_location + '/*-fv04-MUR_subset.nc')
                for file in all_nc_files:
                    dataset = Dataset(file)
                    current_datetime = dataset.variables['time']
                    current_date = num2date(current_datetime[:], current_datetime.units, calendar='standard')[0]
                    all_data['date'].append(current_date)
                    all_data['file_name'].append(file)
                # Then we load this dictionary into a Pandas dataframe
                daily_index_df = DataFrame.from_dict(all_data)
                # We set the 'date' column as index
                daily_index_df.index = daily_index_df['date']
                # Deleting the original 'date' column
                del(daily_index_df['date'])
                return daily_index_df
            else:
                return False
        elif data_source == 'NOAA_OI_SST':
            read_location = daily_sst_read_locations[data_source]
            if os.path.exists(read_location):
                print('We can proceed')
                all_data = {'date': [], 'sst': []}
                all_nc_files = glob.glob(read_location + '/subset.*.nc')
                for file in all_nc_files:
                    print(file)
                    yearly_dataset = Dataset(file)
                    time = yearly_dataset.variables['time']
                    standard_time = num2date(time[:], time.units, calendar='standard')
                    sst_values = yearly_dataset.variables['sst']
                    for current_date, current_sst in zip(standard_time, sst_values):
                        current_sst += 273.15
                        all_data['date'].append(current_date)
                        all_data['sst'].append(current_sst)
                daily_index_df = DataFrame.from_dict(all_data)
                # We set the 'date' column as index
                daily_index_df.index = daily_index_df['date']
                # Deleting the original 'date' column
                del (daily_index_df['date'])
                return daily_index_df
        else:
            return None


def get_lat_lons(data_source='JPL_GHRSST'):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            sample_file = Dataset(sample_nc_files[data_source])
            lons = sample_file.variables['lon'][:]
            lats = sample_file.variables['lat'][:]
            return lats, lons
        elif data_source == 'NOAA_OI_SST':
            sample_file = Dataset(sample_nc_files[data_source])
            lons = sample_file.variables['lon'][:]
            lats = sample_file.variables['lat'][:]
            return lats, lons
    else:
        return None


if __name__ == '__main__':
    daily_index_df = build_daily_sst_index()
    """
    for index, row in daily_index_df.sort_index().iterrows():
        print(index)
        print(row['file_name'])
    """
    # Alright. Now we will actually compute the daily SST anomalies
    # We will loop through each date and use the methods that we've written up to
    # now to accomplish the task
    """
    for date_index, row in daily_index_df.sort_index().iterrows():
        print('Current Date: {0}-{1}-{2}'.format(date_index.year, date_index.month, date_index.day))
        dataset = Dataset(row['file_name'])
        date_sst = dataset.variables['analysed_sst'][0]
        current_date = datetime.date(date_index.year, date_index.month, date_index.day)
        daily_climatology = compute_daily_climatology(data_source='JPL_GHRSST', date=current_date)
        print('Shape: {0}'.format(daily_climatology.shape))
        current_anomaly = date_sst - daily_climatology
        print(np.ma.where(current_anomaly >= 1.0))
        print('--------------------------------')
    """
    mmm_sst_values = assign_maximum_sst_values(data_source='JPL_GHRSST')
    print(mmm_sst_values)
    mmm_sst_values -= 273.15

    # Now we will do our first overall test
    # Get the reef bounds and the reef names
    reef_bounds = get_bound_limits()
    print(reef_bounds.keys())
    vankalai_reef_bounds = reef_bounds['Vankalai Reef']
    vankalai_reef_patch = bounding_box_contents(vankalai_reef_bounds)
    jpl_ghrsst_lats, jpl_ghrsst_lons = get_lat_lons(data_source='JPL_GHRSST')
    vankalai_overlapping_coords = get_overlapping_coords(jpl_ghrsst_lats, jpl_ghrsst_lons, vankalai_reef_patch)
    print(vankalai_overlapping_coords)
    print(jpl_ghrsst_lats[354])
    print(jpl_ghrsst_lons[162])

    # We will select one point: lat=354,lon=162 and check for it's mmm_sst and observe for bleaching activities
    """
    vankalai_mmm_sst = mmm_sst_values[354, 162]
    print(mmm_sst_values.shape)
    print(vankalai_mmm_sst)
    vankalai_historical_sst = []
    vankalai_timeline = []
    for date_index, row in daily_index_df.sort_index().iterrows():
        print('Current Date: {0}-{1}-{2}'.format(date_index.year, date_index.month, date_index.day))
        dataset = Dataset(row['file_name'])
        date_sst = dataset.variables['analysed_sst'][0]
        vankalai_historical_sst.append(date_sst[354, 162])
        current_date = date_index.date()
        vankalai_timeline.append(current_date)
    mmm_sst_y_plot_line = [vankalai_mmm_sst for x in range(len(vankalai_historical_sst))]
    bleaching_line = [vankalai_mmm_sst+1.0 for x in range(len(vankalai_historical_sst))]
    """
    """
    plt_dates = date2num(vankalai_timeline)
    plt.plot_date(plt_dates, mmm_sst_y_plot_line, linestyle='--', color='blue')
    plt.plot_date(plt_dates, bleaching_line, linestyle=':', color='red')
    plt.plot_date(plt_dates, vankalai_historical_sst, linestyle='-', color='black')
    plt.show()
    """
    """
    average_vankalai_mmm_sst = mmm_sst_values[[vankalai_overlapping_coords[:, 0], vankalai_overlapping_coords[:, 1]]]
    print('Shape of this array: {0}'.format(average_vankalai_mmm_sst.shape))
    print(average_vankalai_mmm_sst)
    print(average_vankalai_mmm_sst.mean())
    mean_vankalai_mmm_sst = average_vankalai_mmm_sst.mean()
    mean_vankalai_historical_sst = []
    for date_index, row in daily_index_df.sort_index().iterrows():
        print('Current Date: {0}-{1}-{2}'.format(date_index.year, date_index.month, date_index.day))
        dataset = Dataset(row['file_name'])
        date_sst = dataset.variables['analysed_sst'][0]
        mean_vankalai_historical_sst.append(date_sst[vankalai_overlapping_coords[:, 0], vankalai_overlapping_coords[:, 1]].mean())
    """
    """
    mean_mmm_sst_y_plot_line = [mean_vankalai_mmm_sst for x in range(len(mean_vankalai_historical_sst))]
    bleaching_line_2 = [mean_vankalai_mmm_sst+1.0 for x in range(len(mean_vankalai_historical_sst))]
    plt.plot_date(plt_dates, mean_mmm_sst_y_plot_line, linestyle='--', color='blue')
    plt.plot_date(plt_dates, bleaching_line_2, linestyle=':', color='red')
    plt.plot_date(plt_dates, mean_vankalai_historical_sst, linestyle='-', color='black')
    plt.show()
    """

    # What you should do to check whether we are getting the same values
    # Construct 2 dataframes from both specific location and overall location
    # Then set the date as index and then splice the dates to known bleaching episodes

    # We'll just do the code from scratch. Too many iterations and too much junk in there to go back to
    # This list will hold SST values over time for a single point (354, 162)
    vankalai_one_point_sst = []
    # This list will hold mean SST value over time for a set of points specific to a coral reef patch
    vankalai_average_sst = []
    # This list will hold datetime objects for all dates from which temperatures were recorded
    vankalai_timeline = []
    # Looping through each row in the dataframe object
    for date_index, row in daily_index_df.sort_index().iterrows():
        dataset = Dataset(row['file_name'])
        # Getting the SST of a specific point for 'vankalai_one_point_sst' list
        date_sst = dataset.variables['analysed_sst'][0]
        date_sst -= 273.15
        one_point_sst = date_sst[354, 162]
        # Getting mean SST of specific set of points for 'vankalai_average_sst' list
        mean_sst = date_sst[vankalai_overlapping_coords[:, 0], vankalai_overlapping_coords[:, 1]].mean()
        # Appending these values to relevant list
        vankalai_one_point_sst.append(one_point_sst)
        vankalai_average_sst.append(mean_sst)
        # Then append the current date to the 'vankalai_timeline' list
        vankalai_timeline.append(date_index.date())

    # Now we must calculate the Maximum Monthly Mean SST for the specific point as well
    # as for a collection of points
    vankalai_one_point_mmm_sst = mmm_sst_values[354, 162]
    vankalai_average_mmm_sst = \
        mmm_sst_values[vankalai_overlapping_coords[:, 0], vankalai_overlapping_coords[:, 1]].mean()
    print(mmm_sst_values[vankalai_overlapping_coords[:, 0], vankalai_overlapping_coords[:, 1]])
    print('Maximum Monthly Mean SST at one point: {0:.2f} deg. Celsius'.format(vankalai_one_point_mmm_sst))
    print('Mean Maximum Monthly Mean SST at all points: {0:.2f} deg. Celsius'.format(vankalai_average_mmm_sst))



    kalpitiya_reef_bounds = reef_bounds['Kalpitiya Bar Reef']
    kalpitiya_reef_patch = bounding_box_contents(kalpitiya_reef_bounds)
    kalpitiya_overlapping_coords = get_overlapping_coords(jpl_ghrsst_lats, jpl_ghrsst_lons, kalpitiya_reef_patch)
    print(kalpitiya_overlapping_coords)
    #Overlapping coord for Kalpitiya = 292, 153

    # We'll just do the code from scratch. Too many iterations and too much junk in there to go back to
    # This list will hold mean SST value over time for a set of points specific to a coral reef patch
    kalpitiya_average_sst = []
    # This list will hold datetime objects for all dates from which temperatures were recorded
    kalpitiya_timeline = []
    # Looping through each row in the dataframe object
    for date_index, row in daily_index_df.sort_index().iterrows():
        dataset = Dataset(row['file_name'])
        # Getting the SST of a specific point for 'vankalai_one_point_sst' list
        date_sst = dataset.variables['analysed_sst'][0]
        date_sst -= 273.15
        one_point_sst = date_sst[292, 153]
        # Getting mean SST of specific set of points for 'vankalai_average_sst' list
        mean_sst = date_sst[kalpitiya_overlapping_coords[:, 0], kalpitiya_overlapping_coords[:, 1]].mean()
        # Appending these values to relevant list
        kalpitiya_average_sst.append(mean_sst)
        # Then append the current date to the 'vankalai_timeline' list
        kalpitiya_timeline.append(date_index.date())

    # Now we must calculate the Maximum Monthly Mean SST for the specific point as well
    # as for a collection of points
    kalpitiya_average_mmm_sst = \
        mmm_sst_values[kalpitiya_overlapping_coords[:, 0], kalpitiya_overlapping_coords[:, 1]].mean()
    print(mmm_sst_values[kalpitiya_overlapping_coords[:, 0], kalpitiya_overlapping_coords[:, 1]])
    # print('Maximum Monthly Mean SST at one point: {0:.2f} deg. Celsius'.format(kalpitiya_one_point_mmm_sst))
    print('Mean Maximum Monthly Mean SST at all points: {0:.2f} deg. Celsius'.format(kalpitiya_average_mmm_sst))

    # Punkudutivu Reef
    punkudutivu_reef_bounds = reef_bounds['Punkudutivu Reef']
    punkudutivu_reef_patch = bounding_box_contents(punkudutivu_reef_bounds)
    punkudutivu_overlapping_coords = get_overlapping_coords(jpl_ghrsst_lats, jpl_ghrsst_lons, punkudutivu_reef_patch)
    print(punkudutivu_overlapping_coords)
    # Overlapping coord for Kalpitiya = 292, 153

    # We'll just do the code from scratch. Too many iterations and too much junk in there to go back to
    # This list will hold mean SST value over time for a set of points specific to a coral reef patch
    punkudutivu_average_sst = []
    # This list will hold datetime objects for all dates from which temperatures were recorded
    punkudutivu_timeline = []
    # Looping through each row in the dataframe object
    for date_index, row in daily_index_df.sort_index().iterrows():
        dataset = Dataset(row['file_name'])
        # Getting the SST of a specific point for 'vankalai_one_point_sst' list
        date_sst = dataset.variables['analysed_sst'][0]
        date_sst -= 273.15
        one_point_sst = date_sst[292, 153]
        # Getting mean SST of specific set of points for 'vankalai_average_sst' list
        mean_sst = date_sst[punkudutivu_overlapping_coords[:, 0], punkudutivu_overlapping_coords[:, 1]].mean()
        # Appending these values to relevant list
        punkudutivu_average_sst.append(mean_sst)
        # Then append the current date to the 'vankalai_timeline' list
        punkudutivu_timeline.append(date_index.date())

    # Now we must calculate the Maximum Monthly Mean SST for the specific point as well
    # as for a collection of points
    punkudutivu_average_mmm_sst = \
        mmm_sst_values[punkudutivu_overlapping_coords[:, 0], punkudutivu_overlapping_coords[:, 1]].mean()
    print(mmm_sst_values[punkudutivu_overlapping_coords[:, 0], punkudutivu_overlapping_coords[:, 1]])
    # print('Maximum Monthly Mean SST at one point: {0:.2f} deg. Celsius'.format(kalpitiya_one_point_mmm_sst))
    print('Mean Maximum Monthly Mean SST at all points: {0:.2f} deg. Celsius'.format(punkudutivu_average_mmm_sst))

    # Trincomalee Reefs
    trincomalee_reef_bounds = reef_bounds['Trincomalee Reefs']
    trincomalee_reef_patch = bounding_box_contents(trincomalee_reef_bounds)
    trincomalee_overlapping_coords = get_overlapping_coords(jpl_ghrsst_lats, jpl_ghrsst_lons, trincomalee_reef_patch)
    print(trincomalee_overlapping_coords)
    # Overlapping coord for Kalpitiya = 292, 153

    # We'll just do the code from scratch. Too many iterations and too much junk in there to go back to
    # This list will hold mean SST value over time for a set of points specific to a coral reef patch
    trincomalee_average_sst = []
    # This list will hold datetime objects for all dates from which temperatures were recorded
    trincomalee_timeline = []
    # Looping through each row in the dataframe object
    for date_index, row in daily_index_df.sort_index().iterrows():
        dataset = Dataset(row['file_name'])
        # Getting the SST of a specific point for 'vankalai_one_point_sst' list
        date_sst = dataset.variables['analysed_sst'][0]
        date_sst -= 273.15
        one_point_sst = date_sst[0, 0]
        # Getting mean SST of specific set of points for 'vankalai_average_sst' list
        mean_sst = date_sst[trincomalee_overlapping_coords[:, 0], trincomalee_overlapping_coords[:, 1]].mean()
        # Appending these values to relevant list
        trincomalee_average_sst.append(mean_sst)
        # Then append the current date to the 'vankalai_timeline' list
        trincomalee_timeline.append(date_index.date())

        # Now we must calculate the Maximum Monthly Mean SST for the specific point as well
        # as for a collection of points
    trincomalee_average_mmm_sst = \
            mmm_sst_values[trincomalee_overlapping_coords[:, 0], trincomalee_overlapping_coords[:, 1]].mean()
    print(mmm_sst_values[trincomalee_overlapping_coords[:, 0], trincomalee_overlapping_coords[:, 1]])
    # print('Maximum Monthly Mean SST at one point: {0:.2f} deg. Celsius'.format(kalpitiya_one_point_mmm_sst))
    print('Mean Maximum Monthly Mean SST at all points: {0:.2f} deg. Celsius'.format(trincomalee_average_mmm_sst))

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.style.use('grayscale')
    plt.subplot(4, 1, 1)
    number_of_records = len(vankalai_average_sst)
    plt.plot(vankalai_timeline, vankalai_average_sst, linestyle='-', color='black', linewidth=0.75)
    plt.plot(vankalai_timeline, [vankalai_average_mmm_sst for x in range(number_of_records)], linestyle=':',
             color='cornflowerblue', linewidth=1.5)
    plt.plot(vankalai_timeline, [vankalai_average_mmm_sst + 1.0 for x in range(number_of_records)], linestyle='-.',
             color='#ff1a1a', linewidth=1.5)
    plt.ylim(25, 34)
    plt.title('Deviation of SST across 15 years over Vankalai Reef System')

    # Plotting the data
    plt.subplot(4, 1, 2)
    number_of_records = len(kalpitiya_average_sst)
    plt.plot(kalpitiya_timeline, kalpitiya_average_sst, linestyle='-', color='black', linewidth=0.75)
    plt.plot(kalpitiya_timeline, [kalpitiya_average_mmm_sst for x in range(number_of_records)], linestyle=':',
             color='cornflowerblue', linewidth=1.5)
    plt.plot(kalpitiya_timeline, [kalpitiya_average_mmm_sst + 1.0 for x in range(number_of_records)], linestyle='-.',
             color='#ff1a1a', linewidth=1.5)
    plt.ylim(25, 34)
    plt.title('Deviation of SST across 15 years over Kalpitiya Reef System')

    plt.subplot(4, 1, 3)
    number_of_records = len(punkudutivu_average_sst)
    plt.plot(punkudutivu_timeline, punkudutivu_average_sst, linestyle='-', color='black', linewidth=0.75)
    plt.plot(punkudutivu_timeline, [punkudutivu_average_mmm_sst for x in range(number_of_records)], linestyle=':',
             color='cornflowerblue', linewidth=1.5)
    plt.plot(punkudutivu_timeline, [punkudutivu_average_mmm_sst + 1.0 for x in range(number_of_records)], linestyle='-.',
             color='#ff1a1a', linewidth=1.5)
    plt.ylim(25, 34)
    plt.title('Deviation of SST across 15 years over Punkudutivu Reef System')

    plt.subplot(4, 1, 4)
    number_of_records = len(trincomalee_average_sst)
    plt.plot(trincomalee_timeline, trincomalee_average_sst, linestyle='-', color='black', linewidth=0.75)
    plt.plot(trincomalee_timeline, [trincomalee_average_mmm_sst for x in range(number_of_records)], linestyle=':',
             color='cornflowerblue', linewidth=1.5)
    plt.plot(trincomalee_timeline, [trincomalee_average_mmm_sst + 1.0 for x in range(number_of_records)],
             linestyle='-.',
             color='#ff1a1a', linewidth=1.5)
    plt.ylim(25, 34)
    plt.title('Deviation of SST across 15 years over Trincomalee Reef System')

    plt.show()
