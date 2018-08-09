from netCDF4 import Dataset, num2date
import glob
import os.path
from pandas import DataFrame
import numpy as np
import pandas as pd

CSV_WRITE_PATH = 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES'
WRITE_LOCATION_1 = 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/MONTHLY_CLIMATOLOGIES'
WRITE_LOCATION_2 = 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/MONTHLY_CLIMATOLOGIES_2'

DATAFILE_PATH_1 = 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/JPL_TEMP_DATASET_1'

write_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/JPL_GHRSST/MM_SST',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/NOAA_OI_SST/MM_SST'}

read_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/JPL_TEMP_DATASET_1',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/NOAA_OI_SST_DATA'}

mm_sst_timeframes = {'JPL_GHRSST': {'start': '2002-06-01', 'end': '2009-07-01'},
                     'NOAA_OI_SST': {'start': '1985-01-01', 'end': '1994-01-01', 'drop': ['1991', '1992']}}


def get_monthly_index():
    month_index = [x for x in range(1, 13)]
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                  'September', 'October', 'November', 'December']
    return zip(month_index, month_list)


def compute_monthly_mean_sst(data_source='JPL_GHRSST'):
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            read_location = read_locations[data_source]
            write_location = write_locations[data_source]
            if os.path.exists(read_location) and os.path.exists(write_location):
                print('We can proceed')
                all_data = {'date': [], 'sst': []}
                all_nc_files = glob.glob(read_location+'/*-fv04-MUR_subset.nc')
                for file in all_nc_files:
                    dataset = Dataset(file)
                    time = dataset.variables['time']
                    current_date = num2date(time[:], time.units, calendar='standard')[0]
                    current_sst = dataset.variables['analysed_sst'][0]
                    all_data['date'].append(current_date)
                    all_data['sst'].append(current_sst)
                # Then we load all of this into a Pandas Dataframe for easy indexing
                sst_dataframe = DataFrame.from_dict(all_data)
                print('Create the Pandas Dataframe')
                # Then we set the index of the dataframe to datetime object
                sst_dataframe.index = sst_dataframe['date']
                # Then we can delete the original 'date' column
                del(sst_dataframe['date'])
                # Then we filter the data we need to compute MM_SST
                # We do this by using Pandas indexing functions for datetime objects
                sst_range_dataframe = \
                    sst_dataframe[mm_sst_timeframes[data_source]['start']:mm_sst_timeframes[data_source]['end']]
                # We can delete the original dataframe
                del sst_dataframe
                print('Original Pandas dataframe deleted')

                # Now we can start computing monthly means and write them on to the disk
                for m, m_name in get_monthly_index():
                    # First we index all the rows relevant to a specific month
                    # We will loop through each month and create means for each of them
                    month_sst_dataframe = sst_range_dataframe[sst_range_dataframe.index.month == m]
                    # Calculate the monthly mean
                    month_mean_sst = month_sst_dataframe['sst'].mean(axis=0)
                    # Then we write this 2D array to the disk
                    month_file_write_path = '{0}/MM_SST_{1}_{2:02d}'.format(write_locations[data_source], data_source, m)
                    month_mean_sst.dump(month_file_write_path)
                    print('Wrote the file for {0} to disk'.format(m_name))
                print('Wrote all the files to disk')
            return 1
        elif data_source == 'NOAA_OI_SST':
            read_location = read_locations[data_source]
            write_location = write_locations[data_source]
            if os.path.exists(read_location) and os.path.exists(write_location):
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
                        print(current_sst.shape)
                # Then we load all of this into a Pandas Dataframe for easy indexing
                sst_dataframe = DataFrame.from_dict(all_data)
                print('Create the Pandas Dataframe')
                # Then we set the index of the dataframe to datetime object
                sst_dataframe.index = sst_dataframe['date']
                # Then we can delete the original 'date' column
                # del(sst_dataframe['date'])
                # Then we filter the data we need to compute MM_SST
                # We do this by using Pandas indexing functions for datetime objects
                sst_range_dataframe = \
                            sst_dataframe[mm_sst_timeframes[data_source]['start']:mm_sst_timeframes[data_source]['end']]
                # sst_range_dataframe.drop(sst_range_dataframe.index['1992-01-01': '1994-01-01'], inplace= True)
                # print(sst_range_dataframe['1991-01-01': '1993-01-01'])
                # sst_range_dataframe.drop(sst_range_dataframe['1991-01-01':'1993-01-01'])
                # sst_range_dataframe = sst_range_dataframe[(sst_range_dataframe.index.year != 1991) | (sst_range_dataframe.index!=1992)]
                # sst_range_dataframe.drop('date', sst_range_dataframe.date['1991-01-01':'1993-01-01'], inplace=True)
                # sst_range_dataframe = sst_range_dataframe[(sst_range_dataframe.index.year != 1991) | (sst_range_dataframe.index.year != 1992)]
                # Use this method to drop the dates
                dataframe_1 = sst_range_dataframe['1985-01-01':'1990-12-31']
                dataframe_2 = sst_range_dataframe['1993-01-01':]
                final_sst_range_dataframe = pd.concat([dataframe_1, dataframe_2])

                # Now we can start computing monthly means and write them on to the disk
                for m, m_name in get_monthly_index():
                    # First we index all the rows relevant to a specific month
                    # We will loop through each month and create means for each of them
                    month_sst_dataframe = final_sst_range_dataframe[final_sst_range_dataframe.index.month == m]
                    month_mean_sst = month_sst_dataframe['sst'].mean(axis=0)
                    print(month_mean_sst)
                    print(month_mean_sst.shape)
                    # Then we write this 2D array to the disk
                    month_file_write_path = '{0}/MM_SST_{1}_{2:02d}'.format(write_locations[data_source], data_source,
                                                                            m)
                    month_mean_sst.dump(month_file_write_path)
                    print('Wrote the file for {0} to disk'.format(m_name))
                print('Wrote all the files to disk')
            return 1
        else:
            return 0
