from netCDF4 import Dataset, num2date
import glob
import os
import os.path
from pandas import DataFrame
import numpy as np
import re

write_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/JPL_GHRSST/MMM_SST',
                   'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/NOAA_OI_SST/MMM_SST'}

read_locations = {'JPL_GHRSST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/JPL_GHRSST/MM_SST',
                  'NOAA_OI_SST': 'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DATA_SOURCE/NOAA_OI_SST/MM_SST'}

def compute_maximum_monthly_mean_sst(data_source = 'JPL_GHRSST'):
    all_climatologies = []
    if data_source in ['JPL_GHRSST', 'NOAA_OI_SST']:
        if data_source == 'JPL_GHRSST':
            read_location = read_locations[data_source]
            write_location = write_locations[data_source]
            # A regular expression to filter out the necessary files
            regex_1 = re.compile(r'MM_SST_JPL_GHRSST_\d\d')
            for file in os.listdir(read_location):
                # Match the regular expression to filter the files that we need
                # In all honesty, this is just a precaution
                if regex_1.match(file):
                    print(file)
                    all_climatologies.append(np.load('{0}/{1}'.format(read_location, file)))
            # Now all 12 climatological files are appended
            # Then we can get the maximum means for each pixel
            mmt_sst_values = np.maximum.reduce(all_climatologies)
            print(mmt_sst_values.shape)
            masked_values = mmt_sst_values == -32768.0
            mmt_sst_values_masked = np.ma.masked_array(mmt_sst_values, masked_values)
            print(mmt_sst_values_masked)
            # Writing maximum monthly mean temperature to the disk
            mmt_sst_values_masked.dump('{0}/{1}_MMM_SST'.format(write_locations[data_source], data_source))
            print('Wrote the file to disk')
        elif data_source == 'NOAA_OI_SST':
            read_location = read_locations[data_source]
            write_location = write_locations[data_source]
            # A regular expression to filter out the necessary files
            regex_1 = re.compile(r'MM_SST_NOAA_OI_SST_\d\d')
            for file in os.listdir(read_location):
                # Match the regular expression to filter the files that we need
                # In all honesty, this is just a precaution
                if regex_1.match(file):
                    print(file)
                    all_climatologies.append(np.load('{0}/{1}'.format(read_location, file)))
            # Now all 12 climatological files are appended
            # Then we can get the maximum means for each pixel
            mmt_sst_values = np.maximum.reduce(all_climatologies)
            print(mmt_sst_values.shape)
            masked_values = mmt_sst_values < 0
            mmt_sst_values_masked = np.ma.masked_array(mmt_sst_values, masked_values)
            print(mmt_sst_values_masked)
            # Writing maximum monthly mean temperature to the disk
            mmt_sst_values_masked.dump('{0}/{1}_MMM_SST'.format(write_locations[data_source], data_source))
            print('Wrote the file to disk')
