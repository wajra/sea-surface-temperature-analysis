# sea_surface_temperature_analysis
Code for analysing SST trends over coral reefs of Sri Lanka over 3 decades to identify potential locations for conservation measures

This code analyses sea surface temperature trends over Sri Lankan coral reefs and calculates the following

* Monthly Mean SST values (MMSST)
* Maximum Monthly Mean SST values (MMMSST)
* Degree Heating Weeks over all grids to identify short term and long term thermal stress (DHW)
* Plots DHW trends for selected coral reefs during 1998, 2010, and 2016 El-Nino events
* Ranks and plots coral reefs for future conservation measures by their resilience to thermal stress trends

0.25°x0.25° SST grid data from NOAA [Link](https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html) was used for calculations.
A sample data file is uploaded to the respository as well.
I can send the complete data files over for you to run the same analysis.

## Prerequisite packages for running the code

* Pandas
* Scipy
* Numpy
* Matplotlib
* NetCDF4
