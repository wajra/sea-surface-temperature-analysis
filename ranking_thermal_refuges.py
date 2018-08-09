from datetime import timedelta
import datetime
import re
import glob
import numpy as np
import os.path
from netCDF4 import Dataset, num2date
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
import matplotlib
from collections import defaultdict

def compute_score(row):
    score = row['max_dhw_2016'] + row['mean_dhw_last_3'] + row['decadal_change']*10
    return score


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(1 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

if __name__ == '__main__':
    coral_reef_dhw_history = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES/all_coral_reef_histories_formatted.xlsx')
    coral_reef_trends = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/individual_reefs_sst_trends_saved.xlsx')
    all_coral_reef_dhw_history = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/DHW_HISTORIES_ALL_YEARS/all_coral_reef_histories_rounded.xlsx')

    bleaching_stress_events = {'reef': [], 'events_decade': [], 'event_number': [], 'max_dhw_history': [], 'mean_dhw_history': []}

    all_coral_reef_dhw_history = all_coral_reef_dhw_history[all_coral_reef_dhw_history.max_dhw >= 4.0]

    print(all_coral_reef_dhw_history[all_coral_reef_dhw_history.max_dhw >= 8.0][['reef', 'year']])

    """

    print('Total number of bleaching stress level events since 1982: {0}'.format(all_coral_reef_dhw_history.shape[0]))
    print('Total number of bleaching stress level events in 1998: {0}'.format(
        all_coral_reef_dhw_history[all_coral_reef_dhw_history.year == 1998].shape[0]))
    print('Total number of bleaching stress level events in 2010: {0}'.format(
        all_coral_reef_dhw_history[all_coral_reef_dhw_history.year == 2010].shape[0]))
    print('Total number of bleaching stress level events in 2016: {0}'.format(
        all_coral_reef_dhw_history[all_coral_reef_dhw_history.year == 2016].shape[0]))

    for year, year_df in all_coral_reef_dhw_history.groupby(all_coral_reef_dhw_history.year):
        print(year)
        print('Number of events in {0}: {1}'.format(year, year_df.shape[0]))
        print(year_df.reef.unique())
        print(year_df['max_dhw'])
    """
    for reef, reef_df in all_coral_reef_dhw_history.groupby(all_coral_reef_dhw_history.reef):
        print(reef)
        events_per_decade = (reef_df.shape[0]/34)*10
        print(events_per_decade)
        bleaching_stress_events['reef'].append(reef)
        bleaching_stress_events['events_decade'].append(events_per_decade)
        event_number = reef_df.shape[0]
        bleaching_stress_events['event_number'].append(event_number)
        bleaching_stress_events['max_dhw_history'].append(reef_df.max_dhw.max())
        bleaching_stress_events['mean_dhw_history'].append(reef_df.max_dhw.mean())

    bleaching_stress_events_df = DataFrame.from_dict(bleaching_stress_events)

    print(bleaching_stress_events_df)
    """
    # print(coral_reef_dhw_history)
    combined_df = pd.merge(coral_reef_dhw_history, coral_reef_trends, on='reef', how='inner')

    dhw_history_1998 = coral_reef_dhw_history[coral_reef_dhw_history['year'] == 1998].reset_index()
    dhw_history_2010 = coral_reef_dhw_history[coral_reef_dhw_history['year'] == 2010].reset_index()
    dhw_history_2016 = coral_reef_dhw_history[coral_reef_dhw_history['year'] == 2016].reset_index()

    dhw_history_1998.rename(columns={'max_dhw': 'max_dhw_1998'}, inplace=True)
    dhw_history_2010.rename(columns={'max_dhw': 'max_dhw_2010'}, inplace=True)
    dhw_history_2016.rename(columns={'max_dhw': 'max_dhw_2016'}, inplace=True)

    # print(dhw_history_2010)

    histories_combined = pd.merge(dhw_history_1998, dhw_history_2010, on='reef', how='inner')
    # print(histories_combined)

    histories_combined = pd.merge(histories_combined, dhw_history_2016, on='reef', how='inner')
    print(histories_combined)

    histories_combined.to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/historical_dhws_separate_columns.xlsx')

    trends_and_histories = pd.merge(histories_combined, coral_reef_trends, on='reef', how='inner')

    trends_signficance = trends_and_histories.p >= 0.05

    trends_and_histories.loc[trends_signficance, 'decadal_change'] = 0

    print(trends_and_histories['decadal_change'])

    trends_and_histories['mean_dhw_last_3'] = trends_and_histories[['max_dhw_1998', 'max_dhw_2010', 'max_dhw_2016']].mean(axis=1)

    print(trends_and_histories['mean_dhw_last_3'])

    trends_and_histories['trend_score'] = trends_and_histories.apply(compute_score, axis=1)

    print(trends_and_histories['trend_score'])

    trends_and_histories.to_excel(
        'D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/historical_dhws_and_trends_sep_columns.xlsx')

    """
    ####################

    f, ax = plt.subplots()
    f.set_size_inches((10, 8), forward=True)
    plt.style.use('grayscale')
    scatter_markers = ['s', 'o', (5, 2), 'v', '+', (5, 1)]
    counter = 0
    for year, year_df in all_coral_reef_dhw_history.groupby(all_coral_reef_dhw_history.year):
        plt.scatter(year_df.max_dhw, year_df.max_hotspot, marker=scatter_markers[counter],
                    facecolor='white', edgecolor='black', s=40, label=str(year))
        counter += 1
    #plt.grid(color='grey', linestyle='--')
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', top='off')
    ax.tick_params(axis='y', which='both', right='off')
    plt.legend(loc=2, fontsize=12)
    plt.tight_layout()
    plt.savefig('D:/zzzzz.png', dpi=600)

    ##################################################################################

    coral_reefs_annual_trends = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/SST_TRENDS_VER_1/individual_reefs_sst_trends.xlsx')
    coral_reefs_hot_months_trends = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/SST_TRENDS_VER_1/individual_reefs_hot_months_sst_trends.xlsx')
    coral_reefs_cool_months_trends = pd.read_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/SST_TRENDS_VER_1/individual_reefs_cool_months_sst_trends.xlsx')

    # print(coral_reefs_annual_trends)
    # print(coral_reefs_hot_months_trends)

    individual_merge_df = pd.merge(coral_reefs_annual_trends, coral_reefs_hot_months_trends, on='reef', how='inner')
    individual_merge_df = pd.merge(individual_merge_df, coral_reefs_cool_months_trends, on='reef', how='inner')

    # print('Cool season decadal change in reefs')
    # print(individual_merge_df['ind_cool_season_decadal_change'])

    #################################################
    # Now we'll print some statistics from these trends

    total_number_of_reefs = individual_merge_df.reef.shape[0]
    # print('Total number of reefs: {0}'.format(total_number_of_reefs))

    # Percentage of reefs showing positive increase in SST annually over time
    print('Reefs showing positive increase in annual SST trends')
    reefs_positive_annual_trends = individual_merge_df[individual_merge_df['ind_decadal_change'] > 0]
    print(reefs_positive_annual_trends.reef.shape[0])
    print('Reefs showing positive and significant increase in annual SST trends')
    reefs_positive_significant_annual_trends = individual_merge_df[(individual_merge_df['ind_annual_p'] < 0.05) & (individual_merge_df['ind_decadal_change'] > 0)]
    print(reefs_positive_significant_annual_trends.shape[0])
    print('Number of reefs showing annual positive SST trends: {0}'.format(reefs_positive_annual_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_annual_trends.shape[0]/total_number_of_reefs * 100))
    print('Number of reefs showing annual positive and significant SST trends: {0}'.format(reefs_positive_significant_annual_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_significant_annual_trends.shape[0]/total_number_of_reefs*100))
    print('Reefs that didn\'t show annual positive and significant SST trends: {0}'.format(individual_merge_df[(individual_merge_df['ind_annual_p'] > 0.05) & (individual_merge_df['ind_decadal_change'] > 0)].reef))



    # Percentage of reefs showing positive increase in SST in hot seasons over time
    print('Reefs showing positive increase in summertime SST trends')
    reefs_positive_summertime_trends = individual_merge_df[individual_merge_df['ind_hot_season_decadal_change'] > 0]
    print(reefs_positive_summertime_trends.reef.shape[0])
    print('Number of reefs showing positive trends in summertime SST: {0}'.format(reefs_positive_summertime_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_summertime_trends.shape[0]/total_number_of_reefs*100))
    # print('Reef that didn\'t show positive trends in summertime: {0}'.format(individual_merge_df[individual_merge_df['ind_hot_season_decadal_change'] < 0].ind_hot_season_decadal_change))
    print('Reefs showing positive and significant increase in summertime SST trends')
    reefs_positive_significant_summertime_trends = individual_merge_df[(individual_merge_df['ind_hot_months_p'] < 0.05)
                                                                       & (individual_merge_df['ind_hot_season_decadal_change'] >0)]
    print(reefs_positive_significant_summertime_trends.shape[0])
    print('Number of reefs showing positive and significant summertime SST trends: {0}'.format(reefs_positive_significant_summertime_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_significant_summertime_trends.shape[0]/total_number_of_reefs*100))


    # Percentage of reefs showing positive increase in SST in cool season over time
    print('Reefs showing positive increase in winter SST trends')
    reefs_positive_winter_trends = individual_merge_df[individual_merge_df['ind_cool_season_decadal_change'] > 0]
    print(reefs_positive_winter_trends.reef.shape[0])
    print('Number of reefs showing positive increase in winter SST trends: {0}'.format(reefs_positive_winter_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_winter_trends.shape[0]/total_number_of_reefs*100))
    print('Reefs showing positive and significant increase in winter SST trends')
    reefs_positive_and_significant_winter_trends = individual_merge_df[(individual_merge_df['ind_cool_months_p'] < 0.05) & (individual_merge_df['ind_cool_season_decadal_change'] > 0)]
    print(reefs_positive_and_significant_winter_trends.reef.shape[0])
    print('Number of reefs showing positive and significant winter SST trend: {0}'.format(reefs_positive_and_significant_winter_trends.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_positive_and_significant_winter_trends.shape[0]/total_number_of_reefs*100))
    print('Reefs which showed positive and significant winter SST trends}')
    print(reefs_positive_and_significant_winter_trends.reef)

    # Reefs that showed a higher summer SST trend compared to annual SST trend
    reefs_higher_summertime_trend = individual_merge_df[individual_merge_df['ind_hot_season_sst_increase'] > individual_merge_df['ind_sst_increase']]
    print('Reefs that have shown greater increase in summertime vs annually')
    print('Number of reefs showing higher summertime trends: {0}'.format(reefs_higher_summertime_trend.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_higher_summertime_trend.shape[0]/total_number_of_reefs*100))
    print(reefs_higher_summertime_trend.reef)

    # Reefs that showed a higher winter SST trend compared to annual SST trend
    reefs_higher_winter_trend = individual_merge_df[individual_merge_df['ind_cool_season_sst_increase'] > individual_merge_df['ind_sst_increase']]
    print('Reefs that have shown greater increase in winter vs annually')
    print(reefs_higher_winter_trend.reef)
    print('Number of reefs showing higher winter trends: {0}'.format(reefs_higher_winter_trend.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_higher_winter_trend.shape[0]/total_number_of_reefs*100))

    # Reefs that showed a higher winter SST trend compared to hot months SST trends
    reefs_higher_hot_vs_winter_trend = individual_merge_df[individual_merge_df['ind_cool_season_sst_increase'] > individual_merge_df['ind_hot_season_sst_increase']]
    print(reefs_higher_hot_vs_winter_trend.reef)
    print('Number of reefs that showed a higher cool months trends compared to winter trends: {0}'.format(reefs_higher_hot_vs_winter_trend.shape[0]))
    print('Percentage: {0:.2f}%'.format(reefs_higher_hot_vs_winter_trend.shape[0]/total_number_of_reefs*100))

    cool_minus_warm = reefs_higher_hot_vs_winter_trend['ind_cool_season_decadal_change'] - reefs_higher_hot_vs_winter_trend['ind_hot_season_decadal_change']

    print(cool_minus_warm.mean())

    print('--------\n'*3)
    print(individual_merge_df[['region', 'reef', 'ind_cool_season_decadal_change', 'ind_hot_season_decadal_change']])

    print('Getting the spatial mean of annual SST trend')
    print('Annual mean SST trend for all reefs: {0:.2f} +- ({1:.2f})'.format(individual_merge_df['ind_decadal_change'].mean(), individual_merge_df['ind_decadal_change'].std(ddof=1)))
    print('Cool season mean SST trend for all reefs: {0:.2f} +- ({1:.2f})'.format(individual_merge_df['ind_cool_season_decadal_change'].mean(), individual_merge_df['ind_cool_season_decadal_change'].std(ddof=1)))
    print('Warm season mean SST trend for all reefs: {0:.2f} +- ({1:.2f})'.format(individual_merge_df['ind_hot_season_decadal_change'].mean(), individual_merge_df['ind_hot_season_decadal_change'].std(ddof=1)))

    ##################################################################
    # Number of mass mortality inducing thermal stress events
    print('Number of bleaching level thermal stress events')
    print(all_coral_reef_dhw_history.shape[0])

    print('Number of mass mortality inducing events: {0}'.format(all_coral_reef_dhw_history[all_coral_reef_dhw_history.max_dhw >= 8].shape[0]))
    # print(all_coral_reef_dhw_history[all_coral_reef_dhw_history.max_dhw >= 8])

    print(pearsonr(all_coral_reef_dhw_history.max_hotspot, all_coral_reef_dhw_history.max_dhw))

    all_coral_reef_dhw_history.sort(['year']).to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/SST_TRENDS_VER_1/all_bleaching_level_stress_events.xlsx')

    # Grouping and counting the number of bleaching level thermal stress events in each coral reef

    number_of_bleaching_events = []
    for coral_reef, reef_df in all_coral_reef_dhw_history.groupby(all_coral_reef_dhw_history.reef):
        # print(coral_reef)
        print('Number of bleaching level stress events since 1982 for {0}: {1}'.format(
            coral_reef, reef_df.shape[0]
        ))
        number_of_bleaching_events.append(reef_df.shape[0])

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.style.use('grayscale')
    plot_data = individual_merge_df['ind_decadal_change']
    x_bins = np.arange(plot_data.min(), plot_data.max(), 0.01)
    ax[0, 0].hist(plot_data, bins=x_bins, rwidth=0.75, color='grey', lw=0, normed=True)
    ax[0, 0].set_xticks(np.arange(0.02, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].tick_params(axis='x', which='both', top='off', direction='out')
    ax[0, 0].tick_params(axis='y', which='both', right='off', direction='out')
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax[0, 0].xaxis.set_major_locator(major_locator)
    ax[0, 0].xaxis.set_major_formatter(major_formatter)
    ax[0, 0].xaxis.set_minor_locator(minor_locator)
    ax[0, 0].set_xlabel('Annual SST Trend ($^\circ$C/decade)')
    # ax[0, 0].annotate('(a)', (0.14, 30))
    plt.sca(ax[0, 0])
    plt.xticks(rotation=0)


    # Next subplot
    plot_data_2 = individual_merge_df['ind_hot_season_decadal_change']
    x_bins_2 = np.arange(-0.04, plot_data_2.max(), 0.01)
    ax[0, 1].hist(plot_data_2, bins=x_bins_2, rwidth=0.75, color='grey', lw=0, normed=True)
    ax[0, 1].set_xticks(np.arange(-0.04, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].tick_params(axis='x', which='both', top='off', direction='out')
    ax[0, 1].tick_params(axis='y', which='both', right='off', direction='out')
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax[0, 1].xaxis.set_major_locator(major_locator)
    ax[0, 1].xaxis.set_major_formatter(major_formatter)
    ax[0, 1].xaxis.set_minor_locator(minor_locator)
    # ax[0, 0].annotate('(a)', (0.14, 30))
    plt.sca(ax[0, 1])
    plt.xticks(rotation=0)

    plot_data_3 = individual_merge_df['ind_cool_season_decadal_change']
    x_bins_3 = np.arange(0.00, plot_data_3.max(), 0.01)
    ax[1, 0].hist(plot_data_3, bins=x_bins_3, rwidth=0.75, color='grey', lw=0, normed=True)
    ax[1, 0].set_xticks(np.arange(0.00, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].tick_params(axis='x', which='both', top='off', direction='out')
    ax[1, 0].tick_params(axis='y', which='both', right='off', direction='out')
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax[1, 0].xaxis.set_major_locator(major_locator)
    ax[1, 0].xaxis.set_major_formatter(major_formatter)
    ax[1, 0].xaxis.set_minor_locator(minor_locator)
    # ax[0, 0].annotate('(a)', (0.14, 30))
    plt.sca(ax[0, 1])
    plt.xticks(rotation=0)

    # Next subplot. Number of bleaching events since 1982
    ax[1, 1].hist(number_of_bleaching_events, bins=np.arange(0, 6, 1), rwidth=0.60, color='grey', lw=0)
    ax[1, 1].set_xticks(np.arange(0.5, 8.5, 1))
    ax[1, 1].set_xticklabels(np.arange(0, 8, 1))
    ax[1, 1].spines['right'].set_visible(False)
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].tick_params(axis='x', which='both', top='off', direction='out')
    ax[1, 1].tick_params(axis='y', which='both', right='off', direction='out')

    plt.tight_layout()
    plt.savefig('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/PHOTOS/thermal_stress_trends_over_time.png', dpi=300)

    """
    # Plotting all of this
    fig, ax = plt.subplots(3, 2, figsize=(12, 6))
    plt.style.use('grayscale')
    x_ticks = np.arange(0.02, 0.18, 0.02)
    ax[0, 0].hist(individual_merge_df['ind_decadal_change'], bins=6,
                  rwidth=0.90, color='grey', normed=1, lw=0)
    ax[0, 0].set_xticks(x_ticks)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].spines['top'].set_visible(False)
    # ax[0, 0].yaxis.set_ticks_position('left')
    # ax[0, 0].xaxis.set_ticks_position('bottom')
    ax[0, 0].tick_params(axis='x', which='both', top='off', direction='out')
    ax[0, 0].tick_params(axis='y', which='both', right='off', direction='out')


    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', top='off')
    ax.tick_params(axis='y', which='both', right='off')


    ax[1, 0].hist(individual_merge_df['ind_hot_season_decadal_change'], bins=6,
                  rwidth=0.75, color='black', alpha=0.5, normed=1)
    x_ticks = np.arange(-0.02, 0.16, 0.02)
    ax[1, 0].set_xticks(x_ticks)
    plt.show()


    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

    plot_data = individual_merge_df['ind_decadal_change']
    x_bins = np.arange(plot_data.min(), plot_data.max(), 0.01)
    ax1.hist(plot_data, bins=x_bins, rwidth=0.75, color='grey', lw=0, normed=True)
    ax1.set_xticks(np.arange(0.02, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', which='both', top='off', direction='out')
    ax1.tick_params(axis='y', which='both', right='off', direction='out')
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax1.xaxis.set_major_locator(major_locator)
    ax1.xaxis.set_major_formatter(major_formatter)
    ax1.xaxis.set_minor_locator(minor_locator)
    # ax[0, 0].annotate('(a)', (0.14, 30))
    plt.sca(ax1)
    plt.xticks(rotation=0)

    # Next subplot
    plot_data_2 = individual_merge_df['ind_hot_season_decadal_change']
    x_bins_2 = np.arange(-0.04, plot_data_2.max(), 0.01)
    ax2.hist(plot_data_2, bins=x_bins_2, rwidth=0.75, color='grey', lw=0, normed=True)
    ax2.set_xticks(np.arange(-0.04, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='x', which='both', top='off', direction='out')
    ax2.tick_params(axis='y', which='both', right='off', direction='out')
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax2.xaxis.set_major_locator(major_locator)
    ax2.xaxis.set_major_formatter(major_formatter)
    ax2.xaxis.set_minor_locator(minor_locator)
    # ax[0, 0].annotate('(a)', (0.14, 30))
    plt.sca(ax2)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/PHOTOS/thermal_trends_over_time_styled.png', dpi=600)
    plt.show()
    """

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax5 = plt.subplot2grid((3, 4), (2, 1), colspan=2)

    # First subplot
    plt.style.use('grayscale')
    plot_data = individual_merge_df['ind_decadal_change']
    x_bins = np.arange(plot_data.min(), plot_data.max(), 0.01)
    ax1.hist(plot_data, bins=x_bins, rwidth=0.75, color='grey', lw=0)
    ax1.set_xticks(np.arange(0.02, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', which='both', top='off', direction='out')
    ax1.tick_params(axis='y', which='both', right='off', direction='out')
    ax1.annotate('(a)', xy=(0.03, 5), fontsize=14, fontweight='bold')
    ax1.annotate('{0:.2f}$\pm${1:.2f} $^\circ$C/decade'.format(individual_merge_df['ind_decadal_change'].mean(),
                                                individual_merge_df['ind_decadal_change'].std(ddof=1)),
                 xy=(0.11, 6), fontsize=12)
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax1.xaxis.set_major_locator(major_locator)
    ax1.xaxis.set_major_formatter(major_formatter)
    ax1.xaxis.set_minor_locator(minor_locator)
    formatter = FuncFormatter(to_percent)
    # ax1.yaxis.set_major_formatter(formatter)
    ax1.set_xlabel('Annual SST Trend ($^\circ$C/decade)', fontsize=12)
    ax1.set_ylabel('# of Reefs', fontsize=12)

    # Second subplot
    x_bins_2 = np.arange(-0.04, plot_data_2.max(), 0.01)
    ax2.hist(plot_data_2, bins=x_bins_2, rwidth=0.75, color='grey', lw=0)
    ax2.set_xticks(np.arange(-0.04, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='x', which='both', top='off', direction='out')
    ax2.tick_params(axis='y', which='both', right='off', direction='out')
    ax2.annotate('(b)', xy=(-0.03, 5.8), fontsize=14, fontweight='bold')
    ax2.annotate('{0:.2f}$\pm${1:.2f} $^\circ$C/decade'.format(
        individual_merge_df['ind_hot_season_decadal_change'].mean(),
        individual_merge_df['ind_hot_season_decadal_change'].std(ddof=1)), xy=(0.07, 7), fontsize=12)
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax2.xaxis.set_major_locator(major_locator)
    ax2.xaxis.set_major_formatter(major_formatter)
    ax2.xaxis.set_minor_locator(minor_locator)
    ax2.set_xlabel('Warm-season SST Trend ($^\circ$C/decade)', fontsize=12)
    ax2.set_ylabel('# of Reefs', fontsize=12)

    # Third subplot
    x_bins_3 = np.arange(0.00, plot_data_3.max(), 0.01)
    ax3.hist(plot_data_3, bins=x_bins_3, rwidth=0.75, color='grey', lw=0)
    ax3.set_xticks(np.arange(0.00, 0.16, 0.02))
    # ax[0, 0].set_xlim(0.00, 0.16)
    # ax[0, 0].set_ylim(0, 10)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(axis='x', which='both', top='off', direction='out')
    ax3.tick_params(axis='y', which='both', right='off', direction='out')
    ax3.annotate('(c)', xy=(0.01, 4), fontsize=14, fontweight='bold')
    ax3.annotate('{0:.2f}$\pm${1:.2f} $^\circ$C/decade'.format(
        individual_merge_df['ind_cool_season_decadal_change'].mean(),
        individual_merge_df['ind_cool_season_decadal_change'].std(ddof=1)), xy=(0.085, 5), fontsize=12)
    major_locator = MultipleLocator(0.02)
    major_formatter = FormatStrFormatter('%.2f')
    minor_locator = MultipleLocator(0.01)
    ax3.xaxis.set_major_locator(major_locator)
    ax3.xaxis.set_major_formatter(major_formatter)
    ax3.xaxis.set_minor_locator(minor_locator)
    ax3.set_xlabel('Cool-season SST Trend ($^\circ$C/decade)', fontsize=12)
    ax3.set_ylabel('# of Reefs', fontsize=12)

    # Fourth subplot
    ax4.hist(number_of_bleaching_events, bins=np.arange(0, 6, 1), rwidth=0.60, color='grey', lw=0)
    ax4.set_xticks(np.arange(0.5, 8.5, 1))
    ax4.set_xticklabels(np.arange(0, 8, 1))
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.tick_params(axis='x', which='both', top='off', direction='out')
    ax4.tick_params(axis='y', which='both', right='off', direction='out')
    ax4.annotate('(d)', xy=(0.37, 9.5), fontsize=14, fontweight='bold')
    ax4.annotate('{0:.2f}$\pm${1:.2f} events'.format(
        np.mean(np.array(number_of_bleaching_events)),
        np.std(np.array(number_of_bleaching_events), ddof=1)), xy=(4.5, 12), fontsize=12)
    ax4.set_xlabel('Number of bleaching level \nthermal stress events', fontsize=12)
    ax4.set_ylabel('# of Reefs', fontsize=12)

    # Fifth subplot
    onset_months = [5, 5, 4, 5, 5, 5, 7, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 6, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 4]
    onset_month_dict = defaultdict(int)
    total_months = 0
    for month in onset_months:
        onset_month_dict[str(month)] += 1
        total_months += 1
    for month_key in onset_month_dict.keys():
        print('Number of events in {0} is {2}, {1:.2f}%'.format(month_key, onset_month_dict[month_key]/total_months*100, onset_month_dict[month_key]))

    x_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    x_ticks = np.arange(1.5, 13.5, 1)
    ax5.hist(onset_months, bins=np.arange(1, 13, 1), rwidth=0.6, color='grey', lw=0)
    ax5.set_xlim(1, 14)
    ax5.set_xticks(x_ticks)
    ax5.set_xticklabels(x_labels)
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.tick_params(axis='x', which='both', top='off', direction='out')
    ax5.tick_params(axis='y', which='both', right='off', direction='out')
    ax5.annotate('(e)', xy=(1.5, 39), fontsize=14, fontweight='bold')
    ax5.set_xlabel('Thermal stress onset months', fontsize=12)
    ax5.set_ylabel('No. of events', fontsize=12)

    plt.tight_layout()
    plt.savefig('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/PHOTOS/thermal_stress_trends_over_time_2.png', dpi=300)

    individual_merge_df = pd.merge(individual_merge_df, bleaching_stress_events_df, on='reef', how='inner')
    # Ranking coral reefs by different parameters
    # Annual average SST trend
    individual_merge_df['rank_annual_sst'] = individual_merge_df['ind_decadal_change'].rank(ascending=True, method='dense')
    individual_merge_df['rank_warm_season_sst'] = individual_merge_df['ind_hot_season_decadal_change'].rank(ascending=True, method='dense')
    individual_merge_df['rank_cool_season_sst'] = individual_merge_df['ind_cool_season_decadal_change'].rank(ascending=True, method='dense')
    individual_merge_df['rank_no_of_events'] = individual_merge_df['event_number'].rank(ascending=True, method='dense')
    individual_merge_df['rank_mean_dhw'] = individual_merge_df['mean_dhw_history'].rank(ascending=True, method='dense')



    print(individual_merge_df[['reef', 'rank_annual_sst', 'rank_no_of_events', 'rank_mean_dhw']])

    individual_merge_df['rank_total'] = individual_merge_df.rank_annual_sst + individual_merge_df.rank_warm_season_sst + individual_merge_df.rank_cool_season_sst + individual_merge_df.rank_no_of_events + individual_merge_df.rank_mean_dhw

    individual_merge_df['rank_by_total'] = individual_merge_df['rank_total'].rank(ascending=True, method='dense')

    print(individual_merge_df[['reef', 'rank_by_total']])

    individual_merge_df.sort('rank_by_total').to_excel('D:/GRADUATE_RESEARCH/SEA SURFACE TEMPERATURE/WRITE_FILES/SST_TRENDS_VER_1/coral_reefs_ranked_by_parameters.xlsx')

    print('Columns: {0}'.format(individual_merge_df.columns))

    print('No. of coral reefs which had less than 1 event per decade: {0}'.format(individual_merge_df[individual_merge_df['events_decade'] < 1.0].shape[0]))