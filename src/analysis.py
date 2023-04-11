import os
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

log = logging.getLogger('ICN-Forecasting')

def plot_corralation(df, savepath='Correlation.png'):
    f = plt.figure(figsize=(19, 19))
    plt.matshow(df.corr(numeric_only=True), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(savepath)

def plot_diff(df, key, save_path, days=14):  
    # Orginal Series
    fig, axes = plt.subplots(3,1, figsize=(20,14), sharex=False)
    for site in df['site'].unique():
        site_df = df[df['site'] == site].copy()
        axes[0].plot('timestamp', key, data=site_df, label=site)
        try:
            plot_acf(site_df[key], ax=axes[1], lags=days, label=site)
        except:
            log.error('Error generating ACF plot for {} at site {}'.format(key, site))
        try:
            plot_pacf(site_df[key], ax=axes[2], method='ols', lags=days, label=site)
        except:
            log.error('Error generating PACF plot for {} at site {}'.format(key, site))

        # 1st Differencing
        #site_df['1stdiff'] = site_df[key].diff().dropna()
        #axes[1,0].plot('timestamp', '1stdiff', data=site_df, label=site)
        #plot_acf(site_df['1stdiff'], ax=axes[1,1], lags=days)

        # 2nd Differencing
        #site_df['2nddiff'] = site_df[key].diff().diff().dropna()
        #axes[2,0].plot('timestamp', '2nddiff', data=site_df, label=site)
        #plot_acf(site_df['2nddiff'], ax=axes[2,1], lags=days)

    axes[0].set_title('Original Series')
    axes[0].grid(visible=True, axis='y')
    #axes[1].set_title('1st Differential')
    #axes[2].set_title('2nd Differential')
    axes[0].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    axes[1].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    axes[2].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    plt.savefig(save_path)
    plt.close()

def data_analysis(data, config):
    log.info('Starting data analysis')
    # Select only the sample period data.
    sp_df = data[(data['timestamp'] >= config.sample_period_start) & (data['timestamp'] < config.sample_period_end)]

    # Site specific analysis
    for site in sp_df['site'].unique():
        log.info('Analyzing site {}'.format(site))
        site_df = sp_df[sp_df['site'] == site]

        log.debug('Creating stats csv')
        # Write stats to a csv
        statsdf = site_df.describe()
        statsdf.to_csv(os.path.join(config.save_path, '{}_stats.csv'.format(site)))

        log.debug('Plotting Corralation graph')
        # Plot Corralation graph
        if not os.path.exists(os.path.join(config.save_path, 'plots')):
            os.makedirs(os.path.join(config.save_path, 'plots'))
        plot_corralation(site_df, os.path.join(config.save_path, 'plots', 'Corr_{}.png'.format(site)))

    # Variable specfic analysis
    log.info('Creating ACF, PACF plots for each variable')
    for key in tqdm(config.numeric_cols):
        if key in ['year','month','day','timestamp']:
            continue
        plot_diff(sp_df, key, os.path.join(config.save_path, 'plots', 'Diff_{}.png'.format(key)), days=config.acf_days)
        
    log.info('Finished data analysis')