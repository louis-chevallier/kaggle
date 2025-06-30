# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.019116, "end_time": "2025-06-20T13:51:28.914814", "exception": false, "start_time": "2025-06-20T13:51:28.895698", "status": "completed"}
# # 📋Table of Contents
# * [Import and first glance](#load)
# * [Targets](#targets)
# * [Correlations](#correlations)
# * [Targets vs features](#target_features)
# * [Baseline model](#base)

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 6.325571, "end_time": "2025-06-20T13:51:35.253303", "exception": false, "start_time": "2025-06-20T13:51:28.927732", "status": "completed"}
# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# stats
from scipy import stats

# missing values
import missingno as msno

# ML
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

# %% papermill={"duration": 0.018065, "end_time": "2025-06-20T13:51:35.282794", "exception": false, "start_time": "2025-06-20T13:51:35.264729", "status": "completed"}
# configs

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'

# random
my_random_seed = 12345

# warnings
import warnings
warnings.filterwarnings('ignore')

# %% papermill={"duration": 0.148133, "end_time": "2025-06-20T13:51:35.441893", "exception": false, "start_time": "2025-06-20T13:51:35.293760", "status": "completed"}
# show files
# !ls -l '/kaggle/input/neurips-open-polymer-prediction-2025'

# %% [markdown] papermill={"duration": 0.010364, "end_time": "2025-06-20T13:51:35.463218", "exception": false, "start_time": "2025-06-20T13:51:35.452854", "status": "completed"}
# <a id='load'></a>
# # Import and first glance

# %% papermill={"duration": 0.071142, "end_time": "2025-06-20T13:51:35.545034", "exception": false, "start_time": "2025-06-20T13:51:35.473892", "status": "completed"}
# load data
t1 = time.time()
df_train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df_test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
df_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1,4))

# %% papermill={"duration": 0.045553, "end_time": "2025-06-20T13:51:35.601412", "exception": false, "start_time": "2025-06-20T13:51:35.555859", "status": "completed"}
# preview
df_train.head()

# %% papermill={"duration": 0.04392, "end_time": "2025-06-20T13:51:35.656588", "exception": false, "start_time": "2025-06-20T13:51:35.612668", "status": "completed"}
# show structure of data - train
df_train.info(show_counts=True, verbose=True)

# %% [markdown] papermill={"duration": 0.010941, "end_time": "2025-06-20T13:51:35.678685", "exception": false, "start_time": "2025-06-20T13:51:35.667744", "status": "completed"}
# #### 💡 Targets have different levels of missingness. Let's explore the structure.

# %% papermill={"duration": 0.7337, "end_time": "2025-06-20T13:51:36.423172", "exception": false, "start_time": "2025-06-20T13:51:35.689472", "status": "completed"}
# visualize structure of missing values
msno.matrix(df_train)

# %% papermill={"duration": 0.025456, "end_time": "2025-06-20T13:51:36.462212", "exception": false, "start_time": "2025-06-20T13:51:36.436756", "status": "completed"}
# (preliminary) test set, just three observations
df_test

# %% papermill={"duration": 0.050451, "end_time": "2025-06-20T13:51:36.525020", "exception": false, "start_time": "2025-06-20T13:51:36.474569", "status": "completed"}
# very simple features
df_train['length'] = df_train.SMILES.apply(len)
df_test['length'] = df_test.SMILES.apply(len)

df_train['count_c'] = df_train.SMILES.apply(lambda x : x.count('c'))
df_test['count_c'] = df_test.SMILES.apply(lambda x : x.count('c'))

df_train['count_C'] = df_train.SMILES.apply(lambda x : x.count('C'))
df_test['count_C'] = df_test.SMILES.apply(lambda x : x.count('C'))

df_train['count_O'] = df_train.SMILES.apply(lambda x : x.count('O'))
df_test['count_O'] = df_test.SMILES.apply(lambda x : x.count('O'))

df_train['count_N'] = df_train.SMILES.apply(lambda x : x.count('N'))
df_test['count_N'] = df_test.SMILES.apply(lambda x : x.count('N'))

# list of features
features = ['length', 'count_c', 'count_C', 'count_O', 'count_N']

# %% papermill={"duration": 1.37886, "end_time": "2025-06-20T13:51:37.916137", "exception": false, "start_time": "2025-06-20T13:51:36.537277", "status": "completed"}
# plot features
for f in features:
    plt.figure(figsize=(8,3))
    df_train[f].plot(kind='hist', bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()

# %% [markdown] papermill={"duration": 0.087541, "end_time": "2025-06-20T13:51:38.019268", "exception": false, "start_time": "2025-06-20T13:51:37.931727", "status": "completed"}
# <a id='targets'></a>
# # Targets

# %% papermill={"duration": 0.023771, "end_time": "2025-06-20T13:51:38.059366", "exception": false, "start_time": "2025-06-20T13:51:38.035595", "status": "completed"}
# targets
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# %% papermill={"duration": 0.041584, "end_time": "2025-06-20T13:51:38.118015", "exception": false, "start_time": "2025-06-20T13:51:38.076431", "status": "completed"}
# filtered data frame for each target
df_train_Tg = df_train[~df_train['Tg'].isna()]
df_train_FFV = df_train[~df_train['FFV'].isna()]
df_train_Tc = df_train[~df_train['Tc'].isna()]
df_train_Density = df_train[~df_train['Density'].isna()]
df_train_Rg = df_train[~df_train['Rg'].isna()]

# %% papermill={"duration": 0.267163, "end_time": "2025-06-20T13:51:38.401065", "exception": false, "start_time": "2025-06-20T13:51:38.133902", "status": "completed"}
# target: Glass transition temperature
print(df_train_Tg['Tg'].describe())
plt.figure(figsize=(8,4))
df_train_Tg['Tg'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Tg')
plt.grid()
plt.show()

# %% papermill={"duration": 0.321592, "end_time": "2025-06-20T13:51:38.739296", "exception": false, "start_time": "2025-06-20T13:51:38.417704", "status": "completed"}
# target: Fractional free volume
print(df_train_FFV['FFV'].describe())
plt.figure(figsize=(8,4))
df_train_FFV['FFV'].plot(kind='hist', bins=100, color=default_color_3)
plt.title('FFV')
plt.grid()
plt.show()

# %% papermill={"duration": 0.24973, "end_time": "2025-06-20T13:51:39.006197", "exception": false, "start_time": "2025-06-20T13:51:38.756467", "status": "completed"}
# target: Thermal conductivity
print(df_train_Tc['Tc'].describe())
plt.figure(figsize=(8,4))
df_train_Tc['Tc'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Tc')
plt.grid()
plt.show()

# %% papermill={"duration": 0.26068, "end_time": "2025-06-20T13:51:39.285069", "exception": false, "start_time": "2025-06-20T13:51:39.024389", "status": "completed"}
# target: Polymer density
print(df_train_Density['Density'].describe())
plt.figure(figsize=(8,4))
df_train_Density['Density'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Density')
plt.grid()
plt.show()

# %% papermill={"duration": 0.248309, "end_time": "2025-06-20T13:51:39.551021", "exception": false, "start_time": "2025-06-20T13:51:39.302712", "status": "completed"}
# target: Radius of gyration
print(df_train_Rg['Rg'].describe())
plt.figure(figsize=(8,4))
df_train_Rg['Rg'].plot(kind='hist', bins=50, color=default_color_3)
plt.title('Rg')
plt.grid()
plt.show()

# %% [markdown] papermill={"duration": 0.017452, "end_time": "2025-06-20T13:51:39.586197", "exception": false, "start_time": "2025-06-20T13:51:39.568745", "status": "completed"}
# <a id='correlations'></a>
# # Correlations

# %% papermill={"duration": 6.059128, "end_time": "2025-06-20T13:51:45.665927", "exception": false, "start_time": "2025-06-20T13:51:39.606799", "status": "completed"}
# scatterplot
sns.pairplot(data=df_train[targets],
             diag_kws = { 'color' : default_color_3},
             plot_kws = { 'color' : default_color_3,
                          'alpha' : 0.5,
                          's' : 15})
plt.show()

# %% papermill={"duration": 0.584127, "end_time": "2025-06-20T13:51:46.273415", "exception": false, "start_time": "2025-06-20T13:51:45.689288", "status": "completed"}
# correlation of missingness
msno.heatmap(df_train, cmap='RdBu')

# %% [markdown] papermill={"duration": 0.025941, "end_time": "2025-06-20T13:51:46.324903", "exception": false, "start_time": "2025-06-20T13:51:46.298962", "status": "completed"}
# <a id='target_features'></a>
# # Target vs features

# %% papermill={"duration": 4.704372, "end_time": "2025-06-20T13:51:51.057506", "exception": false, "start_time": "2025-06-20T13:51:46.353134", "status": "completed"}
# scatter plots
for t in targets:
    for f in features:
        c = np.round(df_train[f].corr(df_train[t]), 4)
        plt.scatter(df_train[f], df_train[t], color=default_color_3,
                    alpha=0.5)
        plt.title(t + ' vs ' + f + ' | corr = ' + str(c))
        plt.grid()
        plt.show()

# %% [markdown] papermill={"duration": 0.05501, "end_time": "2025-06-20T13:51:51.171224", "exception": false, "start_time": "2025-06-20T13:51:51.116214", "status": "completed"}
# <a id='base'></a>
# # Baseline model

# %% _kg_hide-output=true papermill={"duration": 9.79443, "end_time": "2025-06-20T13:52:01.021303", "exception": false, "start_time": "2025-06-20T13:51:51.226873", "status": "completed"}
# start H2O
h2o.init(max_mem_size='20G', nthreads=4)

# %% [markdown] papermill={"duration": 0.086628, "end_time": "2025-06-20T13:52:01.232168", "exception": false, "start_time": "2025-06-20T13:52:01.145540", "status": "completed"}
# ### Model for Tg

# %% papermill={"duration": 1.728233, "end_time": "2025-06-20T13:52:03.034356", "exception": false, "start_time": "2025-06-20T13:52:01.306123", "status": "completed"}
train_hex = h2o.H2OFrame(df_train_Tg)
test_hex = h2o.H2OFrame(df_test)

# %% papermill={"duration": 0.071188, "end_time": "2025-06-20T13:52:03.168470", "exception": false, "start_time": "2025-06-20T13:52:03.097282", "status": "completed"}
# setup of model
fit_Tg = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 50,
                                      learn_rate = 0.1,
                                      max_depth = 6,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)

# %% papermill={"duration": 3.077453, "end_time": "2025-06-20T13:52:06.304029", "exception": false, "start_time": "2025-06-20T13:52:03.226576", "status": "completed"}
# run training
fit_Tg.train(features, 'Tg', training_frame = train_hex);

# %% papermill={"duration": 0.079436, "end_time": "2025-06-20T13:52:06.451663", "exception": false, "start_time": "2025-06-20T13:52:06.372227", "status": "completed"}
# short summary
fit_Tg.summary()

# %% papermill={"duration": 0.083846, "end_time": "2025-06-20T13:52:06.599875", "exception": false, "start_time": "2025-06-20T13:52:06.516029", "status": "completed"}
# show cross validation results
fit_Tg.cross_validation_metrics_summary().as_data_frame()

# %% papermill={"duration": 0.364727, "end_time": "2025-06-20T13:52:07.054741", "exception": false, "start_time": "2025-06-20T13:52:06.690014", "status": "completed"}
# predict on training data
pred_train = fit_Tg.predict(train_hex)
pred_train = pred_train.as_data_frame().predict

# %% papermill={"duration": 0.236831, "end_time": "2025-06-20T13:52:07.366097", "exception": false, "start_time": "2025-06-20T13:52:07.129266", "status": "completed"}
# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Tg.Tg, pred_train, color=default_color_1, alpha=0.5)
plt.grid()
plt.show()

# %% papermill={"duration": 0.308454, "end_time": "2025-06-20T13:52:07.735175", "exception": false, "start_time": "2025-06-20T13:52:07.426721", "status": "completed"}
# predict on test data
pred_test = fit_Tg.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Tg = pred_test

# %% [markdown] papermill={"duration": 0.062022, "end_time": "2025-06-20T13:52:07.859318", "exception": false, "start_time": "2025-06-20T13:52:07.797296", "status": "completed"}
# ### Model for FFV

# %% papermill={"duration": 0.88979, "end_time": "2025-06-20T13:52:08.813661", "exception": false, "start_time": "2025-06-20T13:52:07.923871", "status": "completed"}
train_hex = h2o.H2OFrame(df_train_FFV)
test_hex = h2o.H2OFrame(df_test)

# %% papermill={"duration": 0.112498, "end_time": "2025-06-20T13:52:09.026368", "exception": false, "start_time": "2025-06-20T13:52:08.913870", "status": "completed"}
# setup of model
fit_FFV = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                       nfolds=5,
                                       ntrees = 200,
                                       learn_rate = 0.1,
                                       max_depth = 6,
                                       col_sample_rate = 0.7,                                    
                                       stopping_rounds = 10,
                                       stopping_metric = 'MAE',
                                       score_each_iteration = True,                                          
                                       seed=my_random_seed)

# %% papermill={"duration": 11.24344, "end_time": "2025-06-20T13:52:20.362378", "exception": false, "start_time": "2025-06-20T13:52:09.118938", "status": "completed"}
# run training
fit_FFV.train(features, 'FFV', training_frame = train_hex);

# %% papermill={"duration": 0.108503, "end_time": "2025-06-20T13:52:20.581496", "exception": false, "start_time": "2025-06-20T13:52:20.472993", "status": "completed"}
# short summary
fit_FFV.summary()

# %% papermill={"duration": 0.138793, "end_time": "2025-06-20T13:52:20.819327", "exception": false, "start_time": "2025-06-20T13:52:20.680534", "status": "completed"}
# show cross validation results
fit_FFV.cross_validation_metrics_summary().as_data_frame()

# %% papermill={"duration": 0.558003, "end_time": "2025-06-20T13:52:21.457132", "exception": false, "start_time": "2025-06-20T13:52:20.899129", "status": "completed"}
# predict on training data
pred_train = fit_FFV.predict(train_hex)
pred_train = pred_train.as_data_frame().predict

# %% papermill={"duration": 0.27704, "end_time": "2025-06-20T13:52:21.823008", "exception": false, "start_time": "2025-06-20T13:52:21.545968", "status": "completed"}
# scatter plot of predictions vs actual on training data
plt.scatter(df_train_FFV.FFV, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()

# %% papermill={"duration": 0.11794, "end_time": "2025-06-20T13:52:22.012059", "exception": false, "start_time": "2025-06-20T13:52:21.894119", "status": "completed"}
# predict on test data
pred_test = fit_FFV.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.FFV = pred_test

# %% [markdown] papermill={"duration": 0.069905, "end_time": "2025-06-20T13:52:22.169511", "exception": false, "start_time": "2025-06-20T13:52:22.099606", "status": "completed"}
# ### Model for Tc

# %% papermill={"duration": 0.39721, "end_time": "2025-06-20T13:52:22.640144", "exception": false, "start_time": "2025-06-20T13:52:22.242934", "status": "completed"}
train_hex = h2o.H2OFrame(df_train_Tc)
test_hex = h2o.H2OFrame(df_test)

# %% papermill={"duration": 0.114982, "end_time": "2025-06-20T13:52:22.857643", "exception": false, "start_time": "2025-06-20T13:52:22.742661", "status": "completed"}
# setup of model
fit_Tc = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 100,
                                      learn_rate = 0.1,
                                      max_depth = 4,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)

# %% papermill={"duration": 2.07656, "end_time": "2025-06-20T13:52:25.041810", "exception": false, "start_time": "2025-06-20T13:52:22.965250", "status": "completed"}
# run training
fit_Tc.train(features, 'Tc', training_frame = train_hex);

# %% papermill={"duration": 0.12351, "end_time": "2025-06-20T13:52:25.282817", "exception": false, "start_time": "2025-06-20T13:52:25.159307", "status": "completed"}
# short summary
fit_Tc.summary()

# %% papermill={"duration": 0.089598, "end_time": "2025-06-20T13:52:25.478462", "exception": false, "start_time": "2025-06-20T13:52:25.388864", "status": "completed"}
# show cross validation results
fit_Tc.cross_validation_metrics_summary().as_data_frame()

# %% papermill={"duration": 0.314117, "end_time": "2025-06-20T13:52:25.864980", "exception": false, "start_time": "2025-06-20T13:52:25.550863", "status": "completed"}
# predict on training data
pred_train = fit_Tc.predict(train_hex)
pred_train = pred_train.as_data_frame().predict

# %% papermill={"duration": 0.262365, "end_time": "2025-06-20T13:52:26.208343", "exception": false, "start_time": "2025-06-20T13:52:25.945978", "status": "completed"}
# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Tc.Tc, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()

# %% papermill={"duration": 0.115979, "end_time": "2025-06-20T13:52:26.399124", "exception": false, "start_time": "2025-06-20T13:52:26.283145", "status": "completed"}
# predict on test data
pred_test = fit_Tc.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Tc = pred_test

# %% [markdown] papermill={"duration": 0.071301, "end_time": "2025-06-20T13:52:26.544501", "exception": false, "start_time": "2025-06-20T13:52:26.473200", "status": "completed"}
# ### Model for Density

# %% papermill={"duration": 0.376773, "end_time": "2025-06-20T13:52:26.994485", "exception": false, "start_time": "2025-06-20T13:52:26.617712", "status": "completed"}
train_hex = h2o.H2OFrame(df_train_Density)
test_hex = h2o.H2OFrame(df_test)

# %% papermill={"duration": 0.084912, "end_time": "2025-06-20T13:52:27.153953", "exception": false, "start_time": "2025-06-20T13:52:27.069041", "status": "completed"}
# setup of model
fit_Density = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                           nfolds=5,
                                           ntrees = 100,
                                           learn_rate = 0.1,
                                           max_depth = 6,
                                           col_sample_rate = 0.7,                                    
                                           stopping_rounds = 10,
                                           stopping_metric = 'MAE',
                                           score_each_iteration = True,                                          
                                           seed=my_random_seed)

# %% papermill={"duration": 2.006612, "end_time": "2025-06-20T13:52:29.236575", "exception": false, "start_time": "2025-06-20T13:52:27.229963", "status": "completed"}
# run training
fit_Density.train(features, 'Density', training_frame = train_hex);

# %% papermill={"duration": 0.097687, "end_time": "2025-06-20T13:52:29.422112", "exception": false, "start_time": "2025-06-20T13:52:29.324425", "status": "completed"}
# short summary
fit_Density.summary()

# %% papermill={"duration": 0.089422, "end_time": "2025-06-20T13:52:29.589957", "exception": false, "start_time": "2025-06-20T13:52:29.500535", "status": "completed"}
# show cross validation results
fit_Density.cross_validation_metrics_summary().as_data_frame()

# %% papermill={"duration": 0.320882, "end_time": "2025-06-20T13:52:29.989845", "exception": false, "start_time": "2025-06-20T13:52:29.668963", "status": "completed"}
# predict on training data
pred_train = fit_Density.predict(train_hex)
pred_train = pred_train.as_data_frame().predict

# %% papermill={"duration": 0.267409, "end_time": "2025-06-20T13:52:30.375051", "exception": false, "start_time": "2025-06-20T13:52:30.107642", "status": "completed"}
# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Density.Density, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()

# %% papermill={"duration": 0.112292, "end_time": "2025-06-20T13:52:30.571050", "exception": false, "start_time": "2025-06-20T13:52:30.458758", "status": "completed"}
# predict on test data
pred_test = fit_Density.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Density = pred_test

# %% [markdown] papermill={"duration": 0.082043, "end_time": "2025-06-20T13:52:30.729063", "exception": false, "start_time": "2025-06-20T13:52:30.647020", "status": "completed"}
# ### Model for Rg

# %% papermill={"duration": 0.36726, "end_time": "2025-06-20T13:52:31.196703", "exception": false, "start_time": "2025-06-20T13:52:30.829443", "status": "completed"}
train_hex = h2o.H2OFrame(df_train_Rg)
test_hex = h2o.H2OFrame(df_test)

# %% papermill={"duration": 0.096865, "end_time": "2025-06-20T13:52:31.368778", "exception": false, "start_time": "2025-06-20T13:52:31.271913", "status": "completed"}
# setup of model
fit_Rg = H2OGradientBoostingEstimator(distribution = 'gaussian',                                    
                                      nfolds=5,
                                      ntrees = 50,
                                      learn_rate = 0.1,
                                      max_depth = 6,
                                      col_sample_rate = 0.7,                                    
                                      stopping_rounds = 10,
                                      stopping_metric = 'MAE',
                                      score_each_iteration = True,                                          
                                      seed=my_random_seed)

# %% papermill={"duration": 1.176843, "end_time": "2025-06-20T13:52:32.629887", "exception": false, "start_time": "2025-06-20T13:52:31.453044", "status": "completed"}
# run training
fit_Rg.train(features, 'Rg', training_frame = train_hex);

# %% papermill={"duration": 0.087536, "end_time": "2025-06-20T13:52:32.798484", "exception": false, "start_time": "2025-06-20T13:52:32.710948", "status": "completed"}
# short summary
fit_Rg.summary()

# %% papermill={"duration": 0.105875, "end_time": "2025-06-20T13:52:32.986995", "exception": false, "start_time": "2025-06-20T13:52:32.881120", "status": "completed"}
# show cross validation results
fit_Rg.cross_validation_metrics_summary().as_data_frame()

# %% papermill={"duration": 0.353672, "end_time": "2025-06-20T13:52:33.421211", "exception": false, "start_time": "2025-06-20T13:52:33.067539", "status": "completed"}
# predict on training data
pred_train = fit_Rg.predict(train_hex)
pred_train = pred_train.as_data_frame().predict

# %% papermill={"duration": 0.40083, "end_time": "2025-06-20T13:52:33.963636", "exception": false, "start_time": "2025-06-20T13:52:33.562806", "status": "completed"}
# scatter plot of predictions vs actual on training data
plt.scatter(df_train_Rg.Rg, pred_train, color=default_color_1, alpha=0.25)
plt.grid()
plt.show()

# %% papermill={"duration": 0.149802, "end_time": "2025-06-20T13:52:34.214221", "exception": false, "start_time": "2025-06-20T13:52:34.064419", "status": "completed"}
# predict on test data
pred_test = fit_Rg.predict(test_hex)
pred_test = pred_test.as_data_frame().predict

# and add results to submission
df_sub.Rg = pred_test

# %% [markdown] papermill={"duration": 0.084714, "end_time": "2025-06-20T13:52:34.405757", "exception": false, "start_time": "2025-06-20T13:52:34.321043", "status": "completed"}
# ### Show combined results and save submission file

# %% papermill={"duration": 0.104691, "end_time": "2025-06-20T13:52:34.639023", "exception": false, "start_time": "2025-06-20T13:52:34.534332", "status": "completed"}
# show submission data
df_sub.head()

# %% papermill={"duration": 0.093865, "end_time": "2025-06-20T13:52:34.814015", "exception": false, "start_time": "2025-06-20T13:52:34.720150", "status": "completed"}
# save submission file
df_sub.to_csv('submission.csv', index=False)
