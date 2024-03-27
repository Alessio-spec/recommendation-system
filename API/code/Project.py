
import sys
def test_function():
    print(f"Current Python version: {sys.version}")
    return "Project.py was successfully called!"

def main(predictiondate):

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import json
    plt.rcParams["figure.figsize"] = (16,5)
    import shap
    shap.initjs()

    from IPython.display import Latex
    from IPython.display import Image

    from helper_functions import Helper
    from agents import Evaluation_Agent
    from agents import Preparation_Agent
    from agents import Activity_Agent, Usage_Agent, Load_Agent, Price_Agent, X_Recommendation_Agent, Explainability_Agent
    from copy import deepcopy

    import warnings
    warnings.filterwarnings("ignore")
    import os
    from datetime import datetime


    # In[2]:


    helper = Helper()
    helper.save_weather(house_id=3)


    # In[3]:


    DATA_PATH = '../data/'
    EXPORT_PATH = '../export/'


    # In[4]:


    # # preparation of data
    truncation_params = {
        'features': 'all',
        'factor': 1.5,
        'verbose': 1
    }

    scale_params = {
        'features': 'all',
        'kind': 'MinMax',
        'verbose': 1
    }

    aggregate_params = {
        'resample_param': '60T'
    }

    shiftable_devices = ['Tumble Dryer', 'Washing Machine', 'Dishwasher'] # computer und tv sind m. E. non-shiftable, VR

    device_params = {
        'threshold': 0.15
    }

    load_pipe_params = {
        'truncate': truncation_params,
        'scale': scale_params,
        'aggregate': aggregate_params,
        'shiftable_devices': shiftable_devices,
        'device': device_params
    }

    threshold = 0.01
    active_appliances = ['Toaster', 'Tumble Dryer', 'Dishwasher', 'Washing Machine','Television', 'Microwave', 'Kettle']

    activity_params = {
        'active_appliances': active_appliances,
        'threshold': threshold
    }

    time_params = {
        'features': ['hour', 'day_name']
    }

    activity_lag_params = {
        'features': ['activity'],
        'lags': [24, 48, 72]
    }

    activity_pipe_params = {
        'truncate': truncation_params,
        'scale': scale_params,
        'aggregate': aggregate_params,
        'activity': activity_params,
        'time': time_params,
        'activity_lag': activity_lag_params
    }

    #load agent
    device_params = {
        'threshold': threshold
    }

    #usage agent
    device = {
        'threshold' : threshold}

    aggregate_params24_H = {
        'resample_param': '24H'
    }

    usage_pipe_params = {
        'truncate': truncation_params,
        'scale': scale_params,
        'activity': activity_params,
        'aggregate_hour': aggregate_params,
        'aggregate_day': aggregate_params24_H,
        'time': time_params,
        'activity_lag': activity_lag_params,
        'shiftable_devices' : shiftable_devices,
        'device': device
    }

    household_id = 3
    household = helper.load_household(DATA_PATH, household_id, weather_sel=True)

    # calling the preparation pipeline
    prep = Preparation_Agent(household)
    activity_df = prep.pipeline_activity(household, activity_pipe_params)
    load_df, _, _ = prep.pipeline_load(household, load_pipe_params)
    usage_df = prep.pipeline_usage(household, usage_pipe_params)

    # load price data
    price_df = helper.create_day_ahead_prices_df(DATA_PATH, 'Day-ahead Prices_201501010000-201601010000.csv')

    activity_df.to_pickle('../data/processed_pickle/activity_df.pkl')
    load_df,_, _ .to_pickle('../data/processed_pickle/load_df.pkl')
    usage_df.to_pickle('../data/processed_pickle/usage_df.pkl')
    price_df.to_pickle('../data/processed_pickle/price_df.pkl')


    import interpret
    from interpret.glassbox._ebm._ebm import ExplainableBoostingClassifier
    from interpret import show
    from agents import Activity_Agent
    activity_df = pd.read_pickle('../data/processed_pickle/activity_df.pkl')
    activity = Activity_Agent(activity_df)
    X_train, y_train, X_test, y_test = activity.train_test_split(activity_df, '2014-08-21')


    # In[6]:


    model = activity.fit(X_train, y_train, 'ebm')
    show(model.explain_global())



    ebm_local = model.explain_local(X_test, y_test)
    show(ebm_local)



    import lime
    from lime import lime_tabular


    # In[10]:


    model = activity.fit(X_train, y_train, 'logit')
    best_hour = 12


    # In[11]:


    pred_model = activity.fit_Logit(X_train,y_train)
    explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(X_train),
                                                  mode = "classification",
                                                  feature_names = X_train.columns,
                                                  categorical_features = [0])

    exp = explainer.explain_instance(data_row = X_test.iloc[best_hour],
                                    predict_fn = model.predict_proba)

    exp.show_in_notebook(show_table = True)



    from IPython.display import display

    import shap
    shap.initjs()
    usage_df = pd.read_pickle('../data/processed_pickle/usage_df.pkl')


    # In[13]:


    from agents import Usage_Agent
    usage = Usage_Agent(usage_df, "Dishwasher")
    X_train, y_train, X_test, y_test = usage.train_test_split(usage_df, "2014-11-01", train_start='2013-11-01')
    model = usage.fit(X_train, y_train, 'logit')


    # In[14]:


    #X_train_summary = shap.kmeans(X_train, 10)
    X_train_summary = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
    base_value = explainer.expected_value[1]
    shap_values = explainer.shap_values(X_test, check_additivity=False)


    # In[15]:


    display(shap.force_plot(explainer.expected_value[1], shap_values[1], X_test))




    # Load pickle data
    activity_df = pd.read_pickle('../data/processed_pickle/activity_df.pkl')
    load_df = pd.read_pickle('../data/processed_pickle/load_df.pkl')
    usage_df = pd.read_pickle('../data/processed_pickle/usage_df.pkl')
    price_df = pd.read_pickle('../data/processed_pickle/price_df.pkl')


    # The recommendations are not returned if the activity prediction and the
    # usability prediction fall below the thresholds.

    # In[17]:


    recommend = X_Recommendation_Agent(activity_df, usage_df, load_df, price_df, shiftable_devices, model_type='random forest')
    price = recommend.electricity_prices_from_start_time(predictiondate)
    table= recommend.pipeline(date = predictiondate, activity_prob_threshold = 0.4,  usage_prob_threshold = 0.7, evaluation=False, weather_sel=False)
    table


    # After the creation of the table, we derive the appropriate explanation within the
    # function "visualize_recommendation" from the recommendation agent.

    # In[18]:


    result = recommend.visualize_recommendation(table, price, diagnostics=False)

    # In[19]:


    # output, scaled, df = Preparation_Agent(household).pipeline_load(household, load_pipe_params)
    #
    #
    # # The following correlation plot displays a positive correlation between the usage of the washing machine and the tumble dryer, which was expected, as if you use the washing machine and own a tumble dryer one would usually use it afterwards. Furthermore there seems to be a strong positive correlation between the relative humidity (rhum) and the dew point (dwpt) and a strong negative correlation between the air temperature (temp) and the dew point (dwpt).
    # # In general, it can be observed that the data are positively correlated with each other, but there are no strong interdependencies.
    # #
    #
    # # In[20]:
    #
    #
    # plt.title('Correlation of weather features and Usage of Devices')
    # weather_df = df[['Tumble Dryer_usage', 'Washing Machine_usage', 'Dishwasher_usage','dwpt', 'rhum', 'temp', 'wdir', 'wspd']]
    # sns.heatmap(weather_df.corr(), annot=True)
    #
    #
    # # Now we want to examine if a device was used multiple times each day. As for the tumble dryer there exist dates where it was used 8 times, the washing machine was used 9 times a day and the dishwasher was used 7 times a day as a maximum. This is rather rare but interesting for our recommender system when it comes to saving energy and money and must be discussed in terms of further research.
    #
    # # In[21]:
    #
    #
    # # Tumble Dryer in use
    # reIndexed = df.reset_index(drop=True)
    #  # Convert "Time" column to datetime type
    # reIndexed['Time'] = pd.to_datetime(reIndexed['Time'])
    #
    # # Extract date and time components
    # reIndexed['Date'] = reIndexed['Time'].dt.date
    # reIndexed['clock'] = reIndexed['Time'].dt.time
    #
    # #reIndexed['Date'], reIndexed['clock'] = reIndexed['Time'].apply(lambda x: x.date()), reIndexed['Time'].apply(lambda x: x.time())
    # wm = reIndexed['Washing Machine_usage']==1
    # td = reIndexed['Tumble Dryer_usage']==1
    # dw = reIndexed['Dishwasher_usage'] ==1
    # td_inuse = reIndexed[td]
    # td_inuse.groupby('Date')[['Tumble Dryer_usage']].count().describe()
    #
    #
    # # In[22]:
    #
    #
    # # Washing Machine in use
    # wm_inuse = reIndexed[wm]
    # wm_inuse.groupby('Date')[['Washing Machine_usage']].count().describe()
    #
    #
    # # In[23]:
    #
    #
    # # Dishwasher in use
    # dw_inuse = reIndexed[dw]
    # dw_inuse.groupby('Date')[['Dishwasher_usage']].count().describe()
    #
    #
    # # On average each device is used two times a day. As expected the following distribution plot shows a similiar situation for the usage of the washing machine and the tumble dryer.
    #
    # # In[24]:
    #
    #
    # plt.title('Distribution of Tumble Dryer and Washing Machine Usage')
    # sns.distplot(td_inuse['Date'].value_counts().values)
    # sns.distplot(wm_inuse['Date'].value_counts().values)
    # plt.legend(labels=['Tumble Dryer', 'Washing Mashine'])
    #
    #
    # # In[25]:
    #
    #
    # plt.title('Distribution of Dishwasher Usage')
    # sns.distplot(dw_inuse['Date'].value_counts().values, color='green')
    #
    #
    # # In[26]:
    #
    #
    # # Look at dates of interest for Tumble dryer and Washing Machine
    # df_20_04_2014 = df[['dwpt','rhum','temp','wdir','wspd','Tumble Dryer_usage', 'Washing Machine_usage']].filter(like='2014-04-20', axis=0)
    # df_10_08_2014 = df[['dwpt','rhum','temp','wdir','wspd','Tumble Dryer_usage', 'Washing Machine_usage']].filter(like='2014-08-10', axis=0)
    # df_03_09_2015 = df[['dwpt','rhum','temp','wdir','wspd','Tumble Dryer_usage', 'Washing Machine_usage']].filter(like='2015-09-03', axis=0)
    # df_26_06_2014 = df[['dwpt','rhum','temp','wdir','wspd','Tumble Dryer_usage', 'Washing Machine_usage']].filter(like='2014-06-26', axis=0)
    # df_28_09_2013 = df[['dwpt','rhum','temp','wdir','wspd', 'Tumble Dryer_usage', 'Washing Machine_usage']].filter(like='2013-09-28', axis=0)
    #
    #
    # # In[27]:
    #
    #
    # # Look at dates of interest for Dishwasher
    # df_27_09_2013 = df[['dwpt','rhum','temp','wdir','wspd','Dishwasher_usage']].filter(like='2013-09-27', axis=0)
    # df_25_11_2014 = df[['dwpt','rhum','temp','wdir','wspd' ,'Dishwasher_usage']].filter(like='2014-11-25', axis=0)
    # df_19_10_2014 = df[['dwpt','rhum','temp','wdir','wspd', 'Dishwasher_usage']].filter(like='2014-10-19', axis=0)
    # df_01_07_2014 = df[['dwpt','rhum','temp','wdir','wspd', 'Dishwasher_usage']].filter(like='2014-07-01', axis=0)
    #
    #
    #
    # # Draw Plot
    # def plot_df(df, x, y, title="", xlabel='TIme', ylabel='Value', dpi=100, axvspan=True):
    #     plt.figure(figsize=(16,5), dpi=dpi)
    #     plt.plot(x, y, marker='o')
    #     plt.legend(df_27_09_2013.columns.tolist())
    #
    #     plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    #     plt.show()
    #
    # plot_df(df_27_09_2013, x=df_27_09_2013.index, y=df_27_09_2013.values, title='Tumble Dryer and Waching Machine Usage on the 27.09.2013')
    # # In[35]:
    #
    #
    # xai= True; weather_sel = True; household_id = 3; model_type = 'logit'
    # RESULT_PATH = str(EXPORT_PATH)+str(household_id)+ "_" + str(model_type) + "_" + str(model_type) + "_" + str(weather_sel) + "_"
    #
    # with open(RESULT_PATH + 'predictions.pkl','rb') as path_name:
    #     predictions_activity = pd.read_pickle(path_name) #pickle.load
    #
    #
    # # In[36]:
    #
    #
    # config = json.load(open(EXPORT_PATH + str(household_id) + '_' + str(model_type) + '_' + str(model_type) +'_' + str(weather_sel) +'_config.json', 'r'))
    # files = ['df.pkl', 'output.pkl']
    # files = [f"{EXPORT_PATH}{household_id}_{model_type}_{model_type}_{weather_sel}_{file}" for file in files]
    #
    # # initializing the agent
    # evaluation = Evaluation_Agent(DATA_PATH,model_type=model_type ,config=config, load_data=True, load_files=files, weather_sel=weather_sel,xai= xai)
    # evaluation.init_agents()
    #
    #
    # # In[37]:
    #
    #
    # xai_scores_activity = evaluation.predictions_to_xai_metrics(predictions_activity)
    # xai_scores_activity


    # result = recommend.visualize_recommendation(table, price, diagnostics=True)

    return result
if __name__ == '__main__':
    prediction_date = '2014-08-21'  # Example prediction date
    result = main(prediction_date)
    print(result)