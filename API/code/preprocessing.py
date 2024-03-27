from helper_functions import Helper
from agents import Preparation_Agent

def preprocess_data():
    DATA_PATH = '../data/'
    EXPORT_PATH = '../exportNew/'

    helper = Helper()
    helper.save_weather(house_id=3)

    # preparation of data
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

    shiftable_devices = ['Tumble Dryer', 'Washing Machine',
                         'Dishwasher']

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
    active_appliances = ['Toaster', 'Tumble Dryer', 'Dishwasher', 'Washing Machine', 'Television', 'Microwave',
                         'Kettle']

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

    # load agent
    device_params = {
        'threshold': threshold
    }

    # usage agent
    device = {
        'threshold': threshold}

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
        'shiftable_devices': shiftable_devices,
        'device': device
    }

    household_id = 3
    household = helper.load_household(DATA_PATH, household_id, weather_sel=True)

    # calling the preparation pipeline
    prep = Preparation_Agent(household)
    activity_df = prep.pipeline_activity(household, activity_pipe_params)
    load_df, _, _ = prep.pipeline_load(household, load_pipe_params)
    usage_df = prep.pipeline_usage(household, usage_pipe_params)

    # Save preprocessed data
    activity_df.to_pickle(f'{EXPORT_PATH}/activity_df.pkl')
    load_df.to_pickle(f'{EXPORT_PATH}/load_df.pkl')
    usage_df.to_pickle(f'{EXPORT_PATH}/usage_df.pkl')

if __name__ == '__main__':
    preprocess_data()
