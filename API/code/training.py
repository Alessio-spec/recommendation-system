import pandas as pd
from agents import Activity_Agent, Usage_Agent
import joblib  # For saving model objects

EXPORT_PATH = '../exportNew/'

def train_models():
    # Load preprocessed data
    activity_df = pd.read_pickle(f'{EXPORT_PATH}/activity_df.pkl')
    usage_df = pd.read_pickle(f'{EXPORT_PATH}/usage_df.pkl')

    # Initialize agents and train models
    activity_agent = Activity_Agent(activity_df)
    X_train_activity, y_train_activity, X_test_activity, y_test_activity = activity_agent.train_test_split(activity_df, '2023-12-01')  # Example date, adjust accordingly
    model_activity = activity_agent.fit(X_train_activity, y_train_activity, 'ebm')  # Replace 'ebm' with your model of choice

    usage_agent = Usage_Agent(usage_df, "Dishwasher")  # Example appliance, adjust accordingly
    X_train_usage, y_train_usage, X_test_usage, y_test_usage = usage_agent.train_test_split(usage_df, "2024-11-01", train_start='2023-11-01')  # Example dates, adjust accordingly
    model_usage = usage_agent.fit(X_train_usage, y_train_usage, 'logit')  # Replace 'logit' with your model of choice

    # Save trained models
    joblib.dump(model_activity, f'{EXPORT_PATH}/model_activity.pkl')
    joblib.dump(model_usage, f'{EXPORT_PATH}/model_usage.pkl')

    X_train_usage.to_pickle(f'{EXPORT_PATH}/X_train_usage.pkl')

    X_test_activity.to_pickle(f'{EXPORT_PATH}/X_test_activity.pkl')
    X_test_usage.to_pickle(f'{EXPORT_PATH}/X_test_usage.pkl')
    y_test_activity.to_pickle(f'{EXPORT_PATH}/y_test_activity.pkl')


if __name__ == '__main__':
    train_models()
