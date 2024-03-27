import pandas as pd
from agents import X_Recommendation_Agent


EXPORT_PATH = '../exportNew/'

def predict(prediction_date):
    shiftable_devices = ['Tumble Dryer', 'Washing Machine', 'Dishwasher']
    # Load preprocessed data for evaluation or prediction
    activity_df = pd.read_pickle(f'{EXPORT_PATH}/activity_df.pkl')
    load_df = pd.read_pickle(f'{EXPORT_PATH}/load_df.pkl')
    usage_df = pd.read_pickle(f'{EXPORT_PATH}/usage_df.pkl')
    price_df = pd.read_pickle(f'{EXPORT_PATH}/price_df.pkl')

    recommend = X_Recommendation_Agent(activity_df, usage_df, load_df, price_df, shiftable_devices, model_type='random forest')
    price = recommend.electricity_prices_from_start_time(prediction_date)
    table = recommend.pipeline(date=prediction_date, activity_prob_threshold=0.4, usage_prob_threshold=0.7, evaluation=False, weather_sel=False)
    result = recommend.visualize_recommendation(table, price, diagnostics=False)  # Adjust parameters as needed
    return result

if __name__ == '__main__':
    prediction_date = '2024-08-21'  # Example prediction date
    result,ebm_local = predict(prediction_date)
    print(result)

