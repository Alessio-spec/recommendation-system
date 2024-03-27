import pandas as pd
import json
from os import walk
import numpy as np

import matplotlib.pyplot as plt

class Helper:

    @staticmethod
    def read_txt(filename):
        # fl = open(filename, 'r')
        # print(fl.read())
        # fl.close
        with open(filename, 'r') as fl:
            print(fl.read())

    def get_timespan(self, df, start, timedelta_params):

        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end]

    @staticmethod
    def load_txt(filename):
        with open(filename, 'r') as fl:
            return fl.read()
    # def load_txt(filename):
    #     fl = open(filename, 'r')
    #     output = fl.read()
    #     fl.close
    #     return output
    def get_column_labels(self, filename):
        columns = {}
        readme = self.load_txt(filename)
        temp = readme[readme.find('\nHouse'):]

        for house in range(1, 4):
            cols = {}
            temp = readme[readme.find('\nHouse '+str(house)):]
    
            for idx in range(10):
                start = temp.find(str(idx)+'.')+2
                stop = temp.find(',') if temp.find(',') < temp.find('\n\t') else temp.find('\n\t')
                cols.update({'Appliance'+str(idx):temp[start:stop]})
                temp = temp[stop+1:]

            columns.update({house: cols})
        return columns
    def save_weather(self, REFIT_dir = '../data/', house_id = 3, EXPORT_PATH = '../export/'):

        data_sets = {id: f'CLEAN_House{id}.csv' for id in range(1, 4)}
        filename = REFIT_dir + data_sets[house_id]

        readme = REFIT_dir + 'REFIT_Readme.txt'
        columns = self.get_column_labels(readme)

        house = pd.read_csv(filename)
        house.rename(columns=columns[house_id], inplace=True)

        house['Time'] = pd.to_datetime(house['Time'])#here
        house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)

        # Add Weather
        ################################
        from meteostat import Point, Hourly
        from datetime import datetime

        lough = Point(52.766593, -1.223511)
        time = house.index.to_series(name="time")
        time = time.dt.floor('H').tolist()
        start_time = time[0]
        end_time = time[len(house) - 1]
        weather = Hourly(lough, start_time, end_time)
        weather = weather.fetch()

        from sklearn.impute import KNNImputer
        # import numpy as np

        headers = weather.columns.values

        empty_train_columns = []
        for col in weather.columns.values:
            if sum(weather[col].isnull()) == weather.shape[0]:
                empty_train_columns.append(col)
        headers = np.setdiff1d(headers, empty_train_columns)

        imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
        weather = imputer.fit_transform(weather)
        weather = pd.DataFrame(weather)

        time_series = house.index.to_series().dt.floor('H').drop_duplicates()
        time_unique = time_series[time_series.duplicated() == False]  # This ensures unique times only
        weather_truncated = weather.iloc[:len(house)]
        house_truncated = house.iloc[:len(weather)]
        house.index = time_unique
        weather.index = time_unique[:len(weather)]  # Initially set weather index, might be shorter

        # Reindex the weather DataFrame to match the house DataFrame's index
        # Forward fill the missing values
        weather_aligned = weather.reindex(house.index, method='ffill')

        # When merging, instead of using 'time' as a key, we can use the indices directly
        # weather.index = time_unique  # Align the weather data index with the house data index
        # weather_reindexed = weather.reindex(time_series)
        # Merge the house and weather DataFrames based on the index

        # time_unique = pd.date_range(start_time, end_time, freq='h')
        # weather["time"] = time_unique
        # house["time"] = time

        # weather.columns = np.append(headers, "time")

        house = pd.merge(house, weather_aligned, left_index=True, right_index=True, how='left')

        # house = pd.merge(house, weather, how="left", on="time")
        house.drop("time", axis=1, inplace=True)
        house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)
        ################################

        # Daily
        # params = {"resample_param": "24H"}
        daily_weather = house.resample("24H").mean().copy()
        daily_weather.to_pickle(EXPORT_PATH + "weather_unscaled_daily.pkl")
        # Hourly
        # params = {"resample_param": "60T"}
        hourly_weather = house.resample("60T").mean().copy()
        hourly_weather.to_pickle(EXPORT_PATH + "weather_unscaled_hourly.pkl")

    def load_household(self, REFIT_dir, house_id, weather_sel=False):

        data_sets = {id:f'CLEAN_House{id}.csv' for id in range(1,4)}
        filename = REFIT_dir + data_sets[house_id]

        readme = REFIT_dir + 'REFIT_Readme.txt'
        columns = self.get_column_labels(readme)

        house = pd.read_csv(filename)
        house.rename(columns=columns[house_id], inplace=True)
        house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)

        if weather_sel:

            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = house.index.to_series(name="time")
            time = time.dt.floor('H').tolist()
            start_time = time[0]
            end_time = time[len(house) - 1]
            weather = Hourly(lough, start_time, end_time)
            weather = weather.fetch()

            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np

            headers = weather.columns.values

            empty_train_columns = []
            for col in weather.columns.values:
                if sum(weather[col].isnull()) == weather.shape[0]:
                    empty_train_columns.append(col)
            headers = np.setdiff1d(headers, empty_train_columns)

            imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
            weather = imputer.fit_transform(weather)
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            time_unique = pd.date_range(start_time, end_time, freq='h')
            weather["time"] = time_unique
            house["time"] = time

            weather.columns = np.append(headers, "time")

            house = pd.merge(house, weather, how="left", on="time")
            house.drop("time", axis=1, inplace=True)
            house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)
            ################################

        return house

    # def aggregate(self, df, resample_param):
    #     return df.resample(resample_param).mean().copy()

    def aggregate(self, df, resample_param='60T'):
        # Ensure the datetime index is set for resampling
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for resampling.")

        # Select only numeric columns for aggregation
        numeric_df = df.select_dtypes(include=[np.number])

        # Perform the aggregation on numeric columns
        aggregated_df = numeric_df.resample(resample_param).mean()

        # Handle non-numeric columns separately, if needed
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        for column in non_numeric_df.columns:
            # Example: taking the first value in each aggregation period for non-numeric columns
            aggregated_df[column] = non_numeric_df[column].resample(resample_param).first()

        return aggregated_df.copy()

    def plot_consumption(self, df, features='all', figsize='default', threshold=None, title='Consumption'):
     
        df = df.copy()
        features = [column for column in df.columns if column not in ['Unix', 'Issues']] if features == 'all' else features

        fig, ax = plt.subplots(figsize=figsize) if figsize != 'default' else plt.subplots()
        if threshold != None:
            df['threshold'] = [threshold]*df.shape[0]
            ax.plot(df['threshold'], color = 'tab:red')
        for feature in features:
            ax.plot(df[feature])
        ax.legend(['threshold'] + features) if threshold != None else ax.legend(features)
        ax.set_title(title);

    def create_day_ahead_prices_df(self, FILE_PATH, filename):
      
      electricity_prices1 = pd.read_csv(FILE_PATH + filename)
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.replace("2025", "2023")

      electricity_prices2 = pd.read_csv(FILE_PATH + filename)
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.replace("2025", "2024")

      electricity_prices3 = pd.read_csv(FILE_PATH + filename)
      electricity_prices3["MTU (UTC)"] =  electricity_prices3["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]

      electricity_prices = pd.concat([electricity_prices1, electricity_prices2, electricity_prices3])
      electricity_prices.columns = ["Time", "Price"]
      electricity_prices = electricity_prices.set_index(pd.DatetimeIndex(electricity_prices['Time']), drop = True)
      electricity_prices = electricity_prices["Price"]
      return electricity_prices

    def concat_household_scores(self, agent_scores):
        df_names = list(list(agent_scores.values())[0].keys())
        output = {}
        for name in df_names:
            output[name] = pd.concat([scores[name] for household, scores in agent_scores.items()])
        return pd.concat(output, axis=1)

    def shiftable_device_legend(self, EXPORT_PATH):
       
        # get config files stored at the export path
        _, _, filenames = next(walk(EXPORT_PATH))
        config_files = [file for file in filenames if file.find('config.json') != -1]

        legend_shiftable_devices = pd.DataFrame()
        for config_file in config_files:
            config = json.load(open(EXPORT_PATH+config_file, 'r'))
            household_id = config['data']['household']
            devices = config['user_input']['shiftable_devices']
            i = 0
            for device in devices:
                legend_shiftable_devices.loc[household_id, i] = device
                i += 1

        legend_shiftable_devices.sort_index(inplace=True)        
        legend_shiftable_devices.columns.name = 'device'
        legend_shiftable_devices.index.name = 'household'
        return legend_shiftable_devices