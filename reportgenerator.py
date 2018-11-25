import os
import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class ReportGenerator():
    def __init__(self, name):
        self.name = name
        self.df = pd.DataFrame(columns=['model_name', 'series_name', 'error', 'time_taken'])

    def saveReport(self):
        name = os.path.join(os.path.dirname(__file__), "reports", self.name+'.csv')
        self.df.to_csv(name, encoding='utf-8')

    def _add_to_report(self, model_name, series_name, error, time_taken):
        row = {}
        row['model_name'] = model_name
        row['series_name'] = series_name
        row['error'] = error
        row['time_taken'] = time_taken
        self.df = self.df.append(row, ignore_index=True)

    def generate_report(self, X_train, X_test, y_train, y_test, model, series_name, model_name):
        start = time.clock()

        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        error = self._get_error(y_test, predictions)

        time_taken = time.clock() - start

        self._add_to_report(model_name, series_name, error, time_taken)
        self._save_plot(y_test, predictions, model_name, series_name)
        print("Time required to evaluate model - {} on series - {} with error - {} is = {}".format(model_name, series_name, error, time_taken))

    def _get_error(self, true, pred):
        return mean_squared_error(true, pred)

    def _save_plot(self, target, prediction, model_name, series_name):
        plt.figure(figsize=(11, 1.5))
        plt.plot(target, 'k', label="Series - {} Test Target".format(series_name))
        plt.plot(prediction, 'r', label="Model - {}".format(model_name))
        plt.legend(loc='upper left', fontsize='x-small')

        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", model_name+series_name+'.png'))
        plt.close()
