import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.model = {}
        self.training_data = None
        self.testing_labels = None
        self.testing_data = None
        self.prior_frequencies = {}
        self.distinct_nominal_values = {}
        self.target_class = None
        self.data = None
        self.training_percentage = None
        self.target_class = None
        self.df = None

    def read_data(self, file_name):
        self.df = pd.read_csv(file_name)

    def _split_data(self):
        rnd_perm = np.random.permutation(list(range(len(self.df))))
        training_instances_number = int(len(self.df) * self.training_percentage)
        self.training_data = self.df.ix[rnd_perm[:training_instances_number]]
        self.testing_data = self.df.ix[rnd_perm[training_instances_number:]]
        self.testing_labels = list(self.testing_data[self.target_class])
        self.testing_data.drop(self.target_class, axis=1, inplace=True)

    def remove_columns(self, columns_name):
        for column_name in columns_name:
            self.df.drop(column_name, axis=1, inplace=True)

    def _set_prior_frequencies(self):
        for index, row in self.training_data.iterrows():
            if row[self.target_class] in self.prior_frequencies.keys():
                self.prior_frequencies[row[self.target_class]] += 1
            else:
                self.prior_frequencies[row[self.target_class]] = 1

    def _update_model_for_nominal(self,attribute):
        frequencies = {}
        for index, row in self.training_data.iterrows():
            total_instances = self.prior_frequencies[row[self.target_class]] + len(self.distinct_nominal_values[attribute])
            if row[attribute] in frequencies.keys():
                old_val = frequencies[row[attribute]][row[self.target_class]]
                frequencies[row[attribute]][row[self.target_class]] = old_val + 1/total_instances
            else:
                frequencies[row[attribute]] = {}
                for att in self.distinct_nominal_values[self.target_class]:
                    frequencies[row[attribute]][att] = 1/total_instances
                frequencies[row[attribute]][row[self.target_class]] = 2/total_instances
        self.model[attribute] = frequencies

    def _update_model_for_numeric(self):
        pass

    def _transform_to_probabilities(self):
        correction = 0
        for (key, value) in self.distinct_nominal_values.items():
            correction += len(value)
        correction -= len(self.distinct_nominal_values[self.target_class])

        data_length = len(self.training_data) + correction * len(self.distinct_nominal_values[self.target_class])
        for key in self.distinct_nominal_values[self.target_class]:
            freq = self.prior_frequencies[key]
            prob = freq / data_length
            self.prior_frequencies[key] = prob

    def build_model(self, target_class, continues):
        self.target_class = target_class
        self._split_data()

        '''get distinct nominal values'''
        for attribute in self.df.keys():
            if attribute not in continues:
                self.distinct_nominal_values[attribute] = self.df[attribute].unique()

        self._set_prior_frequencies()

        for attribute in self.training_data.keys():
            if attribute in self.distinct_nominal_values.keys():
                self._update_model_for_nominal(attribute)
            else:
                self._update_model_for_numeric()

    def _class_probability(self, row, target_value):
        product = self.prior_frequencies[target_value]
        for attribute in self.testing_data.keys():
            value = row[attribute]
            if attribute in self.distinct_nominal_values:
                prob = self.model[attribute][value][target_value]
                product *= prob
            else:
                '''numeric'''
                pass
        return product

    def _classify_row(self, row):
        mx = -1
        best_value = None
        for target_value in self.distinct_nominal_values[self.target_class]:
            prob = self._class_probability(row, target_value)
            if mx < prob:
                mx = prob
                best_value = target_value
        return best_value

    def classify(self):
        self._transform_to_probabilities()
        result = []
        for index, row in self.testing_data.iterrows():
            classified_value = self._classify_row(row)
            result.append(classified_value)
        return result