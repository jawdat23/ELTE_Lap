import pandas as pd
import numpy as np
from statistical_helper import StatisticalHelper
import random
import operator
import csv


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
        self.generated_df = None

    def read_data(self, file_name):
        self.df = pd.read_csv(file_name)

    def _split_data(self):
        rnd_perm = np.random.permutation(list(range(len(self.df))))
        training_instances_number = int(len(self.df) * self.training_percentage)
        self.training_data = self.df.ix[rnd_perm[:training_instances_number]]
        self.testing_data = self.df.ix[rnd_perm[training_instances_number:]]
        self.testing_labels = list(self.testing_data[self.target_class])
        self.testing_data.drop(self.target_class, axis=1, inplace=True)

    def split_contentious_data(self, contentious_attributes,target, number):
        for contentious_attribute in contentious_attributes:
            df = self.df.sort_values(by=contentious_attribute)
            values = df[contentious_attribute].tolist()
            targets = df[target].tolist()
            result = StatisticalHelper.split_point(values, targets,number-1)
            for i,row in self.df.iterrows():
                if row[contentious_attribute] <= result[0]:
                    self.df.at[i,contentious_attribute] = 0
                    continue
                if row[contentious_attribute] > result[-1]:
                    self.df.at[i, contentious_attribute] = number
                    continue
                for j in range(0, len(result) - 1):
                    if row[contentious_attribute] > result[j] and row[contentious_attribute] <= result[j+1]:
                        self.df.at[i, contentious_attribute] = j + 1

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

    def _update_model_for_numeric(self, attribute):
        self.model[attribute] = {}
        self.model[attribute]['mean'] = self.training_data[attribute].mean()
        self.model[attribute]['std_dev'] = self.training_data[attribute].std()

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
                self._update_model_for_numeric(attribute)

    def _class_probability(self, row, target_value):
        product = self.prior_frequencies[target_value]
        for attribute in self.testing_data.keys():
            value = row[attribute]
            if attribute in self.distinct_nominal_values:
                prob = self.model[attribute][value][target_value]
                product *= prob
            else:
                '''numeric'''
                product *= self._get_numeric_field_probability(attribute, value)
        return product

    def _get_numeric_field_probability(self, attribute, value):
        mean = self.model[attribute]['mean']
        std_dev = self.model[attribute]['std_dev']
        return StatisticalHelper.normal_dist(value, mean, std_dev)

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


    def _get_value_based_on_prob(self, dic):
        prop = random.uniform(0, 1)
        acc = 0
        '''the flag is for: if none is selected then select the last one'''
        flag = True
        for k, v in dic:
            if prop < v + acc:
                selected_value = k
                flag = False
                break
            else:
                acc += v
        if flag:
            selected_value = k
        return selected_value

    def _get_value_based_on_prob_correlated_attribute(self,attribute, dic, dependent_attributes,known_values, data_length):
        prop = random.uniform(0, 1)
        temp = dic['dependent']
        for att in dependent_attributes:
            temp = temp[known_values[att]]

        denomirator = temp/data_length
        temp = dic['all']
        values_probability = {}
        for att in dependent_attributes:
            temp = temp[known_values[att]]

        for distinct_value in self.distinct_nominal_values[attribute]:
            try:
                values_probability[distinct_value] = (temp[distinct_value]/data_length) / denomirator
            except:
                continue

        values_probability = dict(sorted(values_probability.items(), key=operator.itemgetter(1), reverse=True))
        acc = 0
        flag = True

        for k, v in values_probability.items():
            if prop < v + acc:
                selected_value = k
                flag = False
                break
            else:
                acc += v
        if flag:
            selected_value = k
        return selected_value

    def generate_data(self, number,output_file):
        data_set_length = len(self.training_data)
        values_probability = {}
        target_prop = {}
        for k, v in self.prior_frequencies.items():
            target_prop[k] = v/data_set_length
        values_probability[self.target_class] = sorted(target_prop.items(), key=operator.itemgetter(1), reverse=True)
        for attribute in self.df.keys():
            if self.target_class == attribute:
                continue
            values_probability[attribute] = {}
            dic = {}
            for target_value, v in self.prior_frequencies.items():
                for distict_value in self.distinct_nominal_values[attribute]:
                    dic[distict_value] = self.model[attribute][distict_value][target_value]
                values_probability[attribute][target_value] = sorted(dic.items(), key=operator.itemgetter(1),
                                                          reverse = True)
        generated_data = []
        for i in range(0,number):
            instance = {}
            selected_class = self._get_value_based_on_prob(values_probability[self.target_class])
            instance[self.target_class] = selected_class
            for attribute in self.df.keys():
                if attribute == self.target_class:
                    continue
                instance[attribute] = self._get_value_based_on_prob(values_probability[attribute][selected_class])
            generated_data.append(instance)

        with open(output_file, 'w', newline='') as f:
            w = csv.DictWriter(f, generated_data[0].keys())
            w.writeheader()
            for row in generated_data:
                w.writerow(row)

    def generate_data_for_correlated_attributes(self, number, output_file, dependencies):
        data_set_length = len(self.training_data)
        if len(self.df.columns) < len(dependencies) + 2:
            raise Exception('loop depedencies')
        values_probability = {}
        target_prop = {}
        for k, v in self.prior_frequencies.items():
            target_prop[k] = v / data_set_length
        values_probability[self.target_class] = sorted(target_prop.items(), key=operator.itemgetter(1), reverse=True)
        for attribute in self.df.keys():
            if self.target_class == attribute:
                continue
            values_probability[attribute] = {}
            dic = {}
            for target_value, v in self.prior_frequencies.items():
                for distict_value in self.distinct_nominal_values[attribute]:
                    dic[distict_value] = self.model[attribute][distict_value][target_value]
                values_probability[attribute][target_value] = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)

        dependencies = sorted(dependencies.items(), key= operator.itemgetter(1))
        dependencies=dict(dependencies)
        depedencies_intersections_prob = {}
        for attribute, correlated in dependencies.items():
            depedencies_intersections_prob[attribute] = {}
            depedencies_intersections_prob[attribute]['dependent'] = self.df.groupby(correlated).size()
            correlated.append(attribute)
            depedencies_intersections_prob[attribute]['all'] = self.df.groupby(correlated).size()
            correlated.remove(attribute)

        generated_data = []
        for i in range(0,number):
            instance = {}
            selected_class = self._get_value_based_on_prob(values_probability[self.target_class])
            instance[self.target_class] = selected_class

            '''generating independent attributes'''
            for attribute in self.df.keys():
                if attribute == self.target_class or attribute in dependencies:
                    continue
                instance[attribute] = self._get_value_based_on_prob(values_probability[attribute][selected_class])
            '''generate dependent attributes'''
            for attribute, correlated in dependencies.items():
                instance[attribute] = self._get_value_based_on_prob_correlated_attribute(attribute,depedencies_intersections_prob[attribute],dependencies[attribute],instance, data_set_length)

            generated_data.append(instance)

        with open(output_file, 'w', newline='') as f:
            w = csv.DictWriter(f, generated_data[0].keys())
            w.writeheader()
            for row in generated_data:
                w.writerow(row)
