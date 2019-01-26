import math

class StatisticalHelper():

    @staticmethod
    def normal_dist(value, mean, std_dev):
        fact = std_dev * math.sqrt(2 * math.pi)
        expo = pow((value - mean), 2) / (2 * pow(std_dev,2))
        return math.exp(-expo) / fact

    @staticmethod
    def split_point(values, target, number):
        original_values_list = values
        original_target = target
        split_points = [0, len(original_values_list) - 1]
        mid_points = []
        for generated in range(0, number):
            '''select which part to split based on minimum entropy'''
            b = 0
            e = len(original_values_list)
            if generated != 0:
                max_ent = 0

                for i in range(0,generated+1):
                    data_part = original_values_list[split_points[i]: split_points[i+1]]
                    target_part = original_target[split_points[i]: split_points[i+1]]
                    ent = StatisticalHelper.calculate_entropy(data_part, target_part, data_part[len(data_part)-1])
                    if max_ent < ent:
                        max_ent = ent
                        b = split_points[i]
                        e = split_points[i+1]
            values = original_values_list[b:e]
            target = original_target[b:e]
            values_set = list(set(values))
            minimum_expected_info = 999999999
            for i in range(0, len(values_set)-1):
                mid_point = (values_set[i+1] - values_set[i])/2 + values_set[i]
                entropy = StatisticalHelper.calculate_entropy(values, target, mid_point)
                if minimum_expected_info > entropy:
                    minimum_expected_info = entropy
                    end = values_set[i+1]
                    selected_mid_point = mid_point

            split_point = original_values_list.index(end)
            split_points.append(split_point)
            split_points.sort()
            mid_points.append(selected_mid_point)

        #mid_points.append(original_values_list[0])
        #mid_points.append(original_values_list[-1])
        mid_points.sort()
        return mid_points

    @staticmethod
    def calculate_entropy(values, target, mid_point):
        prop = {'>=': {}, '<': {}}
        elements_smaller = 0
        for v in set(target):
            prop['>='][v] = 0
            prop['<'][v] = 0
        for i in range(0,len(values)):
            if mid_point >= values[i]:
                elements_smaller +=1
                prop['>='][target[i]] +=1
            else:
                prop['<'][target[i]] += 1
        elements_greater = len(values) - elements_smaller
        entropy_smaller = 0
        entropy_greater =0
        for v in set(target):
            if elements_smaller != 0 and prop['>='][v] != 0:
                entropy_smaller -= (prop['>='][v] / elements_smaller) * math.log(prop['>='][v] / elements_smaller)
            if elements_greater != 0 and prop['<'][v] != 0:
                entropy_greater -= (prop['<'][v] / elements_greater) * math.log(prop['<'][v] / elements_greater)
        entropy = (elements_greater/len(values)) * entropy_greater + (elements_smaller/len(values)) * entropy_smaller
        return entropy
