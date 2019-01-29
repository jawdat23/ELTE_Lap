import csv
import math
import random
import numpy as np

i =0
class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


class Node():
    value = None
    rate = None
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        #         self.rate = rat
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# this function is to get the most freqent target in a specific dataset
def majorClass(attributes, data, target):
    #     print("data Length: ", len(data))
    freqeuntTarget = {}
    index = attributes.index(target)
    #     print("Index: ",index)
    for tuple in data:
        #         if index < 4:
        #           print (tuple)
        if (tuple[index] in freqeuntTarget):
            freqeuntTarget[tuple[index]] += 1
        else:
            freqeuntTarget[tuple[index]] = 1

    max = 0
    major = ""

    for key in freqeuntTarget.keys():
        if freqeuntTarget[key] > max:
            max = freqeuntTarget[key]
            major = key

    return major


# this function gets the entropy of a dataset that has targetAttr as its target attribute
def entropy(attributes, data, targetAttr):
    freqeuntTarget = {}
    dataEntropy = 0.0
    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1
    i = i - 1

    for entry in data:
        if (entry[i] in freqeuntTarget):
            freqeuntTarget[entry[i]] += 1.0
        else:
            freqeuntTarget[entry[i]] = 1.0

    for freqeuntTarget in freqeuntTarget.values():
        dataEntropy += (-freqeuntTarget / len(data)) * math.log(freqeuntTarget / len(data), 2)

    return dataEntropy


# this fucntion is to get the inforamtion gain of selecting an attribute to split in a specific dataset (data)
def information_gain(attributes, data, attr, targetAttr):
    freqeuntTarget = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (entry[i] in freqeuntTarget):
            freqeuntTarget[entry[i]] += 1.0
        else:
            freqeuntTarget[entry[i]] = 1.0

    for val in freqeuntTarget.keys():
        valProb = freqeuntTarget[val] / sum(freqeuntTarget.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# this function goest between the attributes and get the attribute with the most information gain
def choose_attribute(data, attributes, target):
    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = information_gain(attributes, data, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr

    return best


# this function is to get the distinct values of an attribute
def get_attribute_values(data, attributes, attr):
    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values


# get the data splitted on a specific attribute for a scpecific value
def get_splitted_data(data, attributes, best, val):
    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0, len(entry)):
                if (i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])
    return new_data


def get_number_of_vals(data, attributes, best, val):
    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0, len(entry)):
                if (i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])
    return len(new_data)


def fill_row(data, attributes, row):
    c_data = data
    newAttr = attributes
    for attr in attributes:
        if (not c_data):
            print("here")
            return fill_max(row, attributes, data)
        if (row[attr] != ""):
            c_data = get_splitted_data(c_data, newAttr, attr, row[attr])
            newAttr = newAttr[:]
            newAttr.remove(attr)
    news_Attributes = newAttr
    for attr in attributes:
        if (row[attr] == ""):
            total = 0
            total_num = 0
            rateVals = {}
            list_of_val = get_attribute_values(c_data, newAttr, attr)
            for val in list_of_val:
                total_num += get_number_of_vals(c_data, newAttr, attr, val)
            for val in list_of_val:
                num = get_number_of_vals(c_data, newAttr, attr, val)
                rateVals[val] = total
                total += num / total_num
            r = random.uniform(0, 1) * 100
            chosenVal = None
            for vald in list_of_val:
                chosenVal = vald
                break
            for vald in list_of_val:
                if (r > rateVals[vald]):
                    chosenVal = vald
            row[attr] = chosenVal
            c_data = get_splitted_data(c_data, newAttr, attr, row[attr])
            newAttr = newAttr[:]
            newAttr.remove(attr)
    return row


def fill_max(row, attributes, data):
    for attr in attributes:
        if (row[attr] == ""):
            row[attr] = get_most_occured(data, attributes, attr)


def get_most_occured(data, attributes, attr):
    list_of_val = get_attribute_values(data, attributes, attr)
    max_num = 0
    chosenVal = ""
    for val in list_of_val:
        num = get_number_of_vals(data, attributes, attr, val)
        if max_num < num:
            max_num = num
            chosenVal = val
    return chosenVal


def evaluate_per(generated_data, data, attributes):
    fixed_gen_data = []
    for row in generated_data:
        fixed_row = []
        for val in row:
            fixed_row.append(row[val])
        fixed_gen_data.append(fixed_row)
    for attr in attributes:
        for val in get_attribute_values(data, attributes, attr):
            g1 = get_occurance_time(data, attributes, attr, val)
            g1 = g1 * 100 / len(data)
            g2 = get_occurance_time(fixed_gen_data, attributes, attr, val)
            g2 = g2 * 100 / len(fixed_gen_data)
            print("Attribute: ", attr, " has taken the value: (", val, ") in ", g1, "% of the original data and", g2,
                  "% of the generated data")


def get_occurance_time(data, attributes, attr, val):
    num = 0;
    index = attributes.index(attr)
    for entry in data:
        if (entry[index] == val):
            num += 1
    return num


# building the tree, every node is a key and a childern which is either a dicionary if it is not leaf and a string if it is a leaf
def build_tree(data, attributes, target):
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)
    if not data or (len(attributes) - 1) <= 0:
        tree = {'best': {}}
        tree['best']['stats'] = None
        tree['best']['val'] = default
        return tree
    elif vals.count(vals[0]) == len(vals):
        tree = {'best': {}}
        tree['best']['val'] = vals[0]
        attrs = attributes[:]
        attrs.remove(target)
        tree['best']['stats'] = get_data_stats(data,attrs)
        return tree
    else:
        best = choose_attribute(data, attributes, target)
        tree = {best: {}}
        tree[best]['values'] = []
        for val in get_attribute_values(data, attributes, best):
            tree[best]['values'].append(val)
            new_data = get_splitted_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)

            tree[best][val] = subtree
            if (not isinstance(subtree, str)):
                tree[best][val]['rate'] = len(new_data) * 100 / len(data)
            tree[best][val][0] = len(new_data) * 100 / len(data)
            tree[best]['major'] = majorClass(newAttr, new_data, target)
            tree[best]['father'] = tree

    return tree


def get_attribute_index(data, attributes):
    attributes_index = {}
    for attr in attributes:
        list_of_val = {}
        lend = len(get_attribute_values(data, attributes, attr))
        i = 1
        for val in get_attribute_values(data, attributes, attr):
            list_of_val[val] = i / lend
            i += 1
        attributes_index[attr] = list_of_val
    return attributes_index


def evaluate(generated_data, data, attributes):
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'quality']
    fixed_gen_data = []
    for row in generated_data:
        fixed_row = []
        for val in row:
            fixed_row.append(row[val])
        fixed_gen_data.append(fixed_row)
    attributes_index = get_attribute_index(data, attributes)
    numed_data = []
    numed_gen_data = []
    for row in data:
        i = 0
        numed_row = []
        for val in row:
            try:
                v = attributes_index[attributes[i]][val]
            except:
                v = 0
            i += 1
            numed_row.append(v)
        numed_data.append(numed_row)
    for row in fixed_gen_data:
        i = 0
        numed_row = []
        for val in row:
            try:
                v = attributes_index[attributes[i]][row[val]]
            except:
                v = 0
            i += 1
            numed_row.append(v)
        numed_gen_data.append(numed_row)
    print("Covariance for Original Data: ", np.cov(numed_data))
    print("Covariance for Generated Data: ", np.cov(numed_gen_data))
    print("Covariance of the 2 metrices: ", np.cov(numed_gen_data, numed_data))
    print("Correlation of the 2 metrices: ", np.corrcoef(numed_data, numed_gen_data))

def get_data_stats(data,attributes):
    stats = {}
    stats['attributes'] =[]
    if not data or (len(attributes) - 1) <= 0:
        return ""
    for attr in attributes:
        stats['attributes'].append(attr)
        stats[attr] = {}
        stats[attr]["values"] =[]
        total = 0
        total_num = 0
        rateVals = {}
        list_of_val = get_attribute_values(data, attributes, attr)
        for val in list_of_val:
            total_num += get_number_of_vals(data, attributes, attr, val)
        newAttr = attributes[:]
        newAttr.remove(attr)
        for val in list_of_val:
            stats[attr]["values"].append(val)
            num = get_number_of_vals(data, attributes, attr, val)
            rateVals[val] = total
            total += num / total_num
            c_data = get_splitted_data(data, attributes, attr, val)
            subtree = get_data_stats(c_data,newAttr)
            stats[attr][val] = subtree
            if (not isinstance(subtree, str)):
                stats[attr][val]['rate'] = num / total_num
    return stats

def fill_row_using_stats(stats,row):
    for attr in stats['attributes']:
        rateVals = {}
        total = 0
        for vald in stats[attr]['values']:
            if(stats[attr][vald] == ''):
                return  row
            rateVals[vald] = total
            try:
                total += stats[attr][vald]['rate']
            except:
                print(stats[attr])
                print(stats[attr][vald])
        r = random.uniform(0, 1) * 100
        chosenVal = None
        for vald in stats[attr]['values']:
            chosenVal = vald
            break
        for vald in stats[attr]['values']:
            if (r > rateVals[vald]):
                chosenVal = vald
        row[attr] = chosenVal
    return row

def read_data(file_path):
    data = []
    # fileName = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    with open(file_path) as tsv:
        for line in csv.reader(tsv, delimiter=","):
            data.append(tuple(line))
    random.shuffle(data)
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'quality']
    target = attributes[-1]
    return data,attributes,target

def run_decision_tree(data,attributes,target):
    tree = DecisionTree()
    tree.learn(data, attributes, target)
    attributes.remove(target)
    return tree

def generate_data(tree,attributes,length):
    generated_data = []
    for i in range(0, length):
        tempDict = tree.tree.copy()
        row = {}
        for l in attributes:
            row[l] = ""
        while (True):
            root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
            total = 0
            rateVals = {}
            tempDict = tempDict[list(tempDict.keys())[0]]
            if (root.value == "best"):
                row['quality'] = tempDict['val']
                if(tempDict['stats']):
                    row = fill_row_using_stats(tempDict['stats'],row)
                    generated_data.append(row)
                else:
                    generated_data.append(row)
                break
            list_of_val = tempDict['values']
            for vald in list_of_val:
                rateVals[vald] = total
                total += tempDict[vald]['rate']
            r = random.uniform(0, 1) * 100
            chosenVal = None
            for vald in list_of_val:
                chosenVal = vald
                break
            for vald in list_of_val:
                if (r > rateVals[vald]):
                    chosenVal = vald
            row[root.value] = chosenVal
            tempDict = tempDict[chosenVal]
    return generated_data

file_name = "cars.data"
(data,attributes,target) = read_data(file_name)
tree = run_decision_tree(data,attributes,target)
generated_data = generate_data(tree,attributes,2000)
evaluate_per(generated_data, data, attributes)
print("Evaluation done")
# evaluate(generated_data, data, attributes)
# print("Evaluation done")
print("done")
