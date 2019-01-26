import csv
import math
import random


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


# building the tree, every node is a key and a childern which is either a dicionary if it is not leaf and a string if it is a leaf
def build_tree(data, attributes, target):
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):

        return vals[0]
    else:
        best = choose_attribute(data, attributes, target)
        tree = {best: {}}

        for val in get_attribute_values(data, attributes, best):
            new_data = get_splitted_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree
            tree[best]['rate'] = len(new_data) * 100 / len(data)
            tree[best]['major'] = majorClass(newAttr, new_data, target)
            tree[best]['father'] = tree

    return tree


def run_decision_tree():
    data = []
    # fileName = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    with open("cars.data") as tsv:
        for line in csv.reader(tsv, delimiter=","):
            data.append(tuple(line))

    print("Number of records: %d" % len(data))

    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'quality']
    target = attributes[-1]
    K = 10
    J = 2000
    acc = []
    for k in range(K):
        random.shuffle(data)
        training_set = [x for i, x in enumerate(data) if i % K != k]
        test_set = [x for i, x in enumerate(data) if i % K == k]
        tree = DecisionTree()
        tree.learn(training_set, attributes, target)
        results = []
        rate = 1
        tempDict = tree.tree.copy()
        for entry in test_set:
            tempDict = tree.tree.copy()
            result = ""
            while (isinstance(tempDict, dict)):
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
                tempDict = tempDict[list(tempDict.keys())[0]]
                index = attributes.index(root.value)
                value = entry[index]
                if (value in list(tempDict.keys())):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    rate = tempDict['rate']
                    tempDict = tempDict[value]
                else:
                    result = tempDict['major']
                    break
            results.append(result == entry[-1])

        accuracy = float(results.count(True)) / float(len(results))
        acc.append(accuracy)
        print("Accuracy in fold ", k + 1, " is ", accuracy)

    avg_acc = sum(acc) / len(acc)
    print("Average accuracy: %.4f" % avg_acc)


run_decision_tree()