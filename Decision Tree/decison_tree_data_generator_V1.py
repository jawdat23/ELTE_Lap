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
        if(not c_data ):
            return fill_max(row,attributes,data)
        if(row[attr] != ""):
            c_data = get_splitted_data(c_data, newAttr, attr, row[attr])
            newAttr = newAttr[:]
            newAttr.remove(attr)
    for attr in attributes:
        if(row[attr] == ""):
            total = 0
            total_num = 0
            rateVals = {}
            list_of_val = get_attribute_values(c_data,attributes,attr)
            for val in list_of_val:
                total_num += get_number_of_vals(data, attributes, attr, val)
            for val in list_of_val:
                num = get_number_of_vals(data, attributes, attr, val)
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
    return row

def fill_max(row,attributes,data):
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




# building the tree, every node is a key and a childern which is either a dicionary if it is not leaf and a string if it is a leaf
def build_tree(data, attributes, target):
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)
    if not data or (len(attributes) - 1) <= 0:
        tree = {'best': {}}
        tree['best']['val'] = default
        return tree
    elif vals.count(vals[0]) == len(vals):
        tree = {'best':{}}
        tree['best']['val'] = vals[0]
        return tree
    else:
        best = choose_attribute(data, attributes, target)
        tree = {best: {}}

        for val in get_attribute_values(data, attributes, best):
            new_data = get_splitted_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)

            tree[best][val] = subtree
            if(not isinstance(subtree, str)):
                tree[best][val]['rate'] = len(new_data) * 100 / len(data)
            tree[best][val][0] = len(new_data) * 100 / len(data)
            tree[best]['major'] = majorClass(newAttr, new_data, target)
            tree[best]['father'] = tree

    return tree
def get_attribute_index(data,attributes):
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
def evaluate(generated_data,data,attributes):
    attributes_index = get_attribute_index(data,attributes)
    numed_data = []
    numer_gen_data = []
    for row in data:
        i = 0
        numed_row = []
        for val in row:
            v = attributes_index[attributes[i]][val]
            i+=1
            numed_row.append(v)
        numed_data.append(numed_row)
    for row in generated_data:
        i = 0
        numed_row = []
        take = False
        for val in row:
            v = attributes_index[attributes[i]][row[val]]
            i+=1
            numed_row.append(v)
        numer_gen_data.append(numed_row)


def run_decision_tree():
    data = []
    # fileName = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    with open("cars.data") as tsv:
        for line in csv.reader(tsv, delimiter=","):
            data.append(tuple(line))
    random.shuffle(data)
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'quality']
    target = attributes[-1]
    tree = DecisionTree()
    tree.learn(data, attributes, target)
    generated_data = []
    for i in range(1,2):
        tempDict = tree.tree.copy()
        row = {}
        for l in attributes:
            row[l] = ""
        while (True):
            root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
            total = 0
            rateVals = {}
            tempDict = tempDict[list(tempDict.keys())[0]]
            if(root.value == "best"):
                row['quality'] = tempDict['val']
                break
            list_of_val = get_attribute_values(data, attributes, root.value)
            for vald in list_of_val:
                rateVals[vald] = total
                try:
                    total += tempDict[vald]['rate']
                except:
                    total+=0
            r = random.uniform(0, 1)*100
            chosenVal = None
            for vald in list_of_val:
                chosenVal = vald
                break
            for vald in list_of_val:
                if(r > rateVals[vald]):
                    chosenVal = vald
            row[root.value] = chosenVal
            tempDict = tempDict[chosenVal]
        row = fill_row(data,attributes,row)
        generated_data.append(row)
    evaluate(generated_data,data,attributes)


run_decision_tree()
print("done")
