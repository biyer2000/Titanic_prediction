##############
# Name: Bharat Iyer
# email: biyer@purdue.edu
# Date: 10/27/20
#Using late day

import numpy as np
import sys
import os
import pandas as pd

def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    #print(freqs)
    all_freq = sum(freqs)
    
    entrop = 0
    for fq in freqs:
        prob = fq * 1.0 /all_freq
        if abs(prob) > 1e-8:
            entrop += -prob* np.log2(prob)
    return entrop

def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, l, r, attr, thresh,label, ratios):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.label=label
        self.ratios = ratios

class Tree(object):
    def __init__(self, traindata):
        self.traindata = traindata
    def fill_mode(self, train_data):
        for a in train_data:
            train_data[a] = train_data[a].fillna(train_data[a].mode()[0])
    def fill_median(self, train_data):
        for a in train_data:
            train_data[a] = train_data[a].fillna(train_data[a].median())
    def drop_na(self, train_data):
        train_data = train_data.dropna()
        
    def best_split(self, data, attribute):
            #print(attribute)
            yes = len(data[data.survived == 1]['survived'].tolist())
            no =  len(data[data.survived == 0]['survived'].tolist())
            #df[attribute] = df[attribute].fillna(df[attribute].mode()[0])

            values = data[attribute].unique().tolist()
            values.sort()
            vals = []
            info_gains = []
            for i in range(len(values) - 1):
                median = (values[i] + values[i + 1])/2
                vals.append(median)
                df1 = data[data[attribute] <= median]
                df2 = data[data[attribute] > median]
                died1 = df1[df1['survived'] == 0].shape[0]
                survived1 = df1[df1['survived'] == 1].shape[0]
                died2 = df2[df2['survived'] == 0].shape[0]
                survived2 = df2[df2['survived'] == 1].shape[0]
                #print([yes, no])
                #print([survived1, died1])
                #print([survived2, died2])
                #print(values[i])
                #print(median)
                info_gains.append(infor_gain([yes, no], [[survived1, died1], [survived2, died2]]))
            #print(info_gains)
            return max(info_gains), vals[info_gains.index(max(info_gains))]
            
        
        



    def ID3(self, train_data):
        # 1. use a for loop to calculate the infor-gain of every attribute
        #print(".", end='')
        infor_gain = []
        best_att = ""
        thresholds = []
        atts = []
        survived = train_data[train_data["survived"] == 1].shape[0]
        died = train_data[train_data['survived'] == 1].shape[0]
        size = train_data.shape[0]
        if survived == size:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 1, ratios)
        if train_data[train_data["survived"] == 0].shape[0] == train_data.shape[0]:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 0, ratios)
        dead_end = True

        for att in train_data:
            # 1.1 pick a threshold
            #train_data[att] = train_data[att].fillna(train_data[att].mode()[0])
            if att != 'survived':
                if len(train_data[att].unique().tolist()) != 1:
                    #checks if number of 
                    dead_end = False
                    atts.append(att)
                    val = self.best_split(train_data, att)
                    infor_gain.append(val[0])
                    thresholds.append(val[1])
                
        if dead_end:
            #if all attributes are same
            ratios = [0, 0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
                return Node(None, None, None, -1, 1, ratios)
            else:
                return Node(None, None, None, -1, 0, ratios)
                
        the_chosen_attribute = atts[infor_gain.index(max(infor_gain))]
        the_chosen_threshold = thresholds[infor_gain.index(max(infor_gain))]
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;
        #print(the_chosen_attribute)
        #print(the_chosen_threshold)
        #if train_data['survived']
        if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
            lab = 1
        else:
            lab = 0
        ratios = [0,0]
        #print(lab)
        ratios[0] = train_data[train_data['survived'] == 0].shape[0]
        ratios[1] = train_data[train_data['survived'] == 1].shape[0]
        current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, lab, ratios)
        # 4. split the data into two parts.
        # 5. call ID3() for the left parts of the data
        left_part_train_data = train_data[train_data[the_chosen_attribute] <= the_chosen_threshold]
        left_part_train_data = left_part_train_data.drop(columns = [the_chosen_attribute])
        left_subtree = self.ID3(left_part_train_data)
        right_part_train_data = train_data[train_data[the_chosen_attribute] > the_chosen_threshold]
        right_part_train_data = right_part_train_data.drop(columns = [the_chosen_attribute])
        # 6. call ID3() for the right parts of the data.
        right_subtree = self.ID3(right_part_train_data)
        current_node.left_subtree = left_subtree
        current_node.right_subtree = right_subtree
        return current_node
    #def build_tree()
    
    def ID3_depth(self, train_data, max_depth, depth):
        #print(".", end='')
        #print(depth)
        infor_gain = []
        best_att = ""
        thresholds = []
        atts = []
        #print(train_data.shape)
        #if train_data.shape[0] == 5:
            #print(train_data)
        if train_data[train_data["survived"] == 1].shape[0] == train_data.shape[0]:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 1, ratios)
        if train_data[train_data["survived"] == 0].shape[0] == train_data.shape[0]:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 0, ratios)
        dead_end = True
        if depth >= max_depth:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            if ratios[0] > ratios[1]:
                return Node(None, None, None, -1, 0, ratios)
            else:
                return Node(None, None, None, -1, 1, ratios)

        for att in train_data:
            # 1.1 pick a threshold
            #train_data[att] = train_data[att].fillna(train_data[att].mode()[0])
            if att != 'survived':
                if len(train_data[att].unique().tolist()) != 1:
                    dead_end = False
                    atts.append(att)
                    val = self.best_split(train_data, att)
                    infor_gain.append(val[0])
                    thresholds.append(val[1])
                 
        if dead_end:
            ratios = [0, 0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
                return Node(None, None, None, -1, 1, ratios)
            else:
                return Node(None, None, None, -1, 0, ratios)
                
        the_chosen_attribute = atts[infor_gain.index(max(infor_gain))]
        the_chosen_threshold = thresholds[infor_gain.index(max(infor_gain))]
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;
        #print(the_chosen_attribute)
        #print(the_chosen_threshold)
        #if train_data['survived']
        if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
            lab = 1
        else:
            lab = 0
        ratios = [0, 0]
        ratios[0] = train_data[train_data['survived'] == 0].shape[0]
        ratios[1] = train_data[train_data['survived'] == 1].shape[0]
        current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, lab, ratios)
        # 4. split the data into two parts.
        # 5. call ID3() for the left parts of the data
        left_part_train_data = train_data[train_data[the_chosen_attribute] <= the_chosen_threshold]
        left_part_train_data = left_part_train_data.drop(columns = [the_chosen_attribute])
        left_subtree = self.ID3_depth(left_part_train_data, max_depth, depth + 1)
        right_part_train_data = train_data[train_data[the_chosen_attribute] > the_chosen_threshold]
        right_part_train_data = right_part_train_data.drop(columns = [the_chosen_attribute])
        # 6. call ID3() for the right parts of the data.
        right_subtree = self.ID3_depth(right_part_train_data, max_depth, depth + 1)
        current_node.left_subtree = left_subtree
        current_node.right_subtree = right_subtree
        return current_node
    
    
    def build(self, n, data):
        values = [0,0]
        values[0] = data[data['survived'] == 0].shape[0]
        values[1] = data[data['survived'] == 1].shape[0]
        nod = Node(None, None, n.attribute, n.threshold, n.label, values)
        #nod.label = n.label
        if n.left_subtree is None and n.right_subtree is None:
            nod.left_subtree = None
            nod.right_subtree = None
            return nod
        nod.left_subtree = self.build(n.left_subtree, data[data[n.attribute] <= n.threshold])
        nod.right_subtree = self.build(n.right_subtree, data[data[n.attribute] > n.threshold])
        return nod
    
    def min_split(self, train_data, min_value):
        infor_gain = []
        best_att = ""
        thresholds = []
        atts = []
        #print(train_data.shape)
        #if train_data.shape[0] == 5:
            #print(train_data)
        if train_data[train_data["survived"] == 1].shape[0] == train_data.shape[0]:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 1, ratios)
        if train_data[train_data["survived"] == 0].shape[0] == train_data.shape[0]:
            ratios = [0,0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            return Node(None, None, None, -1, 0, ratios)
        dead_end = True

        for att in train_data:
            # 1.1 pick a threshold
            #train_data[att] = train_data[att].fillna(train_data[att].mode()[0])
            if att != 'survived':
                if len(train_data[att].unique().tolist()) != 1:
                    
                    val = self.best_split(train_data, att)
                    temp = train_data[train_data[att] <= val[1]].shape[0]
                    temp1 = train_data[train_data[att] > val[1]].shape[0]
                    #print(temp)
                    #print(temp1)
                    if temp >= min_value and  temp1 >= min_value:
                        dead_end = False
                        atts.append(att)

                        infor_gain.append(val[0])
                        thresholds.append(val[1])
        if dead_end:
            ratios = [0, 0]
            ratios[0] = train_data[train_data['survived'] == 0].shape[0]
            ratios[1] = train_data[train_data['survived'] == 1].shape[0]
            if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
                return Node(None, None, None, -1, 1, ratios)
            else:
                return Node(None, None, None, -1, 0, ratios)
                
        the_chosen_attribute = atts[infor_gain.index(max(infor_gain))]
        the_chosen_threshold = thresholds[infor_gain.index(max(infor_gain))]
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;
        #print(the_chosen_attribute)
        #print(the_chosen_threshold)
        #if train_data['survived']
        if train_data[train_data['survived'] == 1].shape[0] >= train_data[train_data['survived'] == 0].shape[0]:
            lab = 1
        else:
            lab = 0
        ratios = [0,0]
        #print(lab)
        ratios[0] = train_data[train_data['survived'] == 0].shape[0]
        ratios[1] = train_data[train_data['survived'] == 1].shape[0]
        current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, lab, ratios)
        # 4. split the data into two parts.
        # 5. call ID3() for the left parts of the data
        left_part_train_data = train_data[train_data[the_chosen_attribute] <= the_chosen_threshold]
        left_part_train_data = left_part_train_data.drop(columns = [the_chosen_attribute])
        left_subtree = self.min_split(left_part_train_data, min_value)
        right_part_train_data = train_data[train_data[the_chosen_attribute] > the_chosen_threshold]
        right_part_train_data = right_part_train_data.drop(columns = [the_chosen_attribute])
        # 6. call ID3() for the right parts of the data.
        right_subtree = self.min_split(right_part_train_data, min_value)
        current_node.left_subtree = left_subtree
        current_node.right_subtree = right_subtree
        return current_node
    
        
    def prune(self, n):
        if n.left_subtree is None and n.right_subtree is None:
            error = 0
            if n.label == 0:
                error = n.ratios[1]
            elif n.label == 1:
                error = n.ratios[0]
            return error
        lef = self.prune(n.left_subtree)
        rig = self.prune(n.right_subtree)
        if n.label == 0:
            err = n.ratios[1]
        elif n.label == 1:
            err = n.ratios[0]
        if lef+ rig >= err:
            n.left_subtree = None
            n.right_subtree = None
            #print("cutoff")
            return err
        else:
            return  lef + rig
    def make_prediction(self,dic, n):
        if (n.left_subtree is None and n.right_subtree is None):
            if n.label == 1:
                return 1
            elif n.label == 0:
                return 0
        if dic[n.attribute] <= n.threshold:
            #print("left")
            return self.make_prediction(dic, n.left_subtree)
        elif dic[n.attribute] > n.threshold:
            #print("right")
            return self.make_prediction(dic, n.right_subtree)
    def test(self, data, n, labels):
        data = data.to_dict('records')
        labels = labels.tolist()
        accuracy = 0
        for i in range(len(data)):
            result = self.make_prediction(data[i], n)
            if result == labels[i]:
                accuracy += 1
        return accuracy/len(labels)
    def count_depth(self, n):
        if n is None:
            return 0
        
        
    def print_tree(self, n):
        if n is None:
            return
        print(n.attribute)
        print(n.threshold)
        print(n.label)
        print(n.ratios)
        self.print_tree(n.left_subtree)
        self.print_tree(n.right_subtree)
    def count(self, n):
        c = 1
        if n.left_subtree:
            c += self.count(n.left_subtree)
        if n.right_subtree:
            c += self.count(n.right_subtree)
        return c
    def get_majority(self, data, n, labels ):
        data = data.to_dict('records')
        labels = labels.tolist()
        accuracy = 0
        res = []
        for i in range(len(data)):
            result = self.make_prediction(data[i], n)
            res.append(result)
        if res.count(1) >= res.count(0):
            return 1
        else:
            return 0
     
'''
class PCA(object):
    def __init__(self, n_component):
        self.n_component = n_component
    
    def fit_transform(self, train_data):
        #[TODO] Fit the model with train_data and 
        # apply the dimensionality reduction on train_data.
        
    def transform(self, test_data):
        #[TODO] Apply dimensionality reduction to test_data.
'''
        
if __name__ == "__main__":

    # parse arguments
    import argparse
    import pandas as pd

    '''

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder', dest = "training")
    parser.add_argument('--testFolder', dest = "testing")
    parser.add_argument('--model', dest = "model")
    parser.add_argument('--crossValidK', type=int, default=5, dest = "k")
    parser.add_argument('--depth', dest = 'dep')
    parser.add_argument('--minSplit', dest = 'mins')

    args = parser.parse_args()
    '''
    train_file = args[1]
    train_label = args[2]

    test_file = args[3]
    test_label = args[4]
    df = pd.read_csv(train_file, delimiter=',',  index_col=None, engine='python')   
    x1 = pd.read_csv(train_label, delimiter=',',  index_col=None, engine='python')
    df['survived'] = x1['survived']
    df1 = pd.read_csv(test_file,  delimiter=',',  index_col=None, engine='python')
    x1 = pd.read_csv(test_label, delimiter=',',  index_col=None, engine='python')
    df1['survived'] = x1['survived']
    if args.model == "vanilla":
        valids = []
        tr = Tree(df)
        tr.fill_median(df)
        accuracies = []
        trains = []
        index = 0
        k = args.k
        tr = Tree(df)
        tr.fill_mode(df)

        #tr.fill_mode(tem)
        #print(df.shape[0])
        for i in range(k - 1):
            valid = df.loc[index:index + int(df.shape[0]/k) - 1]
            end = int(df.shape[0]/k)
            #print(index)
            #print(index + end - 1)
            #print(index + end)
            train = df.drop(df.index[index:index + end])
            
            n = tr.ID3(train)
            trains.append(tr.test(train, n, train['survived']))
            print("fold=" + str(i + 1) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
            accuracies.append(tr.test(valid, n, valid['survived']))
            valids.append(train)
            index += end
        valid = df.loc[index:df.shape[0]]
        train = df.drop(df.index[index:df.shape[0]])
        n = tr.ID3(train)
        trains.append(tr.test(train, n, train['survived']))
        print("fold=" + str(k) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
        accuracies.append(tr.test(valid, n, valid['survived']))
        valids.append(train)
        dat = valids[accuracies.index(max(accuracies))]
        
        n = tr.ID3(dat)
        print("test_accuracy: " + str(tr.test(df2,n, df2['survived'] )))


        
        index = df.shape[0]/k + 1
    elif args.model == "depth":
        valids = []
        tr = Tree(df)
        dep = args.dep
        dep = int(dep)
        tr.fill_median(df)
        accuracies = []
        trains = []
        index = 0
        k = args.k
        tr = Tree(df)
        tr.fill_mode(df)

        #tr.fill_mode(tem)
        #print(df.shape[0])
        for i in range(k - 1):
            valid = df.loc[index:index + int(df.shape[0]/k) - 1]
            end = int(df.shape[0]/k)
            #print(index)
            #print(index + end - 1)
            #print(index + end)
            train = df.drop(df.index[index:index + end])
            
            n = tr.ID3_depth(train, dep, 0)
            trains.append(tr.test(train, n, train['survived']))
            valids.append(train)
            print("fold=" + str(i + 1) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
            accuracies.append(tr.test(valid, n, valid['survived']))
            index += end
        valid = df.loc[index:df.shape[0]]
        train = df.drop(df.index[index:df.shape[0]])
        n = tr.ID3_depth(train,dep, 0 )
        trains.append(tr.test(train, n, train['survived']))
        accuracies.append(tr.test(valid, n, valid['survived']))
        valids.append(train)
        dat = valids[accuracies.index(max(accuracies))]

        print("fold=" + str(k) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
        n = tr.ID3_depth(dat, dep, 0)
        print("test_accuracy: " + str(tr.test(df2,n, df2['survived'] )))


        
        index = df.shape[0]/k + 1
    elif args.model == "minSplit":
        valids = []
        mini = args.mins
        mini = int(mini)
        tr = Tree(df)
        dep = args.dep
        tr.fill_median(df)
        accuracies = []
        trains = []
        index = 0
        k = args.k
        tr = Tree(df)
        tr.fill_mode(df)

        #tr.fill_mode(tem)
        #print(df.shape[0])
        for i in range(k - 1):
            valid = df.loc[index:index + int(df.shape[0]/k) - 1]
            end = int(df.shape[0]/k)
            #print(index)
            #print(index + end - 1)
            #print(index + end)
            train = df.drop(df.index[index:index + end])
            
            n = tr.min_split(train, mini)
            valids.append(train)
            trains.append(tr.test(train, n, train['survived']))
            print("fold=" + str(i + 1) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
            accuracies.append(tr.test(valid, n, valid['survived']))
            index += end
        valid = df.loc[index:df.shape[0]]
        train = df.drop(df.index[index:df.shape[0]])
        n = tr.min_split(train,mini)
        valids.append(train)
        trains.append(tr.test(train, n, train['survived']))
        accuracies.append(tr.test(valid, n, valid['survived']))
        dat = valids[accuracies.index(max(accuracies))]

        print("fold=" + str(k) + ", train_set_accuracy: " + str(tr.test(train, n, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, n, valid['survived'])))
        n = tr.min_split(dat, mini)
        print("test_accuracy: " + str(tr.test(df2,n, df2['survived'] )))
        
        index = df.shape[0]/k + 1
    elif args.model == "postPrune":
        valids = []
        accuracies = []
        trains = []
        index = 0
        k = args.k
        tr = Tree(df)
        tr.fill_mode(df)

        #tr.fill_mode(tem)
        #print(df.shape[0])
        for i in range(k - 1):
            valid = df.loc[index:index + int(df.shape[0]/k) - 1]
            end = int(df.shape[0]/k)
            #print(index)
            #print(index + end - 1)
            #print(index + end)
            train = df.drop(df.index[index:index + end])
            
            n = tr.ID3(train)
            valids.append(train)
            nod = tr.build(n, valid)
            tr.prune(nod)
            #trains.append(tr.test(train, n, train['survived']))
            print("fold=" + str(i + 1) + ", train_set_accuracy: " + str(tr.test(train, nod, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, nod, valid['survived'])))
            accuracies.append(tr.test(valid, n, valid['survived']))
            index += end
        valid = df.loc[index:df.shape[0]]
        train = df.drop(df.index[index:df.shape[0]])
        n = tr.ID3(train)
        nod = tr.build(n, valid)
        tr.prune(nod)
        #trains.append(tr.test(train, n, train['survived']))
        print("fold=" + str(k) + ", train_set_accuracy: " + str(tr.test(train, nod, train['survived'])) + ", validation_set_accuracy: " + str(tr.test(valid, nod, valid['survived'])))
        accuracies.append(tr.test(valid, n, valid['survived']))
        valids.append(train)
        dat = valids[accuracies.index(max(accuracies))]
        n = tr.ID3(dat)
        nod = tr.build(n, df2)
        tr.prune(nod)
        print("test_accuracy: " + str(tr.test(df2,n, df2['survived'] )))

        
        index = df.shape[0]/k + 1

        
