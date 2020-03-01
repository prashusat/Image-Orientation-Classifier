#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 00:54:40 2019

@author: vanshsmacpro
"""



import numpy as np
import random
import sys
import pickle
import collections
from scipy.spatial  import distance

np.random.seed(15)

class nueral_net_classi:

    
    def __init__(self, no_class, no_featu, hiden_nod=30,
                 l1=0.0, l2=0.0, epochs=500, alpha=0.01,
                 no_batch=1):

       
        self.no_class = no_class
        self.no_featu = no_featu
        self.hiden_nod = hiden_nod
        self.weight1, self.weight2 = self.intitialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.alpha = alpha
        self.no_batch = no_batch
        
   
    def convert_labelto_cate(self, target, numer_oflabel):
        matrix = []
       
        for i in range(len(target)):
            f_matrix = []
            for j in range(numer_oflabel):
                f_matrix.append(0)
            matrix.append(f_matrix)
        
        for i in range(len(target)):
            for j in range(numer_oflabel):
                if target[i] == j :
                    matrix[i][j] = 1
        
        matrix = np.array(matrix)
        return matrix
    
    
    def second_regularization(self, lambdaaaa, weight1, weight2):
        b = np.sum(weight1 ** 2) + np.sum(weight2 ** 2)
        a = (lambdaaaa / 2.0) * b
        
        return a
    
    
    def first_regularization(self, lambdaaaa, weight1, weight2):
        b =  np.abs(weight1).sum() + np.abs(weight2).sum()
        a = (lambdaaaa / 2.0) * b
        return a
    
    
    def finding_error(self, out, target):
        x = -np.sum(np.log(out) * target, axis=1)
    
        return x
    
    
    
    def calculate_sigm(self, changing):
        x = 1/(1 + np.exp(-changing))
        return x
    
    
    
    
    def calculate_sigm2(self, z):
        sg = self.calculate_sigm(z)
        
        x = sg*(1-sg)
        return x
    
    

    def calculate_softmax(self, calc):
        a = np.sum(np.exp(calc), axis=1)
        a = a.T
        b = np.exp(calc.T)
        x = b/a
        return x
    

        

    def intitialize_weights(self):
        weight1 = np.random.uniform(0, 3,
                               size=(self.hiden_nod, self.no_featu))        
        weight2 = np.random.uniform(0, 3,
                               size=(self.no_class, self.hiden_nod))
        return weight1, weight2
      
    def feed_forward_propogation(self, inputs):
        input1 = np.array(inputs.copy(), dtype = "float64")
        hidden = self.weight1.dot(input1.T)
        out_hidden = self.calculate_sigm(hidden)
        before_out = self.weight2.dot(out_hidden)
        out = self.calculate_sigm(before_out)
        return input1, hidden, out_hidden, before_out, out
    
    def back_propogation(self, input1, hidden, out_hidden, out, change):
        x = out - change
        x1 = self.weight2.T.dot(x) * self.calculate_sigm2(hidden)
        first_gradient = x1.dot(input1)
        second_gradient = x.dot(out_hidden.T)
        return first_gradient, second_gradient      

    def calculate_error(self, target, out):
        t1 = self.first_regularization(self.l1, self.weight1, self.weight2)
        t2 = self.second_regularization(self.l2, self.weight1, self.weight2)
        error_now = self.finding_error(out, target) + t1 + t2
        return 0.5 * np.mean(error_now)

    def _backprop_step(self, train, target):
        input1, hidden, out_hidden, before_out, out = self.feed_forward_propogation(train)
        target = target.T

        first_gradient, second_gradient = self.back_propogation(input1, hidden, out_hidden, out, target)

        
        first_gradient = (self.weight1 * (self.l1 + self.l2)) + first_gradient
        second_gradient = (self.weight2 * (self.l1 + self.l2)) + second_gradient

        error = self.calculate_error(target, out)
        
        return error, first_gradient, second_gradient

   
    
    def predict_proba(self, train):
        x_copy = train.copy()
        input1, hidden, out_hidden, before_out, out = self.feed_forward_propogation(x_copy)
        return self.calculate_softmax(out.T)

    def fit(self, train, target):
        self.error_now = []
        train_data, target_data = train.copy(), target.copy()
        enoded_data = self.convert_labelto_cate(target_data, self.no_class)
        
                
        x_batch = np.array_split(train_data, self.no_batch)
        y_batch = np.array_split(enoded_data, self.no_batch)
        
        for i in range(self.epochs):
            
            error_final = []

            for k, l in zip(x_batch, y_batch):
                
                # update weights
                error, first_gradient, second_gradient = self._backprop_step(k, l)
                error_final.append(error)
                self.weight1 -= (self.alpha * first_gradient)
                self.weight2 -= (self.alpha * second_gradient)
            self.error_now.append(np.mean(error_final))
        return self
    


def scaling_data(train):
    
    temp = train - train.mean()

    temp = temp / temp.max()

    return temp

def train_nn(P_id , label , train_data):
   
        
    
    label = np.array(label, dtype = int).reshape(len(label),1)
    label = label//90
    train_data = np.array(train_data, dtype=int)

    train_dataset = np.concatenate((train_data, label), axis=1)
    random.shuffle(train_dataset)
    
    X_train, y_train = train_dataset[:,:-1], train_dataset[:,-1]
    X_train_scaled = scaling_data(X_train.astype(np.float64))
    
    nueral = nueral_net_classi(no_class=4, no_featu=192, hiden_nod=50, l2=0.5, l1=0.0, epochs=300,
                      alpha=0.001, no_batch=25).fit(X_train_scaled, y_train)
        
    return nueral

def test_nn(nueral, p_id , label , test_data):
   
   
    
    x_t = np.array(test_data)
    X_test_scaled = scaling_data(x_t.astype(np.float64))
    y_hat = nueral.predict_proba(X_test_scaled)
    pred = np.argmax(y_hat, axis=0)
    
    return (p_id, pred*90)

def read_data(fname):
    final_data = []
    file = open(fname, 'r');
    for line in file:
        train_data = tuple([ w for w in line.split(" ")])
        final_data += [(train_data[0], train_data[1], train_data[2:]) ] # Should make tuple for each line (image_id , pixel data, pixel data ... )

    
    return final_data

def solve_train(P_id , label , train_data):
    netWeight = train_nn(P_id , label , train_data)
    pickle.dump(netWeight, open("model_file.txt","wb"))
    
def solve_test(P_id , label , train_data):
    netWeight = pickle.load(open("model_file.txt", "rb"))
    orientation = test_nn(netWeight, P_id , label , train_data)
    return orientation

def preprocess(data):
    P_id = []
    label = []
    train_data = []
    for value in data:
        P_id.append(value[0])
        label.append(value[1])
        train_data.append(value[2])
    return P_id , label , train_data

##### 
    
def covert_to_matrix(file):
    thefile=open(file,"r")
    id_matrix=[]
    label_matrix=[]
    feature_matrix=[]
    
    for i in thefile:
        temp=i.split()
        id_matrix+=[temp[0]]
        label_matrix+=[temp[1]]
        feature_matrix+=[temp[2:]+[temp[1]]]
    
    return id_matrix,label_matrix,feature_matrix
def covert_to_matrix_test(file):
    thefile=open(file,"r")
    id_matrix=[]
    label_matrix=[]
    feature_matrix=[]
    
    for i in thefile:
        temp=i.split()
        id_matrix+=[temp[0]]
        label_matrix+=[temp[1]]
        feature_matrix+=[temp[2:]+[temp[1]]]
    
    return id_matrix,label_matrix,feature_matrix



def convert_string_to_int(data,col):
	for line in data:
		line[col] = float(line[col].strip())

def check_dict(obj):
    return isinstance(obj, dict)

def impurity_gini_fn(batches,check_batch):
	possibilities = float(sum([len(set_of_class) for set_of_class in batches]))
	impurity_gini = 0.0
	for set_of_class in batches:
		size_of_class = float(len(set_of_class))
		if size_of_class == 0:
			continue
		points = 0.0
		for class_val in check_batch:
			m = [i[-1] for i in set_of_class].count(class_val) / size_of_class
			points += m * m
		impurity_gini += (float(1) - points) * (size_of_class / possibilities)
	return impurity_gini

def divide_by_threshold(feature_number,threshold,feature_matrix):
	child_left=[]
	child_right=[]
	for line in feature_matrix:
		if line[feature_number] < threshold:
			child_left.append(line)
		else:
			child_right.append(line)
	return child_left,child_right
    
def split_data(dataset):
	class_values = list(set(row[-1] for row in dataset))
	i_b, v_b, s_b, g_b = 2000, 2000, 2000, None
	thresh=np.random.choice([i for i in range(0,255)],5)
	for index in range(len(dataset[0])-1):
		for ind in thresh:
			groups = divide_by_threshold(index, ind, dataset)
			gini = impurity_gini_fn(groups, class_values)
			if gini < s_b:
				i_b, v_b, s_b, g_b = index, ind, gini, groups
	return {'index':i_b, 'value':v_b, 'batches':g_b}

def select_best_class_value(batches):
        results=[]
        for line in batches:
            results+=[line[-1]]
        best=max(set(results), key=results.count)
        return best

def tree_splitter(node, max_depth, min_size, depth):
	child_left, child_right = node['batches']
	del(node['batches'])
	if (not child_left or not child_right):
		node['child_left'] = node['child_right'] = select_best_class_value(child_left + child_right)
		return
	if (depth >= max_depth):
		node['child_left'], node['child_right'] = select_best_class_value(child_left), select_best_class_value(child_right)
		return
	if (len(child_left) <= min_size):
		node['child_left'] = select_best_class_value(child_left)
	else:
		node['child_left'] = split_data(child_left)
		tree_splitter(node['child_left'], max_depth, min_size, depth+1)
	if (len(child_right) <= min_size):
		node['child_right'] = select_best_class_value(child_right)
	else:
		node['child_right'] = split_data(child_right)
		tree_splitter(node['child_right'], max_depth, min_size, depth+1)

def make_tree(train, max_depth, min_size):
	root = split_data(train)
	tree_splitter(root, max_depth, min_size, 1)
	return root



def predict(node,data_point):
	if data_point[node['index']] < node['value']:
		if check_dict(node['child_left']):
			return predict(node['child_left'], data_point)
		else:
			return node['child_left']
	else:
		if check_dict(node['child_right']):
			return predict(node['child_right'], data_point)
		else:
			return node['child_right']

    
def orient(k):
    id_matrix,label_matrix,feature_matrix=covert_to_matrix(k)
    test_id_matrix,test_label_matrix,test_feature_matrix= covert_to_matrix_test(k)
    for i in range(len(feature_matrix[0])):
            convert_string_to_int(feature_matrix, i)
    for i in range(len(test_feature_matrix[0])):
            convert_string_to_int(test_feature_matrix, i)
    tree=make_tree(feature_matrix,4,1)
    with open('tree_model_file.txt', 'wb') as handle:
        pickle.dump(tree,handle)

def test(k):
   
    id_matrix,label_matrix,feature_matrix=covert_to_matrix(k)
    test_id_matrix,test_label_matrix,test_feature_matrix= covert_to_matrix_test(k)
    for i in range(len(feature_matrix[0])):
            convert_string_to_int(feature_matrix, i)
    for i in range(len(test_feature_matrix[0])):
            convert_string_to_int(test_feature_matrix, i)
    m=0
    with open('tree_model_file.txt', 'rb') as handle:
          tree = pickle.loads(handle.read())
    for row in test_feature_matrix:
            pred=predict(tree,row)
            with open('output.txt', 'a') as the_file:
                a=""
                a+=str(test_id_matrix[m])+" "+str(pred)+"\n"
                the_file.write(a)
            m+=1
    m=0
    for row in test_feature_matrix:
           pred=predict(tree,row)
           if row[-1]==pred:
                m+=1
    print("accuracy",m/len(test_feature_matrix))
            
#The below code was used for cross-validation to choose the correct max_deth parameter
##def performance_with_max_depth():
##    depth_table=[1,2,3,4]
##    time_arr=[]
##    for i in depth_table:
##        print(i)
##        start_time=time.time()
##        tree = make_tree(feature_matrix,i, 1)
##        time_arr+=[time.time()-start_time]
##    
##        
##    plt.plot(depth_table,time_arr)
##    plt.title('Time vs Max_depth')
##    plt.xlabel('Max_depth')
##    plt.ylabel('Time')
##    plt.show()
##
##performance_with_max_depth()
##def performance_with_max_depth():
##    depth_table=[1,2,3,4]
##    accuracy_arr=[]
##    for i in depth_table:
##        tree = make_tree(feature_matrix,i, 1)
##        m=0
##        for row in test_feature_matrix:
##            pred=predict(tree,row)
##            if row[-1]==pred:
##                m+=1
##        accuracy_arr+=[m/len(test_feature_matrix)]
##        print("Accuracy for a depth of:",i,m/len(test_feature_matrix))
##    plt.plot(depth_table,accuracy_arr)
##    plt.title('Accuracy vs Max_depth')
##    plt.xlabel('Max_depth')
##    plt.ylabel('Accuracy')
##    plt.show()
##start_time=time.time()                      
##performance_with_max_depth()
##print("Time taken to check how performance of the model varies with max_depth parameter:",time.time()-start_time)
##
##def performance_with_data_size():
##    size=[5000,10000,15000,20000,25000,30000,35000]
##    accuracy_arr=[]
##    for i in size:
##        print(i)
##        ft=random.sample(feature_matrix,i)
##        tree = make_tree(ft,3, 1)
##        m=0
##        for row in test_feature_matrix:
##            pred=predict(tree,row)
##            if row[-1]==pred:
##                m+=1
##        accuracy_arr+=[m/len(test_feature_matrix)]
##        print("Accuracy for a size of:",i,m/len(test_feature_matrix))
##    plt.plot(size,accuracy_arr)
##    plt.title('Accuracy vs Max_depth')
##    plt.xlabel('Size')
##    plt.ylabel('Accuracy')
##    plt.show()
##start_time=time.time()                      
##performance_with_data_size()
##print("Time taken to check how performance of the model varies with size parameter:",time.time()-start_time)   

###
            
def convert_to_matrix(file):
    thefile = open(file, "r")
    id_matrix = []
    label_matrix = []
    feature_matrix = []

    for i in thefile:

        temp = i.split()
        id_matrix += [temp[0]]
        label_matrix += [temp[1]]
        feature_matrix += [temp[2:]]

    label_matrix=list(map(int,label_matrix))
    label_matrix=np.array(label_matrix)
    label_matrix=label_matrix.reshape(-1,1)
    #print(label_matrix.shape)

    final_feature_matrix=[]
    for i in feature_matrix:
        temp=(list(map(int,i)))
        final_feature_matrix.append((temp))
    final_feature_matrix=np.array(final_feature_matrix)
    with open('model_file.pkl', 'wb') as f:
        pickle.dump(final_feature_matrix, f)
    #print(final_feature_matrix.shape)
    with open('model_label.pkl', 'wb') as fi:
        pickle.dump(label_matrix, fi)

    return id_matrix, label_matrix, final_feature_matrix


def convert_to_matrix_test(file):
    thefile = open(file, "r")
    id_matrix = []
    label_matrix = []
    feature_matrix = []

    for i in thefile:
        temp = i.split()

        id_matrix += [temp[0]]
        label_matrix += [temp[1]]
        feature_matrix += [temp[2:]]
    label_matrix = list(map(int, label_matrix))
    label_matrix = np.array(label_matrix)
    label_matrix = label_matrix.reshape(-1, 1)
    #print(label_matrix.shape)

    final_feature_matrix = []
    for i in feature_matrix:
        temp = (list(map(int, i)))
        final_feature_matrix.append((temp))
    final_feature_matrix = np.array(final_feature_matrix)


    return id_matrix, label_matrix, final_feature_matrix


#id_matrix, label_matrix, feature_matrix = convert_to_matrix("train-data.txt")
#test_id_matrix, test_label_matrix, test_feature_matrix = convert_to_matrix_test("test-data.txt")

"""def slow_method(test_feature_matrix, feature_matrix):
    dist=[]
    final_dist=[]
    for i in test_feature_matrix:
        for j in feature_matrix:
            dist.append(np.sqrt((np.sum(i-j)**2)))

        final_dist.append(dist)
    final_dist=np.array(final_dist)
    print(final_dist.shape)
    return final_dist
final_dist=slow_method(test_feature_matrix, feature_matrix)"""

def prediction(test_feature_matrix,K):
    with open('model_file.pkl', 'rb') as f:
         feature_matrix = pickle.load(f)
    pre=distance.cdist(test_feature_matrix, feature_matrix, 'euclidean')

    index=np.argsort(pre)

    nearest_neighbour=[]
    for i in index:
        nearest_neighbour.append(i[:K])

    nearest_neighbour=np.array(nearest_neighbour)

 #   with open('model_file.pkl', 'wb') as f:
  #      pickle.dump(nearest_neighbour, f)

    return nearest_neighbour


def pickle_load(test_label_matrix,nearest_neighbour):
    with open('model_label.pkl', 'rb') as f:
         label_matrix = pickle.load(f)

    distances = []
    predicted_label = []
    for i in nearest_neighbour:
        get_label = []
        for ind in i:
            get_label.append(label_matrix[ind][0])

        cnt= (collections.Counter(get_label).most_common(1))

        predicted_label.append(cnt[0][0])
    predicted_label=np.array(predicted_label)

    predicted_label=predicted_label.reshape(-1,1)
    diff= np.subtract(predicted_label,test_label_matrix)
    diff=np.array(diff)

    non_zero=np.count_nonzero(diff)

    acc= ((diff.shape[0]-non_zero)/diff.shape[0])*100

    print(acc)


    return distances, predicted_label, acc


#nearest_neighbour=prediction(test_feature_matrix, feature_matrix,40)  #<------- call for training
#distances,predicted_label,acc=pickle_load(label_matrix,test_label_matrix)#<-------call for testing and accuracy

#test_id_matrix=np.array(test_id_matrix)
#test_id_matrix=test_id_matrix.reshape(-1,1)


"""for i in range(len(test_id_matrix)):
    with open("output.txt", 'a') as file:
        a=""

        a+= str(test_id_matrix[i][0])+" "+str(predicted_label[i][0])+"\n"

        file.write(a)"""

   
    
## main function : 



model_type = sys.argv[3]  
print(model_type)
test_or_train = sys.argv[1]


if model_type == "nnet" or model_type == "best" : 
    
    if test_or_train == "train" : 
        train_file = sys.argv[2]
        data = read_data(train_file)
        P_id , label , train_data =  preprocess(data)
        solve_train(P_id , label , train_data)
        
    
               
    elif test_or_train == "test" : 
        
        test_file = sys.argv[2]
        data = read_data(test_file)
        P_id , label , train_data =  preprocess(data)
        out = open("output_nn.txt", "w+")

        
        output = solve_test(P_id , label , train_data)
            
        for i in range(len(output[0])):   
            out.write(str(output[0][i]) + " " + str(output[1][i]) + '\n')
            
if model_type == "tree":
    
    if test_or_train == "train" :
        k = sys.argv[2]
        orient(k)
    elif test_or_train == "test":
      
        k = sys.argv[2]
        test(k)

if model_type == "nearest" : 
    
    if test_or_train == "train" :
        k = sys.argv[2]
        id_matrix, label_matrix, feature_matrix = convert_to_matrix(k)
    elif test_or_train == "test" : 
        te = sys.argv[2]
        test_id_matrix, test_label_matrix, test_feature_matrix = convert_to_matrix_test(te)
        nearest_neighbour = prediction(test_feature_matrix, 40)  # <------- call for training
        distances, predicted_label, acc = pickle_load(test_label_matrix,nearest_neighbour)  # <-------call for testing and accuracy
        test_id_matrix = np.array(test_id_matrix)
        test_id_matrix = test_id_matrix.reshape(-1, 1)
        for i in range(len(test_id_matrix)):
            with open("output_nearest.txt", 'a') as file:
                a = ""
    
                a += str(test_id_matrix[i][0]) + " " + str(predicted_label[i][0]) + "\n"
    
                file.write(a)


                
        
        
        
        
        
        
        
        
        
        
        
