# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

from numpy import *
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

#a = new_data.transpose()
#print(a)
#num = random.random()
#print(num)
#plt.scatter(a[0],a[1],c='b')
#plt.show()

def euclideanDistance(vector1,vector2):
    a = sum(power(vector2-vector1,2))
    return sqrt(a)

def closestDistance(vector1, list1):
    result = euclideanDistance(vector1,list1[0])
    index = 0
    for i in range(len(list1)):
        dist = euclideanDistance(vector1,list1[i])
        if(dist<result):
            result = dist
            index = i
    return result**2
        
    
def kmeans(filename,k,init='random'):
    if(filename=='wine.csv'):
        import numpy as np
        my_data = np.genfromtxt(filename,delimiter=',')
        new_data = preprocessing.minmax_scale(my_data,feature_range=(0,1))
        
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(new_data)
    elif(filename=='churn.csv'):#preprocess the churn dataset
        import numpy as np
        my_data = pd.read_csv(filename,names=["customerID","gender","SeniorCitizer","Partner","Dependents","tenure","PhoneService","MultiLines","InternetService","OnlineSecure","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"])
        #print(my_data.shape)
        my_data = my_data.drop(["customerID"],axis=1)
        my_data = my_data.drop(4709,axis = 0)
        #my_data = my_data.select_dtypes(include=['int'])
        #my_data = my_data[['gender','Partner','Dependents','PhoneService','PaperlessBilling']]#two-value attribute
        my_data1 = my_data['gender']
        my_data2 = my_data['Partner']
        my_data3 = my_data['Dependents']
        my_data4 = my_data['PhoneService']
        my_data5 = my_data['PaperlessBilling']
        my_data6 = my_data['MultiLines']
        my_data7 = my_data['InternetService']
        my_data8 = my_data['OnlineSecure']
        my_data9 = my_data['OnlineBackup']
        my_data10 = my_data['DeviceProtection']
        my_data11 = my_data['TechSupport']
        my_data12 = my_data['StreamingTV']
        my_data13 = my_data['StreamingMovies']
        my_data14 = my_data['Contract']
        my_data15 = my_data['PaymentMethod']
        my_data16 = my_data['SeniorCitizer'].astype(int)
        my_data17 = my_data['tenure'].astype(int)
        my_data18 = my_data['MonthlyCharges']
        my_data18 = pd.to_numeric(my_data18)
        #my_data.TotalCharges = my_data.TotalCharges.astype('float64')
        #my_data19 = my_data['TotalCharges']
        my_data19 = my_data['TotalCharges'].convert_objects(convert_numeric=True)
        #my_data19 = pd.to_numeric(my_data19)
        my_data16=np.array(my_data16)
        my_data17=np.array(my_data17)
        my_data18=np.array(my_data18)
        my_data19=np.array(my_data19)
        
        my_data = my_data.groupby('TotalCharges').TotalCharges.count()
        my_data16 = my_data16.reshape(-1,1)
        my_data17 = my_data17.reshape(-1,1)
        my_data18 = my_data18.reshape(-1,1)
        my_data19 = my_data19.reshape(-1,1)
        #print(my_data19)

        lb = preprocessing.LabelBinarizer()
        my_data1 = lb.fit_transform(my_data1)
        my_data2 = lb.fit_transform(my_data2)
        my_data3 = lb.fit_transform(my_data3)
        my_data4 = lb.fit_transform(my_data4)
        my_data5 = lb.fit_transform(my_data5)
        my_data6 = lb.fit_transform(my_data6)
        my_data7 = lb.fit_transform(my_data7)
        my_data8 = lb.fit_transform(my_data8)
        my_data9 = lb.fit_transform(my_data9)
        my_data10 = lb.fit_transform(my_data10)
        my_data11 = lb.fit_transform(my_data11)
        my_data12 = lb.fit_transform(my_data12)
        my_data13 = lb.fit_transform(my_data13)
        my_data14 = lb.fit_transform(my_data14)
        my_data15 = lb.fit_transform(my_data15)
        #print(my_data1)
        md = np.concatenate((my_data1,my_data2,my_data3,my_data4,my_data5,my_data6,my_data7,my_data8,my_data9,my_data10
                             ,my_data11,my_data12,my_data13,my_data14,my_data15,my_data16),axis=1)
        md = np.concatenate((md,my_data17,my_data16,my_data18),axis=1)
        
        new_data = preprocessing.minmax_scale(md,feature_range=(0,1))
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(new_data)
    #centers = random.sample(list(new_data),1)
    #print(type(centers))
    examples = new_data.shape[0]
    dim = new_data.shape[1] #get the data dimensions
    if(init == 'random'):
        centers = random.sample(list(new_data),k)
    #print(centers)
    #centers = [[random.uniform(-1,1) for j in range(dim)] for i in range(k)]#generate random centers
        centers = array(centers)
    elif(init=='k-means++'):
        centers = random.sample(list(new_data),1)
        for i in range(0,k-1):
            weights = [closestDistance(new_data[j],centers)
                        for j in range(len(new_data))]
            prob = random.random()
            total = 0
            x=-1
            while total<prob:
                x+=1
                total+=weights[x]
            centers.append(new_data[x])
            #centers = array(centers)  
        centers = array(centers)
        #print(centers)
    
    #print(centers.shape)
    clusterAssessment = mat(zeros((examples,2)))
    #list_index = zeros(dim)
    #list_error = zeros(dim)
    #total_sum = 0
    changed =True
    iterations = 0
    while changed:
        total_sum = 0
        iterations+=1
        changed = False
        for i in range(examples):
            minJ = 100000
            minIndex = 0
            for j in range(k):
                temp = euclideanDistance(new_data[i],centers[j])
                if(temp<minJ):
                    minJ = temp
                    minIndex = j
            total_sum += minJ
            if clusterAssessment[i,0] != minIndex:
                changed = True
                clusterAssessment[i, :] = minIndex, minJ**2
                
            
        for j in range(k):  
            pointsEachCluster = new_data[nonzero(clusterAssessment[:,0].A == j)[0]]
        
            centers[j,:] = mean(pointsEachCluster,axis=0)
    c = ['r','g','b','y','k','w','m']
    #for j in range(k):
        #plt.scatter(new_data[nonzero(clusterAssessment[:,0].A == j)[0]].transpose()[0],
                    #new_data[nonzero(clusterAssessment[:,0].A == j)[0]].transpose()[1],
                    #c = c[j])#plot the cluster graph
#plt.show()#show the graph
    #print('K-means Clustering Complete!')
    #print(new_data[nonzero(clusterAssessment[:,0].A == 0)[0]].shape)
    #print(new_data[nonzero(clusterAssessment[:,0].A == 1)[0]].shape)
    #print(new_data[nonzero(clusterAssessment[:,0].A == 2)[0]].shape)
    #print(filename[nonzero(clusterAssessment[:,0].A == 3)[0]].shape)
    #print(clusterAssessment[:,0])
    #print(iterations)
    #print(total_sum)
    #return clusterAssessment[:,0]
    if(filename=='wine.csv'):
        np.savetxt("wine_result.csv", clusterAssessment[:,0], delimiter=",")
    elif(filename=='churn.csv'):
        np.savetxt("churn_result.csv", clusterAssessment[:,0], delimiter=",")
    return clusterAssessment.transpose()[0]
            
def xor(a,b):
    if(a==b):
        return 0
    elif(a!=b):
        return 1
#def initCenters(filename,k):
    #dim = clean_data.shape[1]
    #return [[random.uniform(-1,1) for j in range(dim)] for i in range(k)]#generate random centers

#def kmeans(filename,k,init='random'):
    #if(init=='kmeans++'):
        
#a = initCenters(clean_data,5)
#print(closestDistance(new_data[2],centers))
if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        k = int(sys.argv[2])
        init = sys.argv[3]
        print(kmeans(filename, k, init))
#b = kmeans('churn.csv', 2,'random')

#print(b)


    
