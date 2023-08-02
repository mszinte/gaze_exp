#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:34:59 2022

@author: uriel
"""

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import random 

#os.chdir('/Users/uriel/Documents/cours/Master/M1/stage/Data/EEG/EGG_ERD_alpha_beta')
#data = pd.read_excel('ERD_C3_C4_alpha_beta_décodage.xlsx') # créer un tableau data à partir de ton fichier en format xlsx (excel)

os.chdir('/Users/uriel/Desktop')
data = pd.read_excel('décodage_ERD_alpha_beta_µV_carré.xlsx') # créer un tableau data à partir de ton fichier en format xlsx (excel)



          ######## faire tourner le Knn en faisant une seule boucle 1000#####


### cuisine pour bien avoir les labels/features 
labels1 = np.array(data.tache_code)
features0 = [data.ERD,data.Electrode_code,data.Sujet_code,data.Rythmes_code]

features1 = np.array(features0)

labels = np.reshape(labels1, (-1,1))
features = features1.T

random_labels = shuffle(labels)
random_features =shuffle(features)


### trouver N voisin optimal   

### trouver N voisin optimal   
features_train0, features_test0, labels_train0, labels_test0 = train_test_split( 
              features, labels, test_size = 0.2, random_state=42) 

neighbors = np.arange(40, 50) 
  
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
  
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(features_train0, labels_train0) 
      
    
    train_accuracy[i] = knn.score(features_train0, labels_train0) 
    test_accuracy[i] = knn.score(features_test0, labels_test0) 
  
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 

#### déterminer graphiquement  en jouant sur les fenêtres les voisins optimum 



#### faire tourner le KNN avec le N voisin trouvé 

prediction_allrun=[]
labels_allrun=[]
for run in range(1000):
    
    features_train, features_test, labels_train, labels_test = train_test_split( features,labels, test_size=0.20, random_state=run)
    #features_train, features_test, labels_train, labels_test = train_test_split( random_features,random_labels, test_size=0.20, random_state=run)

    
    
    knn = KNeighborsClassifier(n_neighbors = 45).fit(features_train, labels_train)
    #accuracy = knn.score(features_train, labels_train)
    #print("Précision entrainement:", accuracy)
    
    pred_pop=knn.predict(features_test)
    
    # print(knn.predict(features_test))
    
    prediction_allrun.extend(pred_pop.tolist())    
    labels_allrun.extend(labels_test)    

cm = confusion_matrix(y_true=labels_allrun, y_pred=prediction_allrun,normalize ='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure()
disp.plot(cmap='OrRd')
plt.xticks([0,1,2],['CTRL','PIM','PP'])
plt.yticks([0,1,2],['CTRL','PIM','PP'])
plt.title('N voisin = 31 avec technique 1000 rep')




                    ######essais avec une validation croisé ####



data_global0 = [data.ERD,data.Electrode_code,data.Sujet_code,data.Rythmes_code,data.tache_code]
data_global_array = np.array(data_global0[:])


data_global = data_global_array.T

echantillon = np.array(random.choices(data_global,k=20) ) 
features2 = echantillon[:,0:4]
labels2 = echantillon[:,4]






### KNN
prediction_allrun2=[]
labels_allrun2=[]

for step1 in range (10):
    random.seed(step1)

    for step2 in range (50):
        echantillon = np.array(random.choices(data_global,k=20) ) 
        features3 = echantillon[:,0:4]
        labels3 = echantillon[:,4]
        
        
        
        for step3 in range(20): 
            features_train3, features_test3, labels_train3, labels_test3 = train_test_split( features3,labels3, test_size=0.05, random_state=step3)
        
        
        
            knn2 = KNeighborsClassifier(n_neighbors = 7).fit(features_train3, labels_train3)
            
            pred_pop2=knn2.predict(features_test3)
            
            
            prediction_allrun2.extend(pred_pop2.tolist())    
            labels_allrun2.extend(labels_test3)    




cm2 = confusion_matrix(y_true=labels_allrun2, y_pred=prediction_allrun2,normalize ='true')
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)

plt.figure()
disp2.plot(cmap='OrRd')
plt.xticks([0,1,2],['CTRL','PIM','PP'])
plt.yticks([0,1,2],['CTRL','PIM','PP'])
plt.title('N = 1')


###### trouver N voisin optimal  a postiorit 

features_train4, features_test4, labels_train4, labels_test4 = train_test_split( 
             features3, labels3, test_size = 0.2, random_state=42) 

neighbors = np.arange(1, 16) 
  
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
  
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(features_train4, labels_train4) 
      
    
    train_accuracy[i] = knn.score(features_train4, labels_train4) 
    test_accuracy[i] = knn.score(features_test4, labels_test4) 
  
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 


