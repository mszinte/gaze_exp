#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:26:08 2022

@author: uriel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 00:51:44 2022

@author: uriel
"""

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

os.chdir('/Users/uriel/Desktop')

data = pd.read_excel('ERD_C3_C4_alpha_beta_décodage.xlsx') # créer un tableau data à partir de ton fichier en format xlsx (excel)


labels1 = np.array(data.tache_code)
features0 = [data.ERD,data.Electrode_code,data.Sujet_code,data.Rythmes_code]
features1 = np.array(features0)


labels = np.reshape(labels1, (-1,1)) ## cuisine pour bien avoir du 1D => grace message erreur 
features = features1.T



prediction_allrun=[]
labels_allrun=[]

for run in range(1000):
    
    features_train, features_test, labels_train, labels_test = train_test_split( features,labels, test_size=0.20, random_state=run)

    
    
    #knn = KNeighborsClassifier(n_neighbors = 4).fit(features_train, labels_train)
    #accuracy = knn.score(features_train, labels_train)
    #print("Précision entrainement:", accuracy)
    
    #pred_pop=knn.predict(features_test)
    
    # print(knn.predict(features_test))
    
    gnb = GaussianNB()

    pred_pop = gnb.fit(features_train, labels_train).predict(features_test)

    
    
    prediction_allrun.extend(pred_pop.tolist())    
    labels_allrun.extend(labels_test)    


cm = confusion_matrix(y_true=labels_allrun, y_pred=prediction_allrun,normalize ='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure()
disp.plot(cmap='OrRd')
plt.xticks([0,1,2],['CTRL','PIM','PP'])
plt.yticks([0,1,2],['CTRL','PIM','PP'])







                                         
