import numpy as np
import random
from sklearn import linear_model
import math
import threading
class stru:
    def __init__(self,a,b):
        self.feature = a
        self.label = b
class person:
    def __init__(self,a,b):
        self.weigth = a 
        self.acc = b

data = np.load('dataWithOutReduce.npy')
weigth = []
bestAcc = 0
bestWeigth = []
listBestWeigth = []
population = 4000
mutateRate = 0.1
avgAcc = 0
def check(per):
    global data,bestAcc,bestWeigth,avgAcc
    temp = []
    x_Select = []
    for i in range(512):
        per.weigth.append(random.choice([True, False]))
    for i in range(200):
        selectedFeature = []
        for j in range(512):
                 if(per.weigth[j] == True):
                     selectedFeature.append(data[i][j])
        x_Select.append(selectedFeature)
    #x_train = x_Select
    for i in range(100):
        temp.append(stru(x_Select[i],'0'))
    for i in range(100):
        temp.append(stru(x_Select[100+i],'1'))
        
    logreg = linear_model.LogisticRegression(C=1e5)
    #random.shuffle(temp)
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for i in range (200):
        if(i<120):
            x_train.append(temp[i].feature)
            y_train.append(temp[i].label)
        else:
            x_test.append(temp[i].feature)
            y_test.append(temp[i].label)
    
    logreg.fit(x_train, y_train)
    pred = logreg.predict(x_test)
    acc = 0
    for i in range(len(pred)):
        if(pred[i] == y_test[i]):
            acc = acc+1
    acc = (acc/len(pred))*100
    if(acc > bestAcc):
        bestAcc = acc
        bestWeigth = per.weigth
    per.acc = acc
    avgAcc = avgAcc + acc
def findSurvive(per):
    pool = []
    for i in range(len(per)):
        fitness = per[i].acc/bestAcc
        fitness = math.floor(fitness*100)
        for j in range (fitness):
            pool.append(per[i].weigth)
    return pool

def generate(pool):
    output = []
    for i in range (population):
        dad = math.floor(random.uniform(0,len(pool)))
        mom = math.floor(random.uniform(0,len(pool)))
        output.append(crossOver(pool[dad],pool[mom]))
    return output

def crossOver(dad,mom):
    DNA = []
    midPoint = math.floor(random.uniform(0,len(dad)))
    for i in range(len(dad)):
        if(i>midPoint):
            DNA.append(dad[i])
        else:
            DNA.append(mom[i])
    if(random.uniform(0,1)<mutateRate):
        DNA[math.floor(random.uniform(0,len(dad)))] = random.choice([True, False])
        return person(DNA,0)
    else:
        return person(DNA,0)
def start(name):
    global avgAcc
    rounded = 0 
    Gen = []
    for i in range(population):
        DNA = []
        for i in range(512):
            DNA.append(random.choice([True, False]))
        Gen.append(person(DNA,0))
    while(rounded < 100):
        for i in range(len(Gen)):
            check(Gen[i])
        Gen = findSurvive(Gen)
        Gen = generate(Gen)
        print('end Round ' + str(rounded) + 'best acc is ' + str(bestAcc) + 'threadName = ' +str(name)+' avgAcc = ' +str(avgAcc/population))
        listBestWeigth.append(bestWeigth)
        rounded = rounded + 1
        avgAcc = 0
start('a')
np.save('final',bestWeigth)

'''
t1 = threading.Thread(target = start , args = ('a'))
t2 = threading.Thread(target = start , args = ('b'))
t3 = threading.Thread(target = start , args = ('c'))

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()
'''