__author__ = 'jiachen'

import numpy as np
from numpy import *

def clusterNum():
    i= 0
    f = open("task2/data_no_replicate.txt",'r')
    clusterSet = {}
    for line in f.readlines():
        line = line.strip()
        lineList = line.split('	')
        cluNum = str(lineList[2])
        if cluNum in clusterSet:
            clusterSet[cluNum]+=1
        else:
            clusterSet[cluNum] = 1
        #print '\n'
    print clusterSet
    print '\n'
    print len(clusterSet)
    return clusterSet

def initDataSet():
    i= 0
    #f = open("/Users/jiachen/Desktop/data2-s.txt",'r')
    f = open("task2/data_no_replicate.txt",'r')
    dataSet = []
    for line in f.readlines():
        line = line.strip()
        lineList = line.split('	')
        #print lineList
        dataSet.append([float(lineList[0]),float(lineList[1])])
        i+=1
        #print '\n'
    #print dataSet
    #print '\n'
    #print len(dataSet)
    return dataSet

def initDataMark():
    i= 0
    f = open("task2/data_no_replicate.txt",'r')
    dataMark = []
    for line in f.readlines():
        line = line.strip()
        lineList = line.split('	')
        #print lineList
        dataMark.append(int(lineList[2]))
        i+=1
        #print '\n'
    #print dataSet
    #print '\n'
    #print len(dataSet)
    return dataMark

def initCentroids(dataSet,k):
    centroids = []
    n1 = random.choice(len(dataSet),1)
    p1 = dataSet[n1]
    centroids.append(p1)
    i = 1
    while i<k:
        p = findCentroid(centroids,dataSet)
        centroids.append(p)
        i+=1
    print "init centroids"
    print centroids
    return centroids
'''
def inputCentroids():
    f = open('task2/initCentroids.txt', 'r')
    centroids = []
    for line in f.readlines():
        line = line.strip()
        lineList = line.split(",")
        centroids.append([float(lineList[0]),float(lineList[1])])
    print centroids
    return centroids
'''
def findCentroid(cenList,dataSet):
    max_dist = 0
    for p in dataSet:
        dist = []
        for d in cenList:
            dist.append(distance(d,p))
        #print "juli"
        #print dist
        if(min(dist))>max_dist:
            max_dist = min(dist)
            pn = p
    return pn

def distance(p1,p2):
    dist = sqrt(power(p1[0]-p2[0],2)+power(p1[1]-p2[1],2))
    #print dist
    return dist

def kmeans(dataSet, k):
    dataSetList = np.asarray(dataSet)
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterChanged = True
    dataSetSize = len(dataSet)
    clusterMark = np.zeros((dataSetSize,3))
    ## step 1: init centroids
    centroids = np.asarray(initCentroids(dataSet, k))
    #centroids = np.asarray(inputCentroids())
    #print "init centroids"
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(dataSetSize):
            clusterMark[i,0] = i
            minDist  = 100000000.0
            minCen = -1
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                dist = distance(centroids[j], dataSet[i])
                if dist < minDist:
                    minDist  = dist
                    minCen = j

            ## step 3: update its cluster
            if clusterMark[i,1] != minCen:
                clusterChanged = True
                clusterMark[i,1] = minCen
                clusterMark[i,2] = minDist

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSetList[[num for (num,centro,dis) in clusterMark if centro == j]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)

    print 'Congratulations, cluster complete!'
    f = open('task2/clusterResult.txt', 'w')
    for i in range(dataSetSize):
        f.write(str(clusterMark[i,0])+","+str(clusterMark[i,1])+","+str(clusterMark[i,2]))
        f.write("\n")
    f.close()
    return centroids, clusterMark

def run():
    dataSet = initDataSet()
    dataMark = initDataMark()
    cen,clu = kmeans(dataSet,13)
    classCount,clusterTotal,purity = computePurity(dataMark,clu)
    FScore = computeFScore(classCount,clusterTotal)
    print "purity",purity
    print "F-score",FScore

def computePurity(dataMark,clusterMark):
    dataSize = len(dataMark)
    classCount = {}
    oldClusterCount = {}
    maxClassCount = {}
    clusterCount = {}
    purity = {}
    totalMaxCount = 0
    for i in range(dataSize):
        cluster = clusterMark[i,1]
        if cluster not in classCount:
            classCount[cluster] = {}
            classCount[cluster][dataMark[i]] = 1
        elif dataMark[i] not in classCount[cluster]:
            classCount[cluster][dataMark[i]] = 1
        else:
            classCount[cluster][dataMark[i]] += 1
        if dataMark[i] not in oldClusterCount:
            oldClusterCount[dataMark[i]] = 1
        else:
            oldClusterCount[dataMark[i]] += 1
        if cluster not in clusterCount:
            clusterCount[cluster] = 1
        else:
            clusterCount[cluster] += 1
    print classCount
    print "oldCluster",oldClusterCount
    print "clusterTotal",clusterCount
    for e in classCount:
        maxClass = max(classCount[e])
        maxCount = classCount[e][maxClass]
        maxClassCount[e] = maxCount
        purity[e] = float(maxCount)/float(clusterCount[e])
        totalMaxCount +=maxCount
    print "totalMaxCount",totalMaxCount
    print "dataSize",dataSize
    totalPurity = float(totalMaxCount)/float(dataSize)
    print "maxCount",maxClassCount
    print "purity",purity
    print "totalPurity",totalPurity
    return classCount,clusterCount,totalPurity

def computeFScore(classCount,clusterTotal):
    tp = 0
    tp_plus_fp = 0
    for e in classCount:
        for c in classCount[e]:
            if classCount[e][c]>1:
                tp+=classCount[e][c]*(classCount[e][c]-1)/2
    for e in clusterTotal:
        if clusterTotal[e]>1:
            tp_plus_fp+=clusterTotal[e]*(clusterTotal[e]-1)/2
    precision = float(tp)/float(tp_plus_fp)
    fn = computeTotalFN(classCount)
    recall = float(tp)/float(tp+fn)
    FScore = 2*recall*precision/(recall+precision)
    print "precision",precision
    print "recall",recall
    return FScore

def computeTotalFN(classCount):
    fn = 0
    for e in classCount:
        for c in classCount[e]:
            fn+=computeEachFN(classCount,e,c)
    return fn

def computeEachFN(classCount,curCluster,c):
    negN = 0
    for e in classCount:
        if int(e)>int(curCluster) and c in classCount[e]:
            negN +=classCount[e][c]
    result = classCount[curCluster][c]*negN
    return result

run()
