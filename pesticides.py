import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg

M_TRAIN = 800
M_TOTAL = 1000

def getF(conf_mat):
	precision = conf_mat[1][1]*1.0/(conf_mat[1][1] + conf_mat[0][1])
	recall = conf_mat[1][1] * 1.0/sum(conf_mat[1]) 
	f = (recall * precision)/ (recall+precision)
	return f


def getDict(trainData,training):

    counter = 0
    trainDict = {}
    for row in trainData:
	if counter == 0:
		counter = counter + 1
		continue

	tmpDict = {}
	tmpDict[row[0]] = []
	
	if training == 1:
		last_index = 10
	else:
		last_index = 9

	for i in range(1,last_index):
		if len(row[i]) == 0:
			tmpDict[row[0]].append(-1)
		else:
			tmpDict[row[0]].append(float(row[i]))

	trainDict.update(tmpDict)
	
	counter = counter + 1
	if counter == M_TOTAL:
		break

    print len(trainDict.keys())
    indexList = [0,3,4,5,6,7]

    for idx in indexList:
	tmpList = []
	for lid in trainDict.keys():
	   if trainDict[lid][idx] != -1:
		tmpList.append(trainDict[lid][idx])
	   else:
		tmpList.append(-1)

	maximum = max(tmpList)
	minimum = min(tmpList)

	data_range = (maximum - minimum)*1.0

	for i in range(len(tmpList)):
	    if tmpList[i] != -1:
		tmpList[i]= (tmpList[i] - minimum)/data_range

	for i in range(len(trainDict.keys())):
		lid = trainDict.keys()[i]
		trainDict[lid][idx] = tmpList[i]

    return trainDict



def train(trainDict):
	trainX = []
	trainY = []

	for lid in trainDict.keys():
		trainX.append(trainDict[lid][0:8])
		trainY.append(trainDict[lid][8])


	trainX = np.array(trainX)
	trainY = np.array(trainY)

	
	cList = [1]
	for i in range(50):
		cList.append(cList[-1]*2)

	gammaList = list(np.arange(0.01,0.1,0.01))
	gammaList.extend(np.arange(0.1,0.5,0.1))
	
	
	#cList = [90]
	#gammaList = [0.2]
	
	trainErrorList = []
	testErrorList = []
	mList = []

	for c in cList:
	  #for g in gammaList:
	    #for i in range(10,M_TRAIN):
		i = M_TRAIN
		#clf = SVC(C = c, gamma = g, degree = 3)
		g = 1
		clf = SVC(C = c, degree = 3)

		clf.fit(trainX[0:i],trainY[0:i])

		predictY = clf.predict(trainX[M_TRAIN:])

		conf_mat = confusion_matrix(trainY[M_TRAIN:], predictY, labels=[0,1])
		sens_0 = 1 - ((conf_mat[0][0] * 1.0)/sum(conf_mat[0]))
		sens_1 = 1 - ((conf_mat[1][1] * 1.0)/sum(conf_mat[1]))
	
		avgSens = (sens_0 + sens_1)/2.0
		f = getF(conf_mat)
		print c,'\t',g,'\t',avgSens

		testErrorList.append(avgSens)
		mList.append(i)

		predictY = clf.predict(trainX[0:i])

		conf_mat = confusion_matrix(trainY[0:i], predictY, labels=[0,1])

		sens_0 = 1 - ((conf_mat[0][0] * 1.0)/sum(conf_mat[0]))
		sens_1 = 1 - ((conf_mat[1][1] * 1.0)/sum(conf_mat[1]))

		avgSens = (sens_0 + sens_1)/2.0
		f = getF(conf_mat)
		trainErrorList.append(avgSens)

	win = pg.GraphicsWindow()
	pl1 = win.addPlot()
	pl1.plot(trainErrorList, pen = 'r')
	pl1.plot(testErrorList, pen = 'y')
	pl1.show()
	s = raw_input()
	print M_TRAIN,":",trainErrorList[-1], testErrorList[-1]



def test(testDict,trainDict):
	trainX = []
	trainY = []

	for lid in trainDict.keys():
		trainX.append(trainDict[lid][0:11])
		trainY.append(trainDict[lid][11])

	trainX = np.array(trainX)
	trainY = np.array(trainY)

	c = 90
	g = 0.2

	clf = SVC(C=c, gamma = g, degree =3)
	clf.fit(trainX,trainY)


	testX = []

	for lid in testDict.keys():
		testX.append(testDict[lid][0:11])

	testX = np.array(testX)

	predictY = clf.predict(testX)
	
	predictStringList = []
	for i in range(len(predictY)):
		if predictY[i] == 1:
			predictStringList.append('Y')
		else:
			predictStringList.append('N')

	firstRow = ['Loan_ID','Loan_Status']
	opFile = open("result.csv", "wb")
	csvWriter = csv.writer(opFile, delimiter=',')
	csvWriter.writerow(firstRow)
	
	print len(testDict.keys()), len(predictY)
	
	for i in range(len(testDict.keys())):
		writeString = [testDict.keys()[i],predictStringList[i]]
		csvWriter.writerow(writeString)



csvFile = open('Train.csv')
trainData = csv.reader(csvFile, delimiter = ',')
trainDict = getDict(trainData, 1)

train(trainDict)

csvFile = open('Test.csv')
testData = csv.reader(csvFile, delimiter = ',')
testDict = getDict(testData, 0)

#test(testDict, trainDict)
