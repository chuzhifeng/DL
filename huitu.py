from numpy import *
import matplotlib.pyplot as plt

dataMat = [];labelMat = []
fr = open('testSet.txt')
for line in fr.readlines():
	lineArr = line.strip().split()
	dataMat.append([float(lineArr[0]),float(lineArr[1])])
	labelMat.append(int(lineArr[2]))
dataArr = array(dataMat)
n = shape(dataArr)[0]
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
for i in range(n):
	if int(labelMat[i])== 1:
		xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
	else:
		xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
fig = plt.figure()	
ax = fig.add_subplot(211)
ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
ax.scatter(xcord2, ycord2, s=30, c='green', marker='x')

ax = fig.add_subplot(212)
x = arange(-10.0,10.0,0.1)
y = 1.0/(1+exp(-x))
ax.plot(x,y)
plt.show()
