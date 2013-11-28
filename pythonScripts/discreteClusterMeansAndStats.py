import sys
import math
import numpy as np;
import cPickle  as cp;


def readClusterFile(clusterFile):
	clusterDict = {}
	for line in clusterFile:
		words = line.strip().split(',')
#		print words
		if len(words)==1:
			print "cluster not assigned", words
		for c_i in xrange(1,len(words)):
			cluster = words[c_i].strip()
			if cluster in clusterDict:
				setOfNode=clusterDict.get(cluster)
				setOfNode.add(words[0].strip())
			else:
				setOfNode = set((words[0].strip()));
				clusterDict[cluster] = setOfNode
#			clusterDict.get(cluster, set([])).add(words[0].strip())
	return clusterDict


def readPiFile(piFile):
	piDict = {};
	for line in piFile:
		words = line.strip().split(',')
		user = words[0].strip()
		#print words
		piDict[user] = np.empty([len(words)-2,1], np.float32) # since the pi file ends in a comma
		for i in xrange(1,len(words)-1):
			#print i, piDict.get(user).shape
			topic_k = float(words[i].strip())
			piDict.get(user)[i-1] = topic_k

	return piDict

def writeClusterParams(clusterWriteFile,piDict,clusterDict,K):
	clusterMean = np.empty([K,1],np.float32)
	for k in xrange(0,K):
		clusterMean[k]=0
	for c_i in clusterDict:
		userSet=clusterDict.get(c_i)
		clusterSize = len(userSet)
		for user in userSet:
			userArray = np.array(piDict.get(user), dtype=np.float32)
			#print userArray.shape
			if not userArray.shape:
				clusterSize = clusterSize-1
				continue
			clusterMean = clusterMean+ userArray
			#print clusterMean#piDict.get(user)
		clusterMean = clusterMean/clusterSize
		clusterWriteFile.write(c_i+","+str(clusterSize))
		for index in xrange(0,K):
			clusterWriteFile.write(","+str(clusterMean[index]))
		clusterWriteFile.write("\n")

def getTestCentroids(testFile, clusterMeansFiles,K):
	clusterTopics = {}                                
	for line in clusterMeansFiles:
		words = line.strip().split(',')
		clusterId = words[0].strip()
		print "clusterId", clusterId
		clusterTopics[clusterId] = np.empty([K,1], dtype=np.float32)
		for ind in range(2, len(words)):
			word = words[ind]
			prob = float(word[2:len(word)-2])
			clusterTopics.get(clusterId)[ind-2]=prob
	testId=0;
	testTopics = {}
	for line in testFile:
		testTopics[testId] = np.empty([K,1], dtype=np.float32)
		words = line.strip().split(',')
		for ind in range(0, len(words)):
			testTopics.get(testId)[ind] = float(words[ind])
		userTopic = testTopics.get(testId)
		currC_i = -1,-1
		for c_i in clusterTopics:
			clusterTopic = clusterTopics.get(c_i)
			#print clusterTopic, userTopic
			similar = np.sum(np.multiply(clusterTopic, userTopic)/(np.linalg.norm(clusterTopic,'fro')*np.linalg.norm(userTopic,'fro')))
			if similar>currC_i[1]:
				currC_i=c_i,similar
		print testId, currC_i[0], clusterTopics.get(currC_i[0])
		testId+=1   

def getClusterUserCentroid(clusterCentroidFile, clusterMeansFiles, K, piDict):
	clusterTopics = {}                                
	for line in clusterMeansFiles:
		words = line.strip().split(',')
		clusterId = words[0].strip()
		print "clusterId", clusterId
		clusterTopics[clusterId] = np.empty([K,1], dtype=np.float32)
		for ind in range(2, len(words)):
			word = words[ind]
			prob = float(word[2:len(word)-2])
			clusterTopics.get(clusterId)[ind-2]=prob
		#print words
	currUser = {}
	for c_i in clusterTopics:
		currUser[c_i]= -1,-1
		clusterTopic = clusterTopics[c_i]
		for user in piDict:
			userTopic = piDict.get(user)
			if not userTopic.shape:
				continue
			tuple = currUser.get(c_i)
#			print (np.linalg.norm(clusterTopic.T,'fro')*np.linalg.norm(userTopic.T,'fro'))
	#		print clusterTopic, userTopic
			similar = np.sum(np.multiply(clusterTopic.T, userTopic.T)/(np.linalg.norm(clusterTopic.T,'fro')*np.linalg.norm(userTopic.T,'fro')))
	#		print similar, tuple
			if similar>tuple[1]:
				currUser[c_i]=user,similar
				print "user changed", user
		clusterCentroidFile.write(c_i+","+currUser[c_i][0])
		for k in xrange(0,K):
			clusterCentroidFile.write(","+str(piDict.get(currUser[c_i][0])[k]))
		clusterCentroidFile.write("\n")
		print "done with", c_i
	
	clusterCentroidFile.close()

def main():
	piFile = open(sys.argv[1],'r')
	K = int(sys.argv[2])
	piDict = readPiFile(piFile)

#	clusterReadFile = open(sys.argv[3],'r')
#	clusterMeansWriteFile = open(sys.argv[4],'w')
#	clusterDict = readClusterFile(clusterReadFile)
#	writeClusterParams(clusterMeansWriteFile, piDict, clusterDict, K)
#	clusterMeansWriteFile.flush()
#	clusterMeansWriteFile.close()

	clusterMeansFile = open(sys.argv[3],'r')
	clusterCentroidFile = open(sys.argv[4],'w')
	getClusterUserCentroid(clusterCentroidFile, clusterMeansFile, K, piDict)

#	testFile = open(sys.argv[4],'r')
#	getTestCentroids(testFile, clusterMeansFile, K)



if __name__ == "__main__":
	main()
