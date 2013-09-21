import sys
import numpy as np;
import cPickle  as cp;

#readFile = open(sys.argv[1],'r')
#K = int(sys.argv[2])
#clusterFile = open(sys.argv[2],'r')

#clusterDict = {}
#edgeDict = {}

# pickle this
def readThreadStructureFile(readFile):
	edgeDict = {}
	nodeSet = set()
	for line in readFile:
		words = line.strip().split(' ')
		u1 = words[0].strip()
		u2 = words[1].strip()
		nodeSet.add(u1)
		nodeSet.add(u2)
#		global edgeDict
#		print u1, u2, edgeDict.get((u1,u2),0)
		edgeDict[(u1,u2)] = edgeDict.get((u1,u2), 0) +1
	return {'edgeDict':edgeDict, 'nodeSet':nodeSet}	

def readClusterFile(clusterFile):
	clusterDict = {}
	for line in clusterFile:
		words = line.strip().split(',')
#		print words
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
				
def getWithInAndAccrossParams(clusterDict, edgeDict, nodeSet):
	clusterParams = {}
#	print clusterDict
	for c in clusterDict:
		withinSum = 0
		accrosSum = 0
		withinCount = 0
		accrossCount = 0
		edgeClusterDict1 = {}
		edgeClusterDict2 = {}
		for u1 in clusterDict[c]:
			edgeClusterDict1[u1]= 0
			edgeClusterDict2[u1]= 0
		for u1 in clusterDict[c]:
			for u2 in nodeSet:
				if u2==u1:
					continue
				if u2 in clusterDict[c]:
					withinSum = withinSum + edgeDict.get((u1, u2),0)
					#print [0,1][bool(edgeDict.get((u1, u2),0)==0)], edgeDict.get((u1, u2),0) 
					withinCount = withinCount + [1,0][bool(edgeDict.get((u1, u2),0)==0)]
					edgeClusterDict1[u1] = edgeClusterDict1[u1] + edgeDict.get((u1, u2),0)
					edgeClusterDict2[u2] = edgeClusterDict2[u2] + edgeDict.get((u1, u2),0)
				else: 
					accrosSum = accrosSum + edgeDict.get((u1, u2),0)
					accrossCount = accrossCount + [1,0][bool(edgeDict.get((u1, u2),0)==0)]
			#print u1, edgeClusterDict1[u1]
		clusterParams[c] = (withinSum, accrosSum, len(clusterDict[c]), withinCount, accrossCount, edgeClusterDict1, edgeClusterDict2)
		print c#, clusterParams[c]
	return clusterParams


def main():
	readFile = open(sys.argv[1],'r')
	K = int(sys.argv[2])
	clusterReadFile = open(sys.argv[3],'r')
	clusterWriteFile = open(sys.argv[4],'w')
	results = readThreadStructureFile(readFile)
	edgeDict = results['edgeDict']
	nodeSet = results['nodeSet']
	clusterDict = readClusterFile(clusterReadFile)
	clusterParams = getWithInAndAccrossParams(clusterDict, edgeDict, nodeSet)
	print "I have the clusterParams\n"
	for c in clusterParams:
		dict1 = clusterParams[c][5]
		printResult1 = []
		result1=sorted(dict1, key=dict1.__getitem__, reverse=True)
		count = 0
		for i in result1:
			count = count +1
			if count>=min(10,len(dict1)):
				break
			printResult1.append((i,dict1[i]))
		clusterWriteFile.write(str(c) + " within: " + str(clusterParams[c][0]) + " accross: " + str(clusterParams[c][1]) + " size: " + str(clusterParams[c][2])+ " withinCount: " + str(clusterParams[c][3]) + " accrossCount: " + str(clusterParams[c][4])+"::")
		#clusterWriteFile.write(printResult1)
		for pair in printResult1:
			clusterWriteFile.write("("+str(pair[0])+":"+str(pair[1])+")")
		clusterWriteFile.write("\n");
		print c, "within: ", clusterParams[c][0], "accross: ", clusterParams[c][1], "size: ", clusterParams[c][2], "withinCount: ", clusterParams[c][3], "accrossCount: ", clusterParams[c][4],  " ", printResult1

if __name__ == "__main__":
	main()
