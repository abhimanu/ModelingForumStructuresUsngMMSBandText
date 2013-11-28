import sys
import random

fileReader = open(sys.argv[1],'r');
fileWriter = open(sys.argv[2],'w');
K = int(sys.argv[3])
threshold = float(sys.argv[4])
uniqueCluster = int(sys.argv[5])
sparse = int(sys.argv[6]);
#alwaysAssign  = int(sys.argv[6])

lineNum=0
for line in fileReader:
	lineNum+=1
	words = line.strip().split(',')
#    print words
#	if lineNum>12:
#		break
	listInd = []
	for i in xrange(1,K+1):
		if float(words[i])>threshold:
			listInd.append(i)
	if not listInd:
		if sparse>0:
			continue
		fileWriter.write(words[0])
		ind = int(random.random()*K)
#		print ind, listInd, random.random()
		fileWriter.write(", "+str(ind+1))
	elif uniqueCluster >0:
		fileWriter.write(words[0])
		ind = int(random.random()*len(listInd))
#		print ind, listInd, random.random()
		fileWriter.write(", "+str(listInd[ind]))
	else:
		fileWriter.write(words[0])
		for i in listInd:
			fileWriter.write(", "+str(i))

	fileWriter.write("\n")

fileReader.close()
fileWriter.close()
