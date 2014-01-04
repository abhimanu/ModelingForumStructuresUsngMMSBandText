import sys
import numpy as np
import math

#ldaFile1 = open(sys.argv[1], 'r')
#ldaFile2 = open(sys.argv[2], 'r')
ldaFileString=sys.argv[1]
resultsCompare = open(sys.argv[2], 'w')
numTopics = int(sys.argv[3])
#numWords=int(sys.argv[3])



def getTopicSingle(threadStarterResultsFile1):
	topicWord1={}
	ldaFile1=open(threadStarterResultsFile1,'r')
	for topic in xrange(1,numTopics+1):
		words = ldaFile1.readline().strip().split(",")
		if topic!=int(words[0]):
			print "topic is not equal to word[0] in file1", topic, word[0]
			break
		vocab = words[1:len(words)]
		vocab = map(float, vocab) 
		vocabArray = np.array(vocab)
		topicWord1[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
	
	ldaFile1.close()
	return (topicWord1)


def getTopics(threadStarterResultsFile1, threadStarterResultsFile2):
	topicWord1={}
	topicWord2={}
	ldaFile1=open(threadStarterResultsFile1,'r')
	ldaFile2=open(threadStarterResultsFile2,'r')
	for topic in xrange(1,numTopics+1):
		words = ldaFile1.readline().strip().split(",")
		if topic!=int(words[0]):
			print "topic is not equal to word[0] in file1", topic, word[0]
			break
		vocab = words[1:len(words)]
		vocab = map(float, vocab) 
		vocabArray = np.array(vocab)
		topicWord1[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
	
		words = ldaFile2.readline().strip().split(",")
		if topic!=int(words[0]):
			print "topic is not equal to word[0] in file 2", topic, word[0]
			break
		vocab = words[1:len(words)]
		vocab = map(float, vocab)
		vocabArray = np.array(vocab)
		topicWord2[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
#		print len(vocabArray), vocabArray 
	ldaFile1.close()
	ldaFile2.close()
	return (topicWord1, topicWord2)

def getSimilarity(topicWord1, topicWord2):
	totalSimilarity=0
	for topic1 in xrange(1,numTopics+1):	#one less than actual numTopics
		vocab1=topicWord1[topic1]
		similarity=-1
	#	print vocab1
		closestTopic=-1
		for topic2 in xrange(1, numTopics+1):
			vocab2=topicWord2[topic2]
			similar = np.dot(vocab1, vocab2) 
			if similar>similarity:
				similarity=similar
				closestTopic=topic2
	#		print topic1, topic2, similar
		#resultsCompare.write(str(topic1)+" "+str(closestTopic)+" "+str(similarity)+"\n")
#		print str(topic1)+" "+str(closestTopic)+" "+str(similarity)
		totalSimilarity+=similarity
	return totalSimilarity

	
		

if __name__=="__main__":
#	for factorLoop in xrange(1,30):		# one less than the total
	threadStarterResultsFilePos=ldaFileString + "_factor30" + ".perTopicFullWordIndices.txt"
	threadStarterResultsFileNeg=ldaFileString + "_factor-2" + ".perTopicFullWordIndices.txt"
	ldaFilePos=open(threadStarterResultsFilePos,'r')
	ldaFileNeg=open(threadStarterResultsFileNeg,'r')
	topicWordPos = getTopicSingle(threadStarterResultsFilePos)
	topicWordNeg = getTopicSingle(threadStarterResultsFileNeg)
	for factorLoop in xrange(-3,30):		# one less than the total
		factor=factorLoop
#		nextFactor=factor+1
#		if factor==18 :
#			nextFactor=factor+2
#		if factor==19 :
#			factor=factor+1
#			nextFactor+=1
		threadStarterResultsFile1=ldaFileString + "_factor" + str(factor) + ".perTopicFullWordIndices.txt"
#		threadStarterResultsFile2=ldaFileString + "_factor" + str(nextFactor) + ".perTopicFullWordIndices.txt"
		try:
			ldaFile1=open(threadStarterResultsFile1,'r')
#			ldaFile2=open(threadStarterResultsFile2,'r')
#			(topicWord1, topicWord2) = getTopics(threadStarterResultsFile1, threadStarterResultsFile2)
			topicWord1 = getTopicSingle(threadStarterResultsFile1)
			totalSimilarityPos=getSimilarity(topicWord1, topicWordPos)
			totalSimilarityNeg=getSimilarity(topicWord1, topicWordNeg)
			resultsCompare.write(str(factor)+" "+str(totalSimilarityPos)+ " "+ str(totalSimilarityNeg)+"\n")
			print str(factor)+" "+str(totalSimilarityPos)+ " "+ str(totalSimilarityNeg)
			print "done with factor", factor
		except IOError:
			print "IOError"
			continue
	resultsCompare.close()
