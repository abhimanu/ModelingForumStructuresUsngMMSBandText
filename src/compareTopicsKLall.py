import sys
import numpy as np
import math

#ldaFile1 = open(sys.argv[1], 'r')
#ldaFile2 = open(sys.argv[2], 'r')
#ldaFileString=sys.argv[1]
resultsCompare = open(sys.argv[2], 'w')
numTopics = int(sys.argv[1])
#numWords=int(sys.argv[3])

def getWordMap(wordmapFile):
	readFile = open(wordmapFile, 'r')
	numVocab=int(readFile.readline().strip())
	wordmap={}
	for counter in xrange(0, numVocab):
		words = readFile.readline().strip().split(" ")
		wordmap[words[0]] = int(words[1]) 
	readFile.close()
	return wordmap


def getTopicSingleR(threadStarterResultsFile1, wordmap):
	vocabLen=len(wordmap)
	topicWord1={}
	wordsDict={}
	ldaFile1=open(threadStarterResultsFile1,'r')
	for topic in xrange(0,numTopics+1):
		words = ldaFile1.readline().strip().split(",")
		vocab = words[0:len(words)]
		if topic<>0:
			vocab = map(float, vocab)
			topicWord1[topic] = np.ones((vocabLen,), dtype=float)/(10**306)
		wordsDict[topic] = vocab
	counter=0
	for word in wordsDict[0]:
		word=word.replace('"','')
		index=wordmap[word]
		for topic in xrange(1,numTopics+1):
			topicWord1[topic][index] += wordsDict[topic][counter]	#/math.sqrt(np.dot(vocabArray,vocabArray))
		counter+=1
	ldaFile1.close()
	for topic in xrange(1,numTopics+1):
		topicWord1[int(topic)] = topicWord1[int(topic)]/math.sqrt(np.dot(topicWord1[int(topic)],topicWord1[int(topic)]))
	return (topicWord1)

def getTopicSingleGibbs(threadStarterResultsFile1):
	topicWord1={}
	ldaFile1=open(threadStarterResultsFile1,'r')
	for topic in xrange(1,numTopics+1):
		words = ldaFile1.readline().strip().split(" ")
#		if topic!=int(words[0]):
#			print "topic is not equal to word[0] in file1", topic, word[0]
#			break
		vocab = words[0:len(words)]
		vocab = map(float, vocab) 
		vocabArray = np.array(vocab)
		topicWord1[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
	
	ldaFile1.close()
	return (topicWord1)

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
		topicWord1[int(topic)] = vocabArray#/math.sqrt(np.dot(vocabArray,vocabArray))
	
	ldaFile1.close()
	return (topicWord1)


#def getTopics(threadStarterResultsFile1, threadStarterResultsFile2):
#	topicWord1={}
#	topicWord2={}
#	ldaFile1=open(threadStarterResultsFile1,'r')
#	ldaFile2=open(threadStarterResultsFile2,'r')
#	for topic in xrange(1,numTopics+1):
#		words = ldaFile1.readline().strip().split(",")
#		if topic!=int(words[0]):
#			print "topic is not equal to word[0] in file1", topic, word[0]
#			break
#		vocab = words[1:len(words)]
#		vocab = map(float, vocab) 
#		vocabArray = np.array(vocab)
#		topicWord1[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
#	
#		words = ldaFile2.readline().strip().split(",")
#		if topic!=int(words[0]):
#			print "topic is not equal to word[0] in file 2", topic, word[0]
#			break
#		vocab = words[1:len(words)]
#		vocab = map(float, vocab)
#		vocabArray = np.array(vocab)
#		topicWord2[int(topic)] = vocabArray/math.sqrt(np.dot(vocabArray,vocabArray))
##		print len(vocabArray), vocabArray 
#	ldaFile1.close()
#	ldaFile2.close()
#	return (topicWord1, topicWord2)

def getSimilarity(topicWord1, topicWord2):
	totalSimilarity=0
	topicDict={}
	for topic1 in xrange(1,numTopics+1):	#one less than actual numTopics
		vocab1=topicWord1[topic1]
		similarity=-1
	#	print vocab1
		closestTopic=-1
		finalTopic2=-1
		for topic2 in xrange(1, numTopics+1) :
			vocab2=topicWord2[topic2]
#			print len(np.divide(vocab1, vocab2))
#			similarvec = np.multiply(vocab1, np.log(np.divide(vocab1, vocab2))) 
#			similarvec += np.multiply(vocab2, np.log(np.divide(vocab2, vocab1))) 
#			similarvec = similarvec/2
#			similar=np.sum(similarvec)
			similar = np.dot(vocab1, vocab2) 
			if similar>similarity and topic2 not in topicDict:
				similarity=similar
				closestTopic=topic2
				finalTopic2=topic2
		if finalTopic2<0:
			print "No match"
		topicDict[finalTopic2]=topic1
	#		print topic1, topic2, similar
		#resultsCompare.write(str(topic1)+" "+str(closestTopic)+" "+str(similarity)+"\n")
#		print str(topic1)+" "+str(closestTopic)+" "+str(similarity)
		totalSimilarity+=similarity
	print topicDict
	return totalSimilarity

	
		

if __name__=="__main__":
#	corpus1="topic3Phi.phi"
#	topicCorpus1=getTopicSingleGibbs(corpus1)
#	corpus2="topic3PermutedPhi.phi"
#	topicCorpus2=getTopicSingleGibbs(corpus2)
#	totalSimilarity=getSimilarity(topicCorpus1, topicCorpus2)
#	print totalSimilarity
#	for sub1 in xrange(1,3):
#		for rand1 in xrange(0,10):
#			corpus1="2000corpus_sub"+str(sub1)+"_rand"+str(rand1+1)+".phi"
#			topicCorpus1=getTopicSingleGibbs(corpus1)
#			for sub2 in xrange(1,3):
#				for rand2 in xrange(0,10):
#					corpus2="2000corpus_sub"+str(sub2)+"_rand"+str(rand2+1)+".phi"
#					topicCorpus2=getTopicSingleGibbs(corpus2)
#					totalSimilarity=getSimilarity(topicCorpus1, topicCorpus2)
#					resultsCompare.write(str(sub1)+str(rand1)+" "+str(sub2)+str(rand2)+" "+ str(totalSimilarity)+"\n")
#					print str(sub1)+str(rand1)+" "+str(sub2)+str(rand2)+" "+ str(totalSimilarity)
#					print "done with factor", str(sub1)+str(rand1)+" "+str(sub2)+str(rand2)+" "
	corpus1gibbs="2000corpus_sub"+str(1)+"_rand"+str(1)+".phi"
	corpus2gibbs="2000corpus_sub"+str(2)+"_rand"+str(1)+".phi"
	corpus1R="topicTermWeights_sub"+str(1)+"_rand"+str(1)+".txt"
	corpus2R="topicTermWeights_sub"+str(2)+"_rand"+str(1)+".txt"
	wordmapFile = sys.argv[3]
	wordmap = getWordMap(wordmapFile)
	topicCorpus = {}
	topicCorpus[1]=getTopicSingleGibbs(corpus1gibbs)
	topicCorpus[2]=getTopicSingleGibbs(corpus2gibbs)
	topicCorpus[3]=getTopicSingleR(corpus1R,wordmap)
	topicCorpus[4]=getTopicSingleR(corpus2R, wordmap)
	for index1 in xrange(1,5):
#		print topicCorpus[index1]
		for index2 in xrange(1,5):
			totalSimilarity=getSimilarity(topicCorpus[index1], topicCorpus[index2])
			resultsCompare.write(str(index1)+" "+str(index2)+" "+ str(totalSimilarity)+"\n")
			print str(index1)+" "+str(index2)+" "+ str(totalSimilarity)
	resultsCompare.close()
