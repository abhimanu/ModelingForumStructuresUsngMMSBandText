corpusFile = open('2000corpus','r')
thetaFile = open('topic3Theta.theta','r')

corpusFile1 = open('2000corpus_sub1','w')
corpusFile2 = open('2000corpus_sub2','w')
corpusFile3 = open('2000corpus_sub3','w')

lineCorpus = corpusFile.readline()	#this is to be done for reading off the num docs
for line in thetaFile:
	lineCorpus = corpusFile.readline()
	words=line.strip().split(" ")
	maxIndex = 0
	for index in xrange(0, len(words)):
		if float(words[index]) > float(words[maxIndex]):
			maxIndex = index
	if maxIndex == 0:
		corpusFile1.write(lineCorpus)
	elif maxIndex == 1:
		corpusFile2.write(lineCorpus)
	elif maxIndex == 2:
		corpusFile3.write(lineCorpus)

corpusFile.close()
corpusFile1.close()
corpusFile2.close()
corpusFile3.close()
thetaFile.close()

