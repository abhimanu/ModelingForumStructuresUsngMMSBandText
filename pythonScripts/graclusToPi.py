import sys

fileReader = open(sys.argv[1],'r');
fileWriter = open(sys.argv[2],'w');
K = int(sys.argv[3])

for line in fileReader:
	words = line.strip().split()
	fileWriter.write(words[0])
	for i in xrange(0,K):
		if i==int(words[1]):
			fileWriter.write(",1")
		else:
			fileWriter.write(",0")
	fileWriter.write("\n")

fileReader.close()
fileWriter.close()
