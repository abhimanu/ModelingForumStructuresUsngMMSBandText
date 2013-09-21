import sys

fileReader = open(sys.argv[1],'r');
fileWriter = open(sys.argv[2],'w');
K = int(sys.argv[3])
threshold = float(sys.argv[4])

for line in fileReader:
	words = line.strip().split(',')
	fileWriter.write(words[0])
	for i in xrange(1,K+1):
		if float(words[i])>threshold:
			fileWriter.write(", "+str(i))
	fileWriter.write("\n")

fileReader.close()
fileWriter.close()
