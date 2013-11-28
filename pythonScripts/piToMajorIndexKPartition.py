import sys
import random

fileReader = open(sys.argv[1],'r');

fileWriter1 = open(sys.argv[2]+'1','w');
fileWriter2 = open(sys.argv[2]+'2','w');
fileWriter3 = open(sys.argv[2]+'3','w');
fileWriter4 = open(sys.argv[2]+'4','w');

K = int(sys.argv[3])
#alwaysAssign  = int(sys.argv[6])

lineNum=0
for line in fileReader:
	lineNum+=1
	words = line.strip().split(',')
#    print words
#	if lineNum>12:
#		break
	topic=-1
	for i in xrange(1,K+1):
		if float(words[i])>0.5:
			topic = i-1
	if topic<0:
		continue
	if topic<5:
		fileWriter1.write(line)
	elif topic<10:
		fileWriter2.write(line)
	elif topic<15:
		fileWriter3.write(line)
	elif topic<20:
		fileWriter4.write(line)

fileReader.close()
fileWriter1.close()
fileWriter2.close()
fileWriter3.close()
fileWriter4.close()
