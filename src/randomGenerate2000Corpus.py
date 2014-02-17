import random
import sys

readFile = open(sys.argv[1],'r')
writeFile = open(sys.argv[1]+'_rand'+sys.argv[2],'w')

#readFile.readline() #this is for the firat line
lineNum=0
lineDict=[]
for line in readFile:
	if random.random()>0.25:
		lineNum+=1
        lineDict.append(line)

writeFile.write(str(lineNum)+"\n")
for line in lineDict:
	writeFile.write(line)

readFile.close()
writeFile.close()

