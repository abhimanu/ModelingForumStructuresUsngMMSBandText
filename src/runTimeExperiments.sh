input_file="thread_starter_graph_text.txt"
#numParallelThreads=4
heldAndTestFile="heldAndTestSet.txt"
#zeroEdges=-1
#threads=100
#
#echo "stochastic variational + sub-sampling + poisson property + parallel + 0 edges"
#time ./timeExperiments.sh ${input_file} ${numParallelThreads} ${heldAndTestFile} $zeroEdges $threads
#
#numParallelThreads=1
#zeroEdges=-1
#threads=400
#echo "stochastic variational + sub-sampling + poisson property + 0 edges"
#time ./timeExperiments.sh ${input_file} ${numParallelThreads} ${heldAndTestFile} $zeroEdges $threads

##numParallelThreads=1
##threads=400
##zeroEdges=-1
##echo "stochastic variational + sub-sampling "
##time ./timeExperiments.sh ${input_file} ${numParallelThreads} ${heldAndTestFile} $zeroEdges $threads

numParallelThreads=1
zeroEdges=-1
threads=1
echo "stochastic variational "
time ./timeExperiments.sh ${input_file} ${numParallelThreads} ${heldAndTestFile} $zeroEdges $threads

numParallelThreads=1
zeroEdges=-1
threads=14416
echo "variational "
time ./timeExperiments.sh ${input_file} ${numParallelThreads} ${heldAndTestFile} $zeroEdges $threads
