#dataset_path=$2
#N=$6
#M0=$7
#P0=${11}
#key=$1
#lambda=10
#mean=0.3
#sparse=0
#kl=0
#nnmf=0 
#rank=$8
#d=$9
#initialStep=$5
#debug=0
#min=$4
#last_iter=$(echo "scale=10; $d*$d-1" | bc -l)
#results_dir=${12}
#last_iter=$(echo "scale=10; $d-1" | bc -l)

#echo -e "Key: $key\nLambda: $lambda\nMean: $mean\nSparse: $sparse\nNNMF: $nnmf\nKL: $kl\nRank: $rank\nd: $d\ninitialStep: $initialStep\nminStep: $min\nLast iteration: $last_iter" > ${results_dir}/log-$key.txt

#hadoop fs -rmr /user/abeutel/DSGDout/$key/*
#rm ${results_dir}loss-$key.txt

#params=" -D dsgd.regularizerLambda=$lambda -D dsgd.initMean=$mean -D dsgd.nnmf=$nnmf -D dsgd.sparse=$sparse -D dsgd.KL=$kl -D mapred.child.java.opts=-Xmx2048m -D dsgd.N=$N -D dsgd.M0=$M0 -D dsgd.P0=$P0 -D dsgd.rank=$rank -D dsgd.debug=$debug -D mapred.reduce.tasks=$d $d 3 1 $key $dataset_path"
#output_dir="/user/abeutel/DSGDout/$key"

#echo -e "params: $params" >> ${results_dir}/log-$key.txt

#echo "Iteration 0"
#step=$initialStep
#time hadoop jar DSGD.jar DSGD -D dsgd.stepSize=$step $params ${output_dir}/run0 ${results_dir}

#last=0

#./ldaMmsb.out thread_starter_graph_text.txt 20 400 2 2 1e-9 0.5 0.1 1024 16 thread_starter_30Sep_ZeroEdges_NoInit_20_1-10.csv 44603 heldAndTestSet.txt 1e-10 -1 1.5 1.0 thread_starter_graph_text_graclus_unweighted.index.20 cancer_vocab.txt

input_file=$1
num_topics=$2
outer_iter=400
inner_iter=2
nu_iter=2
stepSizeNu=$3
stochastic_step_kappa=0.5
sampling_threshold=0.1
stochastic_step_tau=1024
numParallelThread=$4
vocabSize=$5
heldAndTestFile=$6
textFactorForNWTextBalance=$7
zeroEdges=$8
diagHyperInit=$9
nonDiagHyperInit=${10}
seedIndexFileName=${11}
vocabFile=${12}
alpha=${13}
dateTime=$(date +"%m-%d-%y")
outputFile=${input_file}-nuStep_${stepSizeNu}-topics_${num_topics}_${dateTime}-txtNWbal_$textFactorForNWTextBalance-zeroEdges_$zeroEdges-diag_$diagHyperInit-nonDiag_$nonDiagHyperInit.csv
dumpFile=${outputFile}.dump

echo -e "input_file:${input_file}\t num_topics: ${num_topics}\t outer_iter ${outer_iter}\t inner_iter ${inner_iter}\t nu_iter ${nu_iter}\t stepSizeNu $stepSizeNu\t stochastic_step_tau ${stochastic_step_kappa}\t sampling_threshold ${sampling_threshold}\t stochastic_step_kappa ${stochastic_step_tau}\t numParallelThread $numParallelThread\t outputFile $outputFile\t vocabSize $vocabSize\t heldAndTestFile $heldAndTestFile\t textFactorForNWTextBalance $textFactorForNWTextBalance\t zeroEdges $zeroEdges\t diagHyperInit $diagHyperInit\t nonDiagHyperInit $nonDiagHyperInit\t seedIndexFileName $seedIndexFileName\t vocabFile $vocabFile\t alpha $alpha"

./ldaMmsb.out ${input_file} ${num_topics} ${outer_iter} ${inner_iter} ${nu_iter} $stepSizeNu ${stochastic_step_kappa} ${sampling_threshold} ${stochastic_step_tau} $numParallelThread $outputFile $vocabSize $heldAndTestFile $textFactorForNWTextBalance $zeroEdges $diagHyperInit $nonDiagHyperInit $seedIndexFileName $vocabFile $alpha

#for i in {1..1}
#do
#	step=$(echo "scale=10; $initialStep / (($i + 1) * 0.5)" | bc -l)
#	step=$(echo $min $step | awk '{if ($1 < $2) print $2; else print $1}')
#
#	echo "Iteration ${i}"
#	echo "Step ${step}"
#	time hadoop jar DSGD.jar DSGD -D dsgd.stepSize=$step $params ${output_dir}/run$i ${output_dir}/run${last}/iter${last_iter} ${results_dir}
#
#	echo "Loss Iteration ${i}"
#	time hadoop jar Frobenius.jar Frobenius $params ${output_dir}-loss/run$i ${output_dir}/run${last}/iter${last_iter} ${results_dir}
#
#	last=$i
#done


#last=0
#for i in {0..14}
#do
#echo "Loss Iteration ${i}"
#time hadoop jar Loss.jar Loss $params ${output_dir}-loss/run$i ${output_dir}/run${last}/iter${last_iter}
#last=$i
#done
