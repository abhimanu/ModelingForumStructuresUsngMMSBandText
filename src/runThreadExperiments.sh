
echo -e "=================================================\n" >> threadStarterExperiments.dump
echo -e "=================================================\n" >> threadStarterExperiments.dump
numTopics=10
#alphaStart=$(echo "0.005" | bc -l)
#for alpha in 0.005 0.01 0.05 0.1
#for alpha in $(echo "$alphaStart" | bc -l) $(echo "$alphaStart * 2" | bc -l) $(echo "$alphaStart * 10" | bc -l) $(echo "$alphaStart *20" | bc -l) 
#do
#	for nonDiagHyperInit in 0.5 1.5 3 5 
alpha=$( echo "0.01" | bc -l )
#for nonDiagHyperInit in $(echo "$alphaStart * 100" | bc -l) $(echo "$alphaStart * 300" | bc -l) $(echo "$alphaStart * 600" | bc -l) $(echo "$alphaStart * 1000" | bc -l)
#	do
#		for diagHyperInit in $( echo "$nonDiagHyperInit + 1" | bc -l ) $( echo "$nonDiagHyperInit + 2" | bc -l  ) $( echo "$nonDiagHyperInit + 4" | bc -l )
#		do
#			echo $diagHyperInit $nonDiagHyperInit $alpha
#			for textFactorForNWTextBalance in 1e-15 1e-10 1e-5 1e-2 1
nonDiagHyperInit=1
diagHyperInit=$( echo "3.5" | bc -l )
for textFactorForNWTextBalance in $(echo "scale=15;1/1000000000000000" | bc -l) $(echo "scale=10;1/10000000000" | bc -l) $(echo "scale=5;1/100000" | bc -l) $(echo "scale=2;1/100" | bc -l)  $(echo "scale=2;0.0" | bc -l)  $(echo "scale=2;1" | bc -l)  $(echo "scale=2;10" | bc -l)
do
	echo $diagHyperInit $nonDiagHyperInit $alpha $textFactorForNWTextBalance
	time . runThreadStarter.sh thread_starter_graph_text.txt $numTopics 1e-9 16 44603 heldAndTestSet.txt $textFactorForNWTextBalance -1 $diagHyperInit $nonDiagHyperInit thread_starter_graph_text_graclus_unweighted.index.$numTopics cancer_vocab.txt $alpha >> threadStarterExperiments.dump
done
nonDiagHyperInit=$( echo "0.5" | bc -l )
diagHyperInit=$( echo "2.5" | bc -l )
for textFactorForNWTextBalance in $(echo "scale=15;1/1000000000000000" | bc -l) $(echo "scale=10;1/10000000000" | bc -l) $(echo "scale=5;1/100000" | bc -l) $(echo "scale=2;1/100" | bc -l)  $(echo "scale=2;0.0" | bc -l)  $(echo "scale=2;1" | bc -l)  $(echo "scale=2;10" | bc -l)
do
	echo $diagHyperInit $nonDiagHyperInit $alpha $textFactorForNWTextBalance
	time . runThreadStarter.sh thread_starter_graph_text.txt $numTopics 1e-9 16 44603 heldAndTestSet.txt $textFactorForNWTextBalance -1 $diagHyperInit $nonDiagHyperInit thread_starter_graph_text_graclus_unweighted.index.$numTopics cancer_vocab.txt $alpha >> threadStarterExperiments.dump
done
nonDiagHyperInit=$( echo "1.5" | bc -l )
diagHyperInit=$( echo "2.5" | bc -l )
for textFactorForNWTextBalance in $(echo "scale=15;1/1000000000000000" | bc -l) $(echo "scale=10;1/10000000000" | bc -l) $(echo "scale=5;1/100000" | bc -l) $(echo "scale=2;1/100" | bc -l)  $(echo "scale=2;0.0" | bc -l)  $(echo "scale=2;1" | bc -l)  $(echo "scale=2;10" | bc -l)
do
	echo $diagHyperInit $nonDiagHyperInit $alpha $textFactorForNWTextBalance
	time . runThreadStarter.sh thread_starter_graph_text.txt $numTopics 1e-9 16 44603 heldAndTestSet.txt $textFactorForNWTextBalance -1 $diagHyperInit $nonDiagHyperInit thread_starter_graph_text_graclus_unweighted.index.$numTopics cancer_vocab.txt $alpha >> threadStarterExperiments.dump
done
nonDiagHyperInit=$( echo "0.5" | bc -l )
diagHyperInit=$( echo "1.5" | bc -l )
for textFactorForNWTextBalance in $(echo "scale=15;1/1000000000000000" | bc -l) $(echo "scale=10;1/10000000000" | bc -l) $(echo "scale=5;1/100000" | bc -l) $(echo "scale=2;1/100" | bc -l)  $(echo "scale=2;0.0" | bc -l)  $(echo "scale=2;1" | bc -l)  $(echo "scale=2;10" | bc -l)
do
	echo $diagHyperInit $nonDiagHyperInit $alpha $textFactorForNWTextBalance
	time . runThreadStarter.sh thread_starter_graph_text.txt $numTopics 1e-9 16 44603 heldAndTestSet.txt $textFactorForNWTextBalance -1 $diagHyperInit $nonDiagHyperInit thread_starter_graph_text_graclus_unweighted.index.$numTopics cancer_vocab.txt $alpha >> threadStarterExperiments.dump
done
#		done
#	done
#done
