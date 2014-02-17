function [] = plotClusters()

% figure;
% % thetaU=load('SO_Thresh1_6Oct_combined_part.pidegree');
% thetaU=load('SO_Thresh1_6Oct_combined.pidegree');
% for i=1:4
%     subplot(2,4,i);
%     viz_stanford(thetaU,5*(i-1)+3,5*(i-1)+7);
%     title(i);
% end
% thetaU=load('SO_Thresh1_6Oct_Poisson_50percent.pidegree');
% for i=1:4
%     subplot(2,4,4+i);
%     viz_stanford(thetaU,5*(i-1)+3,5*(i-1)+7);
%     title(i);
% end
% 
% hold off;
% set(gcf,'Color',[1 1 1]);
% set(gcf, 'PaperPosition', [0 0 28 14]);
% set(gcf, 'PaperSize', [28 14]);
% print(gcf, '-dpng', '-r300', 'piSO.png');

% figure;
% thetaU=load('SO_Thresh1_6Oct_combined.pidegree');
% viz_stanford(thetaU,8,12);
% set(gcf,'Color',[1 1 1]);
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', 'pi2.pdf')
% 
% figure;
% thetaU=load('SO_Thresh1_6Oct_combined.pidegree');
% viz_stanford(thetaU,13,17);
% set(gcf,'Color',[1 1 1]);
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', 'pi3.pdf')
% 
% figure;
% thetaU=load('SO_Thresh1_6Oct_combined.pidegree');
% viz_stanford(thetaU,18,22);
% set(gcf,'Color',[1 1 1]);
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', 'pi4.pdf')

synth = load('syntheticDataset.csv');


% Log Likelihoods
figure;
h1=plot(synth(:,1), synth(:,2),'bo--', synth(:,1), synth(:,3),'r*-')
% xlabel('\alpha or \tau')
% ylabel('RMSE');
legend('RMSE-\alpha', 'RMSE-\eta');
hold off;
set(gcf, 'PaperPosition', [0 0 7 7]);
set(gcf, 'PaperSize', [7 7]);
print(gcf, '-dpdf', '-r300', 'synth.pdf')

% 
%
%
% % Log Likelihoods
% SO_ll = load('finalAnalysis/llPlots/SO_Thresh1_6Oct_ZeroEdges_Init_Combined_1e-10_20.csv.LL.txt');
% TS_ll = load('finalAnalysis/llPlots/thread_starter_graph_text.txt-numParallel_4-zeroEdges_-1-threads_100-10-06-13.csv.LL.txt');
% UM_ll = load('finalAnalysis/llPlots/username_6Oct_ZeroEdges_Init_Combined_1e-10_10.csv.LL.txt');
% 
% 
% % Time Comparison
% ParallelStochasticSubSample= load('finalAnalysis/llPlots/thread_starter_graph_text.txt-numParallel_4-zeroEdges_-1-threads_100-10-06-13.csv.LL.txt');
% StochaticSubSample = load('finalAnalysis/llPlots/thread_starter_graph_text.txt-numParallel_1-zeroEdges_-1-threads_400-10-06-13.csv.LL.txt');
% Stochastic = load('finalAnalysis/llPlots/thread_starter_graph_text.txt-numParallel_1-zeroEdges_-1-threads_1-10-06-13.csv.LL.txt');
% Variational = load('finalAnalysis/llPlots/thread_starter_graph_text.txt-numParallel_1-zeroEdges_-1-threads_14416-10-06-13.csv.LL.txt');
% 
% % Edge distribution bar-graph
% UM_edge = load('finalAnalysis/username_mention_graph.stats.edgebar');
% TS_edge = load('finalAnalysis/thread_starter_graph_text.stats.edgebar');
% SO_edge = load('finalAnalysis/s_o_word_ids.stats.edgebar');
% 
% 
%Local Topic Variations bar-graph
UM_var = load('finalAnalysis/username_6Oct.topicvar');
TS_var = load('finalAnalysis/thread_starter_4Oct_Combined.topicvar');
SO_var = load('finalAnalysis/SO_Thresh1_6Oct.topicvar');
% 
% % Plot similarity matrix
% TS_PO_5 = load('thread_starter_4Oct_Poisson_10_0.5_similarity.csv');
% TS_5 = load('thread_starter_1Oct_combined_0.5_similarity.csv');
% 
% 
% % Plot similarity matrix
% figure;
% % labels = ['1';'2';'3';'4';'5';'6';'7';'8';'9';'10';'11+'];
% h1 = subplot(1,2,1);
% spy(TS_PO_5);
% h2 = subplot(1,2,2);
% spy(TS_5);
% xlabel('');
% hold off;
% set(gcf, 'PaperPosition', [0 0 19.5 7.5]);
% set(gcf, 'PaperSize', [19.5 7.5]);
% print(gcf, '-dpng', '-r300', 'SimilarityMatTS.png')
% hold off;
%
%
%
%Local Topic Variations bar-graph
figure;
% labels = ['1';'2';'3';'4';'5';'6';'7';'8';'9';'10';'11+'];
h1 = subplot(1,2,1);
bar([1:9]*10,[UM_var,TS_var,SO_var]);
ylabel('Number of local topic variations')
xlabel('Percentage amount of variation')
legend('UM', 'TS', 'SO');
h2 = subplot(1,2,2);
bar([1:9]*10,[UM_var,TS_var,SO_var]);
ylim(h2, [0 4.2e6]);
ylabel('Number of local topic variations')
xlabel('Percentage amount of variation')
legend('UM', 'TS', 'SO');
hold off;
set(gcf, 'PaperPosition', [0 0 5 5]);
set(gcf, 'PaperSize', [5 5]);
print(gcf, '-dpdf', '-r300', 'TopicVariationsLocal.pdf')
hold off;
%
%
% % Edge distribution bar-graph
% figure;
% % labels = ['1';'2';'3';'4';'5';'6';'7';'8';'9';'10';'11+'];
% h1 = subplot(1,2,1);
% bar(1:11,[UM_edge(:,2),TS_edge(:,2),SO_edge(:,2)]);
% xlabel('Edge weights')
% ylabel('Number of edges')
% legend('UM', 'TS', 'SO');
% h2 = subplot(1,2,2);
% bar(1:11,[UM_edge(:,2),TS_edge(:,2),SO_edge(:,2)]);
% ylim(h2, [0 23e4]);
% % set(gca,'xlim',[0 12], 'xtick', 1:12, 'xticklabel', labels);
% % axes('xlim',[1 11],'xtick',1:11);
% % xt=cellstr(get(gca,'xticklabel'));
% % xt{11}='11+';
% % set(gca,'xticklabel',xt)
% xlabel('Edge weights')
% ylabel('Number of edges')
% legend('UM', 'TS', 'SO');
% hold off;
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', 'EdgeDistribution.pdf')
% hold off;
% 
% 
% % Log Likelihoods
% figure;
% h1=plot(SO_ll(:,2), -SO_ll(:,1),'go--', TS_ll(:,2), -TS_ll(:,1),'r*-', UM_ll(:,2), -UM_ll(:,1), 'b+-')
% xlabel('Time in minutes')
% ylabel('Negative Log Likelihood')
% legend('SO', 'TS', 'UM');
% hold off;
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', '3LLPlots.pdf')
% 
% 
% % Time Comparison
% figure;
% ha=subplot(1,2,1);
% plot(ParallelStochasticSubSample(:,2), -ParallelStochasticSubSample(:,1),'go--', ...
%     StochaticSubSample(:,2), -StochaticSubSample(:,1),'r.-', ...
%     Stochastic(:,2), -Stochastic(:,1),'b*-', Variational(:,2), -Variational(:,1), 'k+-')
% xlabel('Time in minutes')
% ylabel('Negative Log Likelihood')
% legend('PSSV', 'SSV', 'SV', 'V');
% hb=subplot(1,2,2);
% plot(ParallelStochasticSubSample(:,2), -ParallelStochasticSubSample(:,1),'go--', ...
%     StochaticSubSample(:,2), -StochaticSubSample(:,1),'r.-', ...
%     Stochastic(:,2), -Stochastic(:,1),'b*-', Variational(:,2), -Variational(:,1), 'k+-')
% ylim(hb,[0.45e5,0.8e5])
% xlabel('Time in minutes')
% ylabel('Negative Log Likelihood')
% legend('PSSV', 'SSV', 'SV', 'V');
% hold off;
% set(gcf, 'PaperPosition', [0 0 5 5]);
% set(gcf, 'PaperSize', [5 5]);
% print(gcf, '-dpdf', '-r300', 'SpeedOptimization.pdf')

end