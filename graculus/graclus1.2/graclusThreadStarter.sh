tail -22095 thread_starter_graph_text_graclus_unweighted.txt | cut -d' ' -f1 > thread_starter_graph_text_graclus_unweighted.graph.index

paste thread_starter_graph_text_graclus_unweighted.graph.temp thread_starter_graph_text_graclus_unweighted.graph.part.100 > thread_starter_graph_text_graclus_unweighted.index.100
