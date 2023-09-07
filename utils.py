import nltk
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
from nltk.stem.porter import *
import contractions
import re
import itertools
from itertools import chain, combinations
from nltk import everygrams
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import networkx as nx
import leidenalg
import igraph as ig
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict


def custom_n_grams(list_of_sentences:list, ngrams=1):
    """
    Calculates ngrams for a list of sentences
    
     list_of_sentences: A list of sentences as an input
     ngrams: The ngram parameter (default 1)
    """
    
    result = []
    
    for sentence in list_of_sentences:
        gram_pairs = list(everygrams(sentence,ngrams,ngrams))
        gram_pairs = [sorted(x) for x in gram_pairs]
        result.append(gram_pairs)
    
    result = list(itertools.chain(*result))
    ngrams = dict(Counter(map('_'.join,result)))
    
    return pd.DataFrame.from_dict(ngrams, 
                                  orient="index", 
                                  columns=["frequency"]).sort_values(by="frequency", 
                                                                     ascending=False)


def prepare_text(text:list, custom_stop_words = [], use_stemmer=False) -> list:
    """
    Processes a list of texts [big_text_a, big_text_b] and returns
    a list of words for each sentence in the texts.
    
    Example: [big_text_a, big_text_b] -> [[word_a, word_b], [word_a, word_b]]
    
     text: The texts to convert or prepare as a list
     custom_stop_words: Providing a list of custom stop words to filter out (default: empty)
     use_stemmer: Option to use a stemmer for cleaning words (lemmatization preferred)
    """
    
    # Merge all text elements together to one big chunk/corpus of text
    text_merged = ' '.join(text)
    
    # Uses sentence tokenizer to split the text into individual sentences
    sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    extracted_sentences = sentence_tokenizer.tokenize(text_merged)
    
    # Result to store prepared text or words in a list
    cleaned_sentences = []
    
    # Outer loop to process each sentence
    for extracted_sentence in extracted_sentences:
        
        # Fix to convert &-words: R&D -> RandD
        extracted_sentence = extracted_sentence.replace('&', 'and')
        
        # Split up each sentence into a list of words with Word_Tokenize 
        raw_words = nltk.tokenize.word_tokenize(extracted_sentence)
        
        # Temporarily store cleaned words of one sentence
        cleaned_words = []
        
        # Inner loop to process each word of a sentence
        for raw_word in raw_words:
            
            # Fix contractions: I'm -> I am
            clean_word = contractions.fix(raw_word)
            
            # Remove punctuation: Turtle. -> Turtle
            clean_word = re.sub(r'[^\w\s]', '', clean_word)
            
            # Lower case: Turtle -> turtle
            clean_word = clean_word.lower()
            
            # Stemming
            if use_stemmer:
                stemmer = PorterStemmer()
                clean_word = stemmer.stem(clean_word)
            
            # Lemmatizing
            lem = WordNetLemmatizer()
            clean_word = lem.lemmatize(clean_word)
            
            # Remove stop words
            useless_words = nltk.corpus.stopwords.words("english")
            useless_words = useless_words + custom_stop_words
            
            if clean_word in useless_words:
                continue
            
            # Do not consider the words with the following characteristics:
            # words with a length <=2 or empty strings "" or numbers
            if len(clean_word) <= 2 or clean_word == "" or clean_word.isnumeric():
                continue
            else:
                cleaned_words.append(clean_word)
        
        # Add clean words to one sentence
        cleaned_sentences.append(cleaned_words)
        
    # Return result and remove empty list elements
    return [ele for ele in cleaned_sentences if ele != []]


def create_context_matrix(sentences):
    """
    https://stackoverflow.com/questions/42814452/co-occurrence-matrix-from-nested-list-of-words
    Creates a context-matrix based on a nested list of sentences.
    It counts all the words that appear together in one sentence (context)
     
     sentences: A list of sentences [[cleaned_wordA, cleaned_wordB], [...]]
     
     Returns a unique list of words and the co-occurrence matrix as a DataFrame
    """
    
    # Define index/unique names
    words = set(list(itertools.chain.from_iterable(sentences)))
    # Build up a template
    occurrences = OrderedDict((word, OrderedDict((word, 0) for word in words)) for word in words)
    
    # Find the co-occurrences:
    for l in sentences:
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1
    
    co_occ_matrix_df = pd.DataFrame.from_dict(occurrences, orient="index")
    
    # Ensure that there a no self-references or self-loops
    np.fill_diagonal(co_occ_matrix_df.values, 0)

    return words, co_occ_matrix_df

def create_co_occurrence_matrix(sentences, window=2): # alternative method not applied
    """
    Creates a co-occurrence matrix based on a list of sentences for a set window.
    Example: "i like apples". 
     With a window of 2 there would be a connection between i<>like; like<>apples and i<>apples
     With a window of 1 there would be a connection between i<>like; like<>apples
    
     sentences: A list of sentences [[cleaned_wordA, cleaned_wordB], [...]]
     window: The number of words to consider (default 2)
    """
    # Step 1: Get unique list of words
    # Important: words have to be lowercase which should be done in cleaning
    unique_words = sorted(set(list(itertools.chain.from_iterable(sentences))))
    
    # Step 2: Build co-occurrence matrix
    np_co_occ = np.zeros((len(unique_words),
                          len(unique_words)))
    
    # Create proper template to be filled then
    matrix = pd.DataFrame(data=np_co_occ, 
                 index=unique_words, 
                 columns=unique_words)

    # Iterate over each sentence
    for sentence in sentences:
        # Iterate over each word
        for i in range(len(sentence)):
            for j in sentence[i:i+1+window]:
                pair = (sentence[i], j)
                matrix[pair[0]][pair[1]] += 1
                matrix[pair[1]][pair[0]] += 1

    # Avoid self-references 
    np.fill_diagonal(matrix.values, 0)
    
    return matrix



def create_nxgraph(co_occurrence_matrix: pd.DataFrame) -> nx.Graph:
    """
    Creates a networkx graph based on a co-occurrence DataFrame.
    Prints out the current graph attributes
    
     co_occurrence_matrix: co-occurrence DataFrame
    """
    # Create graph by using from_pandas_adjacency function
    G = nx.from_pandas_adjacency(co_occurrence_matrix)
    # Prints out the graphs attributes to get an idea of it's shape
    
    print(get_graph_attributes(G))
    return G


# Graph to gml
def graph_to_gml(graph:nx.Graph, file_name:str):
    """
    Saves the graph as a gml file in the graphs folder.
    
     graph: The graph to store.
     file_name: The name of the file
    """
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    nx.write_gml(graph, f"graphs/{file_name}_{dt_string}.gml")


def get_graph_attributes(graph:nx.Graph, simple=True):
    '''
        Generates a summary of relevant graph attributes.
         
         graph: The graph to be analyzed.
         simple: If simple is true (default) - Typical descriptive characteristics will be returned
    '''
    simple_output = {
        "Nodes": graph.number_of_nodes(),
        "Edges": graph.number_of_edges(),
        "Connected components":nx.number_connected_components(graph),
        "Density": round(nx.density(graph),3),
    }
    
    if simple:
        return simple_output
    else:
        simple_output.update({
            "Network diameter": round(nx.diameter(graph),3),
            "Avg shortest path length": round(nx.average_shortest_path_length(graph),3)
        })
        return simple_output



def calc_degree_betweenness_graph(graph:nx.Graph):
    """
    Returns a sorted DataFrame that contains the Degree and Betweenness 
    Centrality of a graph.
     
     graph: The graph to be analyzed.
    
    """
    
    # Get the (sorted) graph degree values as a dict
    degree_info = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    # Create a DataFrame 
    degree_info_df = pd.DataFrame(degree_info, columns=["node","degree"])
    
    # Attach the betweenness centrality as a column
    betweenness_info = sorted(nx.betweenness_centrality(graph).items(), key=lambda x: x[1], reverse=True)
    betweenness_info_df = pd.DataFrame(betweenness_info, columns=["node","betweenness"])
    
    return pd.merge(degree_info_df, betweenness_info_df, on="node", how="inner")


def identify_nodes_to_drop(node_characteristics:pd.DataFrame, filter_on='both', threshold=0.5): # alternative method not applied
    """
    Returns a list of nodes (words) to drop from the graph based on node
    degree and betweenness and a threshold.
    
     node_characteristics: DataFrame containing degree and betweenness values
     for each node.
     
     filter_on: Filter on degree and betweenness centrality or only on one of them
     
     threshold: A threshold when to cut-off (default 0.5 = median)
    """
    
    thresholds = node_characteristics[['degree', 'betweenness']].quantile(threshold).values
    degree_threshold = thresholds[0]
    betweenness_threshold = thresholds[1]
    
    if filter_on == 'both':
        nodes_to_drop = node_characteristics.loc[
            (node_characteristics.degree < degree_threshold) &
            (node_characteristics.betweenness < betweenness_threshold),
            "node"
        ].values
    elif filter_on == 'degree':
        nodes_to_drop = node_characteristics.loc[
            (node_characteristics.degree < degree_threshold),
            "node"
        ].values
    else:
        nodes_to_drop = node_characteristics.loc[
            (node_characteristics.betweenness < betweenness_threshold),
            "node"
        ].values              
    
    return nodes_to_drop


def identify_edges_to_drop(edge_characteristics:pd.DataFrame, threshold=0.5): # alternative method not applied
    """
    Returns a list of edges to drop from the graph based on edge betweenness centrality and
    a given threshold.
    
    edge_characteristics: Characteristics about the edges.
    threshold: Threshold when to cut-off (default 0.5 = median)
    """
    
    edge_betweenness_threshold = edge_characteristics[['edge_betweenness_centrality']].quantile(threshold).values
    edges_to_drop = edge_characteristics.loc[edge_characteristics.edge_betweenness_centrality < edge_betweenness_threshold[0]].index.to_list()
    
    return edges_to_drop


# The following functions are about community analysis
def create_community_graph(graph: nx.Graph):
    """
    Applies the leiden algorithm to identify communities within the graph.
    Returns a (new) graph with new node attributes containing the respective
    community and color.
    
     graph: The graph to apply the community detection on.
    """
    # Create a copy of the existing graph
    graph_new = graph.copy()
    
    # Since networkx by default does not support leiden algorithm
    # networkx graph has to be transformed into an igraph
    h = ig.Graph.from_networkx(graph_new)
    
    # Run leiden algorithm
    leiden_coms = leidenalg.find_partition(h,
                                       leidenalg.ModularityVertexPartition,
                                       n_iterations=-1,
                                       seed=42)
    
    # Create a mapping dataframe of all nodes and their related
    # communities
    mapping_node_cluster = pd.DataFrame({
        "node": leiden_coms.graph.vs()["_nx_name"],
        "community": leiden_coms.membership
    })
    
    # Add a color attribute to each node with respect to it's related
    # community.
    colors = [
        '#A8DADC',
        '#FFCDB2',
        '#e9c46a',
        '#cfe0c3',
        '#decbb7', 
        '#D7C3EB',
        '#F3BDCF', 
        '#C7D3E3',
        '#F0EFCF',
        '#F59606',
        '#B998C7',
        '#A48496',
        '#2792EF',
        '#499678'
    ]
    # Create a mapping between colors and number of communities
    # E.g, {0: '#005f73, 1: '#0a9396', 2: '#ee9b00'}
    map_com_to_colors = dict(zip(list(range(0, len(colors))), colors))
    
    # Add color to the dataframe
    mapping_node_cluster['community_color'] = mapping_node_cluster['community'].map(dict(map_com_to_colors))
    attr_content = mapping_node_cluster[["community", "community_color"]].to_dict(orient="records")
    
    # Create a mapping of node and related attributes (community and community_color)
    # E.g., {'word': {'community':0, 'community_color':'#005f73'}}
    node_attr_map = dict(zip(mapping_node_cluster['node'].tolist(),attr_content))
    nx.set_node_attributes(graph_new, node_attr_map)
    
    return graph_new


# Function to extract communities from graph and build summaries
def get_community_summary(graph: nx.Graph):
    """
    Get a statistical summary of all communities within a graph.
    
     graph: The graph containing communities
    """

    # Create a dataframe with node related community
    community_info = pd.DataFrame.from_dict((nx.get_node_attributes(graph, 'community')),
                                             orient="index",
                                             columns=["community"]
                                            ).reset_index(names="node")

    # Group by community and count how many nodes per community exist
    community_stats = community_info.groupby(["community"], 
                                             as_index=False).count().sort_values(by="node",
                                                                                 ascending=False).rename(columns={'node':'number_nodes'})

    # Calculate share
    community_stats['pct_of_all_nodes'] = round(community_stats['number_nodes'] / 
                                                community_stats['number_nodes'].sum(),3)*100
    
    return community_stats


def top_n_words_by_community(graph: nx.Graph, top_n=5):
    """
    Returns a DataFrame of the top N words (nodes) per community by
    using degree_centrality and betweenness_centrality.
    
     graph: The graph containing the communities
     top_n: The number of top n words to return per community (default 5)
    """
    
    # Get degree and betweenness centralities for each node in graph
    nodes_degree_centralities = nx.degree_centrality(graph)
    nodes_betweenness_centralities = nx.betweenness_centrality(graph)

    # Create a df with a node - community mapping
    community_info = pd.DataFrame.from_dict((nx.get_node_attributes(graph, 'community')),
                                                 orient="index",
                                                 columns=["community"]
                                                ).reset_index(names="node")
    
    # Add two new columns with degree centrality and betweenness centrality
    community_info["degree_centrality"] = community_info["node"].apply(lambda x: nodes_degree_centralities[x])
    community_info["betweenness_centrality"] = community_info["node"].apply(lambda x: nodes_betweenness_centralities[x])
    
    # Return dataframe
    return (community_info.sort_values(by=['community', 
                                           'degree_centrality', 
                                           'betweenness_centrality'],
                                       ascending=[True, False, False]).groupby('community').head(top_n))
    
    
def graph_per_community(graph: nx.Graph, community=0):
    """
     Based on a provided graph, returns a new graph filtered on one given community.
     
     graph: The graph containing the communities
     community: The community to extract (default 0 = the first one)
    """
    
    # Filter for nodes that are assigned the provided community
    nodes = [k for k,v in nx.get_node_attributes(graph, "community").items() if v==community]
    
    # Create subgraph based on filtered nodes
    sub_graph = graph.subgraph(nodes).copy()
    
    return sub_graph


def plot_community_graph(graph: nx.Graph, title=None, layout="spring", scale=1, dim=2, seed=42):
    """
    Plots a graph in a selected layout
    
    graph: The graph to plot
    title: Title for the graph (default None)
    layout: Default is spring layout - there are many others
     https://networkx.org/documentation/stable/reference/drawing.html
    scale: Default scale to plot graph - Increase to show more distance between nodes
    dim: Default is 2
    seed: Some layouts have random influence
    """
    
    # Normalize node and edge sizes for proper and better visualization
    # Size of node is shown by their degree centrality
    # Edge weights show how often the nodes appear together
    
    # Determine node sizes by using degree centrality
    degree_centrality_df = pd.DataFrame.from_dict(
        nx.degree_centrality(graph), 
        orient="index",
        columns=["degree_centrality"]
    )
    
    # Normalization for gephi to visualize
    max_old = degree_centrality_df['degree_centrality'].max()
    min_old = degree_centrality_df['degree_centrality'].min()

    max_new = 1000
    min_new = 300

    degree_centrality_df["norm_bc"] = degree_centrality_df["degree_centrality"].apply(lambda x: ((max_new - min_new) / (max_old - min_old)) * (x - max_old) + max_new
                                                                    )
    node_sizes = degree_centrality_df["norm_bc"].values
   

    # Determine edge weights
    edge_weights = {(w1,w2):a['weight'] for w1, w2, a in list(graph.edges(data=True))}
    edge_weights_df = pd.DataFrame.from_dict(edge_weights, orient="index", columns=["weight"])

    # normalization for visualization purposes
    max_old = edge_weights_df["weight"].max()
    min_old = edge_weights_df["weight"].min()

    min_new = 0.1
    max_new = 1

    edge_weights_df["norm_weight"] = edge_weights_df["weight"].apply(lambda x: ((max_new - min_new) / (max_old - min_old)) * (x - max_old) + max_new)
    edge_weights_final = edge_weights_df["norm_weight"].values

    # Choose plot layout
    # https://networkx.org/documentation/stable/reference/drawing.html
    if layout == "spring":
        pos = nx.spring_layout(graph, scale=scale, seed=seed, dim=dim)
    elif layout == "kawai":
        pos = nx.kamada_kawai_layout(graph, scale=scale, dim=dim)
    elif layout == "shell":
        pos = nx.shell_layout(graph, scale=scale, dim=dim)
    else: 
        pos = nx.random_layout(graph, seed=seed, dim=dim)

    # Define figure size
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    # Visualize graph components
    nx.draw_networkx_edges(graph, 
                           pos, 
                           alpha=0.3, 
                           width=edge_weights_final, 
                           edge_color="black")

    nx.draw_networkx_nodes(graph, 
                           pos,
                           node_size=node_sizes, 
                           node_color=np.array(list(nx.get_node_attributes(graph,'community_color').values())), 
                           alpha=0.9)

    nx.draw_networkx_labels(graph, 
                            pos, 
                            font_size=6
                           )
    
    plt.title(title)
