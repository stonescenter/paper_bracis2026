import pandas as pd
import numpy as np

import multiprocessing
import smart_open
import gensim
from gensim.models import Word2Vec

import networkx as nx
import time
import datetime
#from datetime import datetime
import os.path

from walks.explorer import TemporalRandomWalk, UniformRandomWalk, TSAW
from common.classifier import *
from process.data import load_graph, load_graph_from_nx, create_dataset, create_dataset2
from logger.logging import *

import gc

embedding_size = 128
embeddings = {}
unseen_node_embedding = np.zeros(embedding_size)


PATH_DATASET = '/exp-local/steve/datasets/temporal/cleaned/'

def get_embedding_value(u):
  global embeddings
  global unseen_node_embedding

  try:
      return embeddings[u]
  except KeyError:
      return unseen_node_embedding
  
def save_file(name, data):

  i = 1
  with open(name, 'w') as fo:
    for row in data:
      source = int(row[0])
      target = int(row[1])

      fo.write('{}\t {}\t {}\n'.format(source, target, i))
      i = i + 1
    fo.close()

def save_embedding(dict_embedding, size_emb, func_get_embedding, label):

    file_name = str(size_emb) + '_embeddidng_' + label + '.txt'

    with open(file_name, 'w') as outfile:
      for node, e in dict_embedding.items():

        embedding = " ".join(str(x) for x in func_get_embedding(dict_embedding, node))
        outfile.write('{} {}\n'.format(node, embedding))

    print(f"Embedding {file_name}, nro embeddings {len(dict_embedding)} saved!")
    outfile.close()

def build_all_results(datasets, type_walks, clfs, binary_operators, params):
 
  embedding_size = params['embedding_size']
  embedding_model = params['embedding_model']
  num_walks_per_node = params['num_walks_per_node']
  walk_length = params['walk_length']
  context_window_size = params['context_window_size']

  train_subset = params['train_subset']
  test_subset = params['test_subset']
  sep = params['sep']
  save_walks = params['save_walks']
  path_save_walks = params['path_save_walks']
  re_use_walks = params['re_use_walks']

  workers = int(multiprocessing.cpu_count()*float(params['workers']))
  
  for name, shortname in datasets.items():
    path_file = PATH_DATASET + name

    print("")
    print(f"Using file: {path_file}")
    print(f"Data base: {shortname}")

    #graph, all_edges, edges_graph, edges_train, edges_test = create_dataset(path_file, sep,
    #    train_subset=train_subset, test_subset=test_subset)
    graph, all_edges, edges_graph, edges_newer = create_dataset(path_file, sep,
        train_subset=train_subset, test_subset=test_subset)
    
    #print(graph.info())
    print(graph.info())
    print(f"Total Graph nodes : {len(graph.nodes())}, all edges in graph: {len(graph.edges())}")
    print(f"Total Edges       : {len(all_edges)}")
    #print(f"Total Edges Train : {len(edges_train)}")
    #print(f"Total Edges Test  : {len(edges_test)}")
    #print(f"Total             : {len(edges_train)+len(edges_test)}")

    num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1)
    #link_examples, link_labels, link_examples_test, link_labels_test = create_positive_negative_data(
    #  graph, edges_train, edges_test)
   

    print(f"\nProcessing walks: num_cw: {num_cw} , context: {context_window_size}")
    
    p, q = None, None
    seconds = 0
    for type_walk in type_walks:
      walks = []
      begin = time.time()
      
      fname = "walks_"+shortname+"_"+ params['model']['id'] + "_" +type_walk+".npy"
      fname = path_save_walks + fname

      if re_use_walks:
        if os.path.exists(fname):
          print(f"Loading file {fname}")
          walks = np.load(fname, allow_pickle=True).tolist()
          print(f"Succesfully loaded type {type(walks)}, name : {fname} ")
        else:
          print(f"Error loading walks, path file {fname} doesn't exist")
      else:
        if type_walk == 'tsaw':
          # treinar so no 70% do grafos
          edges_graph[['source', 'target']] = edges_graph[['source', 'target']].astype(str)
          g = nx.from_pandas_edgelist(edges_graph, "source", "target")

          random_walks = TSAW(g)

          #walks = random_walks.run_walk(type_walk="simple_rw", num_walks=num_walks_per_node, walk_length=walk_length, repeted=False)
          walks = random_walks.run_walk(type_walk="tsaw", num_walks=num_walks_per_node, walk_length=walk_length, repeted=False)

        elif type_walk == 'deepwalk':

          static_rw = UniformRandomWalk(graph)
          walks = static_rw.run_optimized(
              #nodes=graph.nodes(), 
              num_walks=num_walks_per_node, 
              walk_length=walk_length,
              workers=workers
          )
        elif type_walk == 'temporal':

          temporal_rw = TemporalRandomWalk(graph)
          lamb=params['temporal']['lambda']
          '''
          walks = temporal_rw.run(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="temporal",
              is_forward=True,
              lamb=lamb,
              debug=False
          )
          '''
          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="temporal",
              is_forward=True, ## se vai ascedente ou descendente
              lamb=1,
              beta = 0.8, 
              model = params['model'],
              workers = workers,
              parallel_exploration='nodes'
          )
        elif type_walk == 'spatial' or type_walk == 'spatial_hub' :

          temporal_rw = TemporalRandomWalk(graph)
          #beta=params[type_walk]['beta']
          beta=1
          
          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias=type_walk,
              beta = beta,
              model = params['model'],
              is_forward=True,
              debug=False,
              workers=workers
          )

        elif type_walk == 'spatial_normalization':

          temporal_rw = TemporalRandomWalk(graph)
          beta = params[type_walk]['beta']
          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="spatial_normalization",
              is_forward=True,
              beta = beta,
              model = params['model'],            
              debug=False,
              workers=workers
          )

        elif type_walk == 'spatial_temporal':

          temporal_rw = TemporalRandomWalk(graph)
          lamb=params['spatial_temporal']['lambda']
          beta=params['spatial_temporal']['beta']

          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="spatial_temporal",
              is_forward=True,
              lamb=lamb,    
              beta=beta,
              model = params['model'],            
              debug=False,
              workers=workers
          )
        
        elif type_walk == 'spatial_temporal_normalization':

          temporal_rw = TemporalRandomWalk(graph)
          lamb, beta = params[type_walk]['lambda'], params[type_walk]['beta']
          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="spatial_temporal_normalization",
              is_forward=True,
              lamb=lamb,    
              beta = beta,
              model = params['model'],            
              debug=False,
              workers=workers,
              parallel_exploration='nodes'
          )
        elif type_walk == 'spatial_temporal_normalization_mult':

          temporal_rw = TemporalRandomWalk(graph)
          lamb, beta = params[type_walk]['lambda'], params[type_walk]['beta']
          walks = temporal_rw.run_walk(
              num_cw=num_cw,
              cw_size=context_window_size,
              max_walk_length=walk_length,
              walk_bias="spatial_temporal_normalization_mult",
              is_forward=True,
              lamb=lamb,    
              beta = beta,
              model = params['model'],            
              debug=False,
              workers=workers
          )
      
        end = time.time()
        seconds = end - begin  

        if save_walks:
          walks_array = np.asarray(walks, dtype=object)
          np.save(fname, walks_array)
          del walks_array
      
      print("\n")
      print(f"Type walk: {type_walk}, date: {datetime.datetime.now()}")
      print(f"\tModel id: {params['model']['id']}, desc: {params['model']['desc']}, alpha : {params['model']['alpha']}")
      print(f"\tWalk_length: {walk_length}, num walks per node: {num_walks_per_node}, Nro walks: {len(walks)}, Context: {context_window_size}, Total Time: {str(datetime.timedelta(seconds=seconds))}")
      print(f"\tWalk param: {params[type_walk]}, ")

      print(f"Training Embedding generation process ...") 
                  
      begin = time.time() 

      model = Word2Vec(
          walks, # 75% do dataset
          vector_size=embedding_size,
          window=context_window_size,
          min_count=0,
          sg=embedding_model,
          workers=4,
          #iter=1,
          #epochs=1,
          sample=1e-3,
          negative=3,
          hs=0
      )      
      end = time.time()
      seconds = end - begin
       
      del walks

      global embeddings
      embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
      print(f"\t {len(embeddings)} embedding for {type_walk}, Total Time:{str(datetime.timedelta(seconds=seconds))}")

      print(f"Calculating  AUC metrics...")

      results = [run_link_prediction(op, clf, get_embedding_value, graph, edges_newer, test_subset, n_runs=5) for clf in clfs for op in binary_operators ]
      embeddings.clear()

      results = [
        {"Classifier": r['Classifier'],
        "Binary_operator": r['Binary-operator'].__name__,
        "ROC-AUC": r['ROC-AUC'],
        "ROC-AUC-scores": r['ROC-AUC-scores'],
        "AP": r['AP'],
        "AP-scores": r['AP-scores']} for r in results]
           
      for r in results:
        for k, v in r.items():
          print(f"\t{k}: {v}")  
      
      print("-------------------------------")

      del model

if __name__ == "__main__":

  #log_file = 'results.log'
  #setup_logging(log_file)

  params = {
    'embedding_size': 128,
    'embedding_model': 1, # Skipgram
    'num_walks_per_node': 32,
    'walk_length': 80, # 16,32
    'context_window_size': 10,
    'workers': 0.50, # %
    'train_subset': 0.70, # train for learning representation with embeddings
    'test_subset': 0.20,  # 20% para testing e a diferença 80% para treino
    'path_save_walks': "results/walks/",

    'save_walks': True,
    're_use_walks': False,

    #'alpha': 0.65,
    'tsaw' : {},
    
    'model' : {
      'alpha': 0.65, # 
      'desc': '(P1+P2)/2', # 'alpha*PT + (1-alpha)*PS' '(P1+P2)/2' , # desc modelo average: (P1+P2)/2
      'id': 'average' # alpha, average
    },

    'deepwalk': {
      'p': 1,
      'q': 1
    }, 

    'temporal': {
      'lambda': 1
    },

    'spatial': {
      'beta': 1
    },

    'spatial_hub': {
      'gamma': 1
    },

    'spatial_temporal': {
      'lambda': 0.8,
      'beta': 1
    },

    'spatial_normalization': {
      'beta': 1
    },

    'spatial_temporal_normalization': {
      'lambda': 0.8,
      'beta': 1
    },
    
    'spatial_temporal_normalization_mult': {
      'lambda': 0.8,
      'beta': 0.2
    },    

    'node2vec1': {
      'p': 1.5,
      'q': 0.5
    },

    'node2vec2': {
      'p': 0.5,
      'q': 1
    },

    'sep': ' '

  }
  
  datasets = {
      # we use other function load_graph2() # contem columnas e sep = ','
      #'ml_wikipedia.csv': 'Wikipedia',
      #'ml_enron.csv': 'Enron',
      #'ml_CollegeMsg.csv': 'UCI',
      #'ml_mooc.csv': 'MOOC',
      #'ml_reddit.csv': 'Reddit',
      ### ---------------------------------------------------------

      
      # we use other function load_graph() # nao contem columna e sep = ' ' 
      'ia-contact-short.edges': 'ia-contact-short',
      #'ia-enron-employees.edges': 'ia-enron-employees',
      #'ia-contact.edges' : 'ia-contact',  #ok , error com node2vec
      #'fb-forum.edges': 'fb-forum' # ok

  }

  # ablation study
  type_walks = ['spatial_hub', 'temporal', 'tsaw', 'spatial_normalization', 'spatial_temporal', 'spatial_temporal_normalization'] #testing for ablation 


  clfs = ['logistic']
    
  binary_operators = [operator_l2]
  #binary_operators = [operator_l2, temporal_operator]

  build_all_results(datasets, type_walks, clfs, binary_operators, params)

# tem que cargar o envirotment em conda "stellar"
