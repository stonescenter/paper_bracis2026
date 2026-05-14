import pandas as pd
import numpy as np
import random
from pathlib import Path

# Convert str to list with ast
import ast

import networkx as nx
from stellargraph import StellarGraph

from sklearn.model_selection import train_test_split
from common.classifier import positive_and_negative_links, labelled_links

import warnings
import logging

#logger = logging.getLogger(__name__)

# load all dataset in graph
def load_graph(path, sep=" "):

  assert Path(path).is_file() == True

  edges = pd.read_csv(
      path,
      sep=sep,
      header=None,
      names=["source", "target", "time"],
      usecols=["source", "target", "time"],
  )

  assert len(edges) > 0

  #edges[["source", "target"]] = edges[["source", "target"]].astype(int)
  edges[['source', 'target']] = edges[['source', 'target']].astype(str)

  # get just nodes
  nodes = pd.DataFrame(
      index=np.unique(
          pd.concat([edges["source"], edges["target"]], ignore_index=True)
      )
  )

  # retorna um grafo com nós
  #return nodes, edges
  return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges

def load_graph2(path, sep=','):
  
  assert Path(path).is_file() == True

  df = pd.read_csv(path)
  # ,u,i,ts,label,idx

  df[["u", "i"]] = df[["u", "i"]].astype(str)
  #df[['ts', 'label']] = df[['ts', 'label']].astype(int)
  df[['ts']] = df[['ts']].astype(int)

  edges = pd.DataFrame()

  edges['source'] = df.u.values
  edges['target'] = df.i.values
  edges['time'] = df.ts.values
  #edges['label'] = df.label.values

  nodes = pd.DataFrame(
      index=np.unique(
          pd.concat([edges["source"], edges["target"]], ignore_index=True)
      )
  )

  return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges

def load_dataset(path, sep=" ", ascending=True, casting="str"):

  assert Path(path).is_file() == True

  edges = pd.DataFrame()
  if sep == ' ':
    df = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=["source", "target", "time"],
        usecols=["source", "target", "time"],
    )

    if casting == "str":
      edges[['source', 'target']] = df[['source', 'target']].astype(str)
    elif casting =="int":
      edges[["source", "target"]] = df[["source", "target"]].astype(int)
    
    edges['time'] = df.time.values
  elif sep == ',':

    df = pd.read_csv(path)
    # ,u,i,ts,label,idx

    df[["u", "i"]] = df[["u", "i"]].astype(str)
    #df[['ts', 'label']] = df[['ts', 'label']].astype(int)
    df[['ts']] = df[['ts']].astype(int)
   

    edges['source'] = df.u.values
    edges['target'] = df.i.values
    edges['time'] = df.ts.values

  edges.sort_values(by='time', ascending=ascending)

  return edges



def load_graph_from_nx(g):
  random.seed(888)

  source, target, time = [], [], []

  # geramos numeros aleatorios
  for (u, v) in g.edges():
    n = random.randint(1, g.number_of_edges())
    g.add_edge(u,v,time=n)
    source.append(u)

  edges =  nx.to_pandas_edgelist(g, nodelist=source)
  edges[['source', 'target']] = edges[['source', 'target']].astype(str)
  edges[['time']] = edges[['time']].astype(int)

  nodes = pd.DataFrame(
      index=np.unique(
          pd.concat([edges["source"], edges["target"]], ignore_index=True)
      )
  )

  return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges

def load_static_graph_to_nx(path, sep=","):
  edges  = pd.read_csv(
      path,
      sep=sep,
      header=None,
      names=["source", "target", "time"],
      usecols=["source", "target", "time"],
  )

  edges[['source', 'target']] = edges[['source', 'target']].astype(int) # para node2vec e deepwalk tem que ser int

  g = nx.from_pandas_edgelist(edges, "source", "target", "time")

  return g, edges

def create_dataset2(path_file, train_subset=0.75, test_subset=0.5):

  # cargamos todo dataset
  full_graph, all_edges = load_graph(path_file, sep=' ')
  #full_graph, all_edges = load_graph2(path_file, sep=',')
  #all_edges = load_dataset(path_file, sep=' ', ascending=True)

  # number of edges to be kept in the graph train=0.25
  #num_edges_graph = int(len(all_edges) * (1 - train_subset)) # 75%
  split = int(len(all_edges) * train_subset) # 75%

  # we split the dataset
  # keep older edges in graph, and predict more recent edges
  edges_older = all_edges[:split] # this set is for network representation
  edges_newer = all_edges[split:] # 75%

  # split recent edges further to train and test sets
  #edges_train, edges_test = train_test_split(edges_newer, test_size=test_subset)
  edges_train, edges_test = train_test_split(all_edges, test_size=test_subset)


  #graph = StellarGraph(
  #    nodes=pd.DataFrame(index=full_graph.nodes()),
  #    edges=edges_newer,
  # 
  #    edge_weight_column="time",
  #)

  #print(graph.info())
  print(full_graph.info())

  return full_graph, all_edges, edges_train, edges_test


def create_dataset(path_file, sep, casting="str", train_subset=0.75, test_subset=0.25):

  # cargamos todo dataset
  #full_graph, all_edges = load_graph(path_file  , sep=' ')
  all_edges = load_dataset(path_file, sep=sep, ascending=True, casting=casting)

  # number of edges to be kept in the graph train=0.25
  #num_edges_graph = int(len(all_edges) * (1 - train_subset)) # 75%
  split = int(len(all_edges) * train_subset) # 75%

  # we split the dataset
  # keep older edges in graph, and predict more recent edges
  edges_graph = all_edges[:split] # this set is for network representation
  edges_newer = all_edges[split:] # remainder for train

  # split recent edges further to train and test sets
  #edges_train, edges_test = train_test_split(edges_newer, test_size=test_subset)
  
  nodes = pd.DataFrame(
      index=np.unique(
          pd.concat([all_edges["source"], all_edges["target"]], ignore_index=True)
      )
  )

  full_graph = StellarGraph(
      nodes=nodes,
      edges=edges_graph,  
      edge_weight_column="time",
  )

  return full_graph, all_edges, edges_graph, edges_newer