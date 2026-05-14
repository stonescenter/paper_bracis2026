from numpy  import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

class BaseNetwork:
  #def __new__(cls, *args, **kwargs):
  #  return super().__new__(cls)

  def __init__(self, g):
    self.g = g

  def _convert(self, largest : bool = False):
    G = self.g.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    Gcc = None
    if largest:
      Gcc = max(nx.connected_components(G), key=len)
    else:
      Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
      #Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)

    if len(Gcc) > 0:
      G = G.subgraph(Gcc[0])
    else:
      G = G.subgraph(Gcc)

    self.g = nx.convert_node_labels_to_integers(G, first_label=0)

  def draw(self):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(self.g)
    nx.draw(self.g, pos, node_color="lightblue", node_size=500, with_labels=True)

  def degree(self):
    vk = dict(self.g.degree()).values()
    vk = np.array(list(vk))
    return vk

  def momment(self, m):
      M = 0
      N = len(self.g)
      for i in self.g.nodes:
          M = M + self.g.degree(i)**m
      M = M/N
      return M

  def complexidade(self):
    m = self.momment(2)

    vk = dict(self.g.degree()).values()
    vk = np.array(list(vk))

    md = mean(vk)

    return m/md

  def print_info(self):

    print('Number of nodes:', len(self.g))
    print('Number of edges:', self.g.number_of_edges())

    #print('Primeiro momento de k:', self.momment(1))
    #print('Segundo momento de k:', self.momment(2))
    #print('Terceiro momento de k:', self.momment(3))
    #print('Variância de k:', np.var(self.degree()))
    #print('Mediana de k:', np.median(self.degree()))
    print('Average Degree:', 2*self.g.number_of_edges()/len(self.g))
    #print('Calculo da complexidade:', self.complexidade())

  def avg_degree(self):
    return  2*self.g.number_of_edges()/len(self.g)

  def degree_distribution(self):
      '''
        A partir do grau dos vértices, podemos calcular a distribuição do grau.
      '''
      vk = dict(self.g.degree())
      vk = list(vk.values())  # we get only the degree values
      vk = np.array(vk)
      maxk = np.max(vk)
      mink = np.min(vk)
      kvalues= np.arange(0,maxk+1) # possible values of k
      Pk = np.zeros(maxk+1) # P(k)
      for k in vk:
          Pk[k] = Pk[k] + 1
      Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one

      return kvalues, Pk

  def plot_degree_distribution(self, ks=None, Pk=None):

      if (ks is None or Pk is None):
        ks, Pk = self.degree_distribution()

      plt.figure(figsize=(8,5))
      fig = plt.subplot(1,1,1)
      fig.set_xscale('log')
      fig.set_yscale('log')
      plt.plot(ks,Pk,'bo')
      plt.xlabel("k", fontsize=20)
      plt.ylabel("P(k)", fontsize=20)
      plt.title("Degree distribution", fontsize=20)
      #plt.grid(True)
      plt.savefig('degree_dist.eps') #save the figure into a file
      plt.show(True)

  def momment_of_degree_distribution(self, m):
      k,Pk = self.degree_distribution()
      M = sum((k**m)*Pk)
      return M

  def shannon_entropy(self):
      k,Pk = self.degree_distribution()
      H = 0
      for p in Pk:
          if(p > 0):
              H = H - p*math.log(p, 2)
      return H

  def normalized_shannon_entropy(self):
      k,Pk = self.degree_distribution()
      H = 0
      for p in Pk:
          if(p > 0):
              H = H - p*math.log(p, 2)
      return H/math.log(len(self.g),2)

  def transitivity(self):
      return nx.transitivity(self.g)

  def average_clustering(self):
    return nx.average_clustering(self.g)

  def average_shortest_path_length(self):
    '''
      The distance between pairs of nodes is given by the number of edges between
      them, where edges and nodes are not repeated. The distance with the minimal
      length is called the shortest path. The average shortest path length can be
      calculated by:
    '''
    if nx.is_connected(self.g) == True:
      l = nx.average_shortest_path_length(self.g)
      #print("Average shortest path length:", "%3.4f"%l)
    else:
      print("The graph has more than one connected component")

      #Gcc=sorted(nx.connected_component_subgraphs(self.g), key = len, reverse=True)
      G=list(self.g.subgraph(c) for c in nx.connected_components(self.g))[0]
      N = len(self.g)
      M = G.number_of_edges()
      avg = 2*M/N
      l = nx.average_shortest_path_length(self.g)
      print("New average shortest path length:", "%3.4f"%l)
      l = log(N)/log(self.avg_degree())
      print("The predicted value is log(N)/log(average degree)", l)

    return l

  def diameter(self):
    '''
      The shortest path with the longest length is the diameter.
    '''
    return nx.diameter(self.g)

  def nodes(self):
    return self.g.nodes()

  def num_nodes(self):
    return len(self.g)

  def num_edges(self):
    return self.g.number_of_edges()

from scipy.stats import binom
from typing import Optional
from random import *

class GraphNetwork(BaseNetwork):

  def __init__(self, file_name: str, type_file: str = "edge"):
    g = None
    self.name = file_name

    if type_file == "edge":
      g = nx.read_edgelist(file_name, nodetype=int, data=("time", int) )
    elif type_file == "gml":
      g = nx.read_gml(file_name)

    BaseNetwork.__init__(self, g)

class RandomNetwork(BaseNetwork):
  def __init__(self, N, p):
    self.name = "random binomial"
    g = nx.gnp_random_graph(N, p, seed=None, directed=False)
    print(f"\t {self.name} model created, with p:{p}")


    BaseNetwork.__init__(self, g)
    #super().__init__(self, g)

  def plot(self):
    pos = nx.fruchterman_reingold_layout(self.g);
    plt.figure(figsize=(8,8));
    plt.axis("off");
    nx.draw_networkx_nodes(self.g, pos, node_size=300, node_color="black");
    nx.draw_networkx_edges(self.g, pos, alpha=0.500);
    nx.draw_networkx_labels(self.g, pos, font_color="white");
    plt.show()

  def momment_degree_distribuition(self, m):
    k,Pk = self.degree_distribution()
    M = sum((k**m)*Pk)
    return M

  def plot_degree_distribution(self):

    ks, Pk = self.degree_distribution()

    plt.figure(figsize=(8,6))
    plt.plot(ks,Pk,'bo', label='Data')
    plt.xlabel("k", fontsize=20)
    plt.ylabel("P(k)", fontsize=20)
    plt.title("Degree distribution", fontsize=20)
    plt.grid(True)
    plt.savefig('degree_dist.eps') #save the figure into a file

    ## Fitting of. binomial distribution
    pk = binom.pmf(ks, N, p)
    plt.plot(ks, pk, 'r', label='Binomial distribution')
    plt.legend()
    plt.show(True)

class ModelNetwork(BaseNetwork):
  #def __init__(self, N: int, p: float, k: Optional[int] = 10, n: str = 'smallworld'):
  def __init__(self,  *args):
    N = args[0]
    n = args[1]
    name = ""
    g = None
    if n=='smallworld':
      k = args[2]
      p = args[3]
      #print(f"k:{k} p:{p}")
      g = nx.watts_strogatz_graph(N, k, p)
      print(f"\t {n} model created, with k:{k}, p:{p}")

    elif n == 'erdos':
      p = args[2] # probabilidade de conexão
      g = nx.erdos_renyi_graph(N,p)
      print(f"\t {n} renyi model created, p:{p}")

    elif n == "barabasi":
      m = args[2]
      g = nx.barabasi_albert_graph(N, m)
      print(f"\t {n} model created, with m:{m}")

    elif n == 'waxman':
      #p = args[2] # probabilidade de conexão
      beta = args[2]
      alpha = args[3]
      g = nx.waxman_graph(N, beta=beta, alpha=alpha)
      print(f"\t {n} model created, beta:{beta}, alpha:{alpha}")

    elif n == "configuration":
      #sequence = nx.random_powerlaw_tree_sequence(100, tries=5000)
      a = args[2]
      seq = np.random.zipf(a, N) #Zipf distribution
      #seq = np.random.poisson(10, N) #Poisson distribution

      if(sum(seq)%2 != 0): # the sum of stubs have to be even
          pos = randint(0, len(seq))
          seq[pos] = seq[pos]+ 1

      g = nx.configuration_model(seq)
      print(f"\t {n} model created, with a:{a}")
    else:
      print("G graph is None")

    self.name = n

    BaseNetwork.__init__(self, g)
    #super().__init__(self, g)

  def momment_degree_distribuition(self, m):
    k,Pk = self.degree_distribution()
    M = sum((k**m)*Pk)
    return M

  def plot(self):
    pos = nx.fruchterman_reingold_layout(self.g);
    plt.figure(figsize=(8,8));
    plt.axis("off");
    nx.draw_networkx_nodes(self.g, pos, node_size=300, node_color="black");
    nx.draw_networkx_edges(self.g, pos, alpha=0.500);
    nx.draw_networkx_labels(self.g, pos, font_color="white");
    plt.show();

  def plot_degree_distribution(self):

    ks, Pk = self.degree_distribution()

    plt.figure(figsize=(8,6))
    plt.plot(ks,Pk,'bo', label='Data')
    plt.xlabel("k", fontsize=20)
    plt.ylabel("P(k)", fontsize=20)
    plt.title("Degree distribution", fontsize=20)
    plt.grid(True)
    plt.savefig('degree_dist.eps') #save the figure into a file

    ## Fitting of. binomial distribution
    pk = binom.pmf(ks, N, p)
    plt.plot(ks, pk, 'r', label='Binomial distribution')
    plt.legend()
    plt.show(True)



def get_measures(g):
  features = {}
  r=nx.degree_assortativity_coefficient(g.g)
  #centrilities = calculate_principal_centralities(G.g)
  features['model'] = g.name
  features['coef_assortativity'] = r
  features['second_momment'] = g.momment(2)
  features['transitivity'] = g.transitivity()
  features['average_clustering'] = g.average_clustering()
  features['shannon_entropy'] = g.shannon_entropy()
  features['average_shortest_path_length'] = g.average_shortest_path_length()
  features['average_degree'] = g.avg_degree()
  features['diameter'] = g.diameter()


  return features


def create_models(N, avg_degree, cat, name, qty=1):

  X = []
  y = []

  # p é Probability for edge creation
  # p = <k>/(N-1)
  p = int(avg_degree)

  beta = 0.1
  alpha = 1

  for q in range(0, qty):
    G = ModelNetwork(N, "waxman", beta, alpha)
    G._convert()

    features = get_measures(G)
    features['name'] = name
    features['setup'] ="beta:" + str(beta) +", alpha:" + str(alpha)
    X.append(features)
    y.append(cat)

  # p é Probability for edge creation
  # p = <k>/(N-1)
  p = avg_degree/(N-1)
  for q in range(0, qty):
    G = ModelNetwork(N, "erdos", p)
    G._convert()

    features = get_measures(G)
    features['name'] = name
    features['setup'] = "p:" + str(round(p,2))
    X.append(features)
    y.append(cat)


  # um BARABASI
  # m é o Number of edges to attach from a new node to existing nodes
  # If m does not satisfy 1 <= m < n, or the initial graph number of nodes m0 does not satisfy m <= m0 <= n.
  # m = int(avg_degree/2)
  m = int(avg_degree/(2))
  for q in range(0, qty):
    if 1 <= m < N:
      G = ModelNetwork(N, "barabasi", m)
      G._convert()

      features = get_measures(G)
      features['name'] = name
      features['setup'] ="m:" + str(m)
      X.append(features)
      y.append(cat)

    else:
      print(f"Error creating Barabasi model for {name} m: {m}")

  # um SMALLWORD com diferentes p
  # watts_strogatz_graph - smallworld
  k  = int(avg_degree)
  ps = [0.01, 0.002, 0.005, 1]
  for q in range(0, qty):
    for p in ps:
      k  = int(avg_degree)
      if k == 1: # nao pode ser menor
        k = round(avg_degree)
      elif k > N:
        k  = int(avg_degree/2)
      else:
        k = int(avg_degree)

      G = ModelNetwork(N, "smallworld", k , p)
      #print("_convert")
      G._convert()

      features = get_measures(G)
      features['name'] = name
      features['setup'] ="avg_degree:" + str(k) +", p:" + str(round(p,2))
      X.append(features)
      y.append(cat)

  return X, y

def create_data_set(map_data, quantity=10, quantity_by_net=1, max_nodes=2500):

  no_process = 0
  processed  = 0

  X = []
  y = []

  for entry in map_data:
    tipo = entry['type']
    nets = entry['nets'] # lista

    no_process = 0
    processed  = 0
    print(f"For {tipo} nets :")

    for n in nets:

      if len(n['nets']) == 1: # contem so uma rede

        N = int(n['analyses']['num_vertices'])
        avg_degree = round(n['analyses']['average_degree'], 2)

        # redes com numero de nós muito grande descartamos demora demais criando
        if N > max_nodes:
          print(f"Network {n['title']}, not processed because have {N} nodes execeded the limit {max_nodes}")
          continue

        m = int(avg_degree/2)
        if m > N:
          print(f"Network {n['title']}, no processed because the parameter {m} > {N} nodes")
          continue

        # criamos 3 modelos para cada entrada de Biological
        # um ERDOS
        print(f"Creating models for '{n['title']}' , {N} nodes, average degree {avg_degree}")
        X, y = create_models(N, avg_degree, tipo, n['title'], quantity_by_net)
        processed+=1
      else:
        no_process+=1

      print(f"{processed} networks processes")

      if processed == quantity:
        break


    print(f"{processed} nets processed.")
    print(f"{no_process} nets not process.")

  return X, y


def create_base_graph(path_file):
  g,  edges = load_static_graph_to_nx(path_file, sep=" ")

  print("File:", path_file)
  print('Average Degree:', 2*g.number_of_edges()/len(g))
  print('Num vertices:', len(g.nodes()))
  print('Num edges:', g.number_of_edges())

  avg_degree = 2*g.number_of_edges()/len(g)

  # Calcular o grau de todos os nós
  degrees = dict(g.degree())

  # Encontrar o grau máximo
  max_degree = max(degrees.values())

  # Encontrar o(s) nó(s) com o grau máximo
  nodes_with_max_degree = [node for node, degree in degrees.items() if degree == max_degree]

  print("Max degre", max_degree)
  #G.draw()

  return g

def create_sintetic_models(g):

  sintetic_models = []

  N = len(g.nodes())

  avg_degree = 2*g.number_of_edges()/N

  ###################################################################
  # p é Probability for edge creation
  # p = <k>/(N-1)
  p = avg_degree/(N-1)

  params = {}
  params['avg_degree'] = avg_degree
  params['p'] = p

  graph_name = 'erdos'

  model = ModelNetwork(N, graph_name, p)
  model._convert()
  #model.draw()
  sintetic_models.append(model)

  ###################################################################
  # um SMALLWORD com diferentes p
  # watts_strogatz_graph - smallworld
  k  = int(avg_degree)
  p = 0.01

  k  = int(avg_degree)
  if k == 1: # nao pode ser menor
    k = round(avg_degree)
  elif k > N:
    k  = int(avg_degree/2)
  else:
    k = int(avg_degree)

  params = {}
  params['avg_degree'] = avg_degree
  params['k'] = k
  params['p'] = p

  graph_name = 'smallworld'
  model = ModelNetwork(N, graph_name, k , p)
  model._convert()
  #model.draw()
  sintetic_models.append(model)

  ###################################################################
  # um BARABASI
  # m é o Number of edges to attach from a new node to existing nodes
  # If m does not satisfy 1 <= m < n, or the initial graph number of nodes m0 does not satisfy m <= m0 <= n.
  # m = int(avg_degree/2)
  m = int(avg_degree/(2))

  params = {}
  params['avg_degree'] = avg_degree
  params['m'] = m

  graph_name = 'barabasi'

  model = ModelNetwork(N, graph_name, m)
  model._convert()
  #model.draw()

  sintetic_models.append(model)

  ###################################################################

  avg_degree = 2*g.number_of_edges()/len(g)
  p = int(avg_degree)

  beta = 0.1
  alpha = 1
  N = len(g.nodes())


  graph_name = 'waxman'
  # Creamos um modelo sintetico
  model = ModelNetwork(N, graph_name, beta, alpha)
  model._convert()
  #model.draw()
  sintetic_models.append(model)


  return sintetic_models
