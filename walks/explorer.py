import numpy as np 
from typing import List
from collections import namedtuple

import warnings
from collections import defaultdict, deque
from scipy import stats

import random
import time 
import math

import random as rn
import numpy.random as np_rn

from scipy import stats
from scipy.special import softmax
from tqdm import tqdm

import networkx as nx

from stellargraph import StellarGraph

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

RandomState = namedtuple("RandomState", "random, numpy")

# Some parts of code is based on the project StellarGraph

def _global_state():
    return RandomState(rn, np_rn)

def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))

global _rs
_rs = _global_state()

def random_state(seed):
    """
    Create a RandomState using the provided seed. If seed is None, return the global RandomState.

    Args:
        seed (int, optional): random seed

    Returns:
        RandomState object
    """
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)

def naive_weighted_choices(weights, size=None):
    probs = np.cumsum(weights) 
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return None
    thresholds = random.random() if size is None else random.random(size)

    idx = 0
    # posicao a ser insertado
    idx = np.searchsorted(probs, thresholds * total, side="left")

    return idx
    
class UniformRandomWalk():
    """
    Performs uniform random walks on the given graph
    """

    def __init__(self, graph, num_walks=None, length=None, seed=None):
        #__init__(graph, seed=seed)
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = length
    
    def chunk_nodes(self, workers):
        nodes = list(self.graph.nodes())

        chunk_size = len(nodes) // workers

        node_chunks = [(nodes[i:i + chunk_size]) for i in range(0, len(nodes), chunk_size)]

        print(f"Each cpu is processing: {chunk_size} nodes of {len(nodes)}, {len(node_chunks)}")

        return node_chunks
    
    def run(self, nodes, n=None, length=None, seed=None):
        n = n
        length = length
        nodes = self.graph.node_ids_to_ilocs(nodes)

        # for each root node, do n walks
        return [self._walk(node, length) for node in nodes for _ in range(n)]
           
    def walk(self, start_node, length):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
            if len(neighbours) == 0:
                # dead end, so stop
                break
            else:
                # has neighbours, so pick one to walk to
                current_node = random.choice(neighbours)
            walk.append(current_node)

        return list(self.graph.node_ilocs_to_ids(walk))
    
    def neighbors(self, node):
        #neighbours = self.graph.neighbor_arrays(node, use_ilocs=True)
        neighbours = self.graph.neighbor_arrays(node)
        return neighbours

    def generate_walks_parallel(self, start_node):
      walk = [start_node]
      current_node = start_node

      for _ in range(self.length - 1):
          neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
          #neighbours = self.graph.neighbor_arrays(current_node)
          if len(neighbours) == 0:
              # dead end, so stop
              break
          else:
              # has neighbours, so pick one to walk to
              current_node = np.choice(neighbours)
          walk.append(current_node)

      return list(self.graph.node_ilocs_to_ids(walk))
      #return walk
    
    def random_walk(self, start_node):
        walk = [start_node]
        current_node = start_node

        for _ in range(self.walk_length - 1):
            neighbors = list(self.neighbors(current_node))

            if len(neighbors) == 0:
                break
 
            current_node = random.choice(neighbors)
            walk.append(current_node)

        return walk
    
    def generate_walks_for_nodes(self, nodes):
        walks = []
        for node in nodes:
            for _ in range(self.num_walks):
                walk = self.random_walk(node)
                walks.append(walk)

        return walks
    

    def run_walk(self, num_walks=10, walk_length=80, seed=None, workers=10):

        #nodes = self.graph.node_ids_to_ilocs(nodes)
        self.num_walks = num_walks
        data_to_process = self.chunk_nodes(workers) # funciona

        self.walk_length = walk_length
        
        walks = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.generate_walks_for_nodes, chunk ) for chunk in data_to_process] # 

            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunk processed "):
                result = future.result()

                if result is not None:
                    walks.extend(result)             
           
        return walks


class TSAW:
  #def __init__(self, g: nx.Graph, walk_length: int, walks_per_node: int):
  def __init__(self, g: nx.Graph):

    #self.walk_length = walk_length
    #self.walk_per_node = walks_per_node
    self.g = g

  def random_walk(self, walk_length: int, start_node: str):

    prev_node = start_node
    walks = [start_node]

    for i in range(walk_length):
      nodes = list(self.g.neighbors(prev_node))
      nodes = list(set(nodes) - set(walks))

      if nodes == 0:
        break

      new_node = random.choice(nodes)
      walks.append(new_node)
      prev_node = new_node

    return walks


  def simple_random_walk(self, start_node, num_steps):
    walk = [start_node]
    current_node = start_node
    #print(f"start {start_node} ,type: {type(start_node)}")
    for _ in range(num_steps-1):
        neighbors = list(self.g.neighbors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
        current_node = next_node

    return walk

  def viased_random_walk(self, start_node, num_steps):
    walk = [start_node]
    current_node = start_node

    for _ in range(num_steps-1):
        neighbors = list(self.g.neighbors(current_node))
        if not neighbors:
            break

        # Compute the degrees of the neighboring nodes
        degrees = np.array([self.g.degree(neighbor) for neighbor in neighbors])

        # The probability of moving to a neighbor is proportional to its degree
        probabilities = degrees / degrees.sum()

        next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
        walk.append(next_node)
        current_node = next_node

    return walk

  def random_tsaw(self, start_node, walk_length: int, 
                  use_probabilities: bool = False,
                  repeted: bool = False) -> List[str]:
    """
    Generate a random walk starting on start
    :param g: Graph
    :param start: starting node for the random walk
    :param use_probabilities: if True take into account the weights assigned to each edge to select the next candidate
    :return:
    """

    def all_node_are(walk, neighbors):

      for n in neighbors:
        if not n in walk:
          return False

      return True

    walk = [start_node]
    print(f"start {start_node} ,type: {type(start_node)}")
    node = None

    i = 0
    while i < walk_length-1:

        #print("i=", i)
        #print(f"\ti : {i}, node : {walk[i]}, neigh={list(g.neighbors(walk[i]))}")
        neighbours =self.g.neighbors(walk[i])
        neighs = list(neighbours)
        if use_probabilities:
            probabilities = [self.g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
            sum_probabilities = sum(probabilities)
            probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
            node = np.random.choice(neighs, p=probabilities)
        else:
            node = random.choice(list(set(neighs) - set(start_node))) # sem pegar o inicio

        if (node == start_node and len(neighs) == 1):
          break

        if len(walk) == walk_length-1:
          break

        if all_node_are(walk, neighs):
          break

        if repeted:
          walk.append(str(node))
        else:
          if not node in walk:
            walk.append(str(node))
            #print(f"walk : {walk}, added: {node}")
          else:
            #print("not added ", node)
            continue

        i+=1

    return walk

  def true_self_avoiding_walk(self, start_node, num_steps):
    walk = [start_node]
    current_node = start_node
    visited = set(walk)

    for _ in range(num_steps):
        neighbors = list(self.g.neighbors(current_node))
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            break
        next_node = random.choice(unvisited_neighbors)
        walk.append(next_node)
        visited.add(next_node)
        current_node = next_node

    return walk

  def visitation_exponential_transition_probability(self, visitation_count, unvisited_neighbors):
    visit_counts = np.array([visitation_count[n] for n in unvisited_neighbors])
    probabilities = np.exp(-visit_counts)
    probabilities /= probabilities.sum()  # Normalize to get probabilities

    return probabilities

  def transition_probability(self, current_node, unvisited_neighbors):
    def freq(n):
        return 1.0/self.g.degree(n)

    lamda = math.log(2)

    degrees = np.array([self.g.degree(neighbor) for neighbor in unvisited_neighbors])
    numerator = np.exp(-lamda*freq(current_node))
    probabilities = numerator / degrees.sum()*numerator
    return probabilities

  def true_self_avoiding_walk(self, start_node, num_steps):
    walk = [start_node]
    current_node = start_node
    visited = set(walk)

    visitation_count = {node: 0 for node in self.g.nodes}
    visitation_count[start_node] += 1

    for _ in range(num_steps):
        neighbors = list(self.g.neighbors(current_node))
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            break

        # Get transition probabilities for unvisited neighbors
        #print(current_node, unvisited_neighbors)
        #probabilities = self.transition_probability(current_node, unvisited_neighbors)
        probabilities = self.visitation_exponential_transition_probability(visitation_count, unvisited_neighbors)
        #print(probabilities)

        # Select the next node based on the transition probabilities
        next_node = random.choices(unvisited_neighbors, weights=probabilities, k=1)[0]
        walk.append(next_node)
        visited.add(next_node)
        current_node = next_node

    return walk

  def random_tsaw(self, start_node, walk_length: int, use_probabilities: bool = False, repeted: bool = False) -> List[str]:
    return self.true_self_avoiding_walk(start_node, num_steps=walk_length)

  def run_walk(self, type_walk: str, num_walks: int, walk_length: int, repeted: bool):

    nodes = list(self.g.nodes())
    walks = []

    for w in range(num_walks):
      random.shuffle(nodes)
      for node in nodes:
        if type_walk == "simple_rw":
          walks.append(self.simple_random_walk(start_node=node, num_steps=walk_length))
        elif type_walk == "tsaw":
          walk = self.random_tsaw(start_node=node, walk_length=walk_length, repeted=repeted)
          if not walk in walks:
            walks.append(walk)

    return walks

  def _exp_biases(self, t_0, times, decay):
      # t_0 assumed to be smaller than all time values
      #return softmax(t_0 - np.array(times) if decay else np.array(times) - t_0)
      return softmax(np.array(times) - t_0 if decay else np.array(times) - t_0)


  def _spatial_biases(self, degrees, beta=1):
    return softmax(beta/np.array(degrees))
    #return softmax(np.array(degrees)) 
  
  def _sample(self, n, biases, np_rs):
    '''
      retorna uma amostra, aleatoria a partir de um vetor de biases
      n: numero de tempos
      biases: probabilidades
    '''
    if biases is not None:
        assert len(biases) == n
        # passa um random state global e os todas as probabilidades
        return naive_weighted_choices(np_rs, biases)
    else:
      # gera um numero entre n
        return np_rs.choice(n)
        
  def _temporal_biases(self, time, times, bias_type, is_forward):
      if bias_type is None:
          # default to uniform random sampling
          return None

      # time is None indicates we should obtain the minimum available time for t_0
      t_0 = time if time is not None else min(times)

      if bias_type == "temporal":
          # exponential decay bias needs to be reversed if looking backwards in time
          return self._exp_biases(t_0, times, decay=is_forward)

      else:
          raise ValueError("Unsupported bias type")

  def _step(self, node, time, bias_type, p, q, np_rs):
      """
      Perform 1 temporal step from a node. Returns None if a dead-end is reached.

      """
      neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)
      neighbors = self.graph.neighbors(node)

      #print(f"Debug neighoburs len: {len(neighbours)}")
      neighbours = neighbours[times > time]
      times = times[times > time]
      #print(f"Debug neighoburs len: {len(neighbours)}, {len(times)}")

      biases = None
      if len(neighbours) > 0:
          # lista de probabilidades

          if bias_type == "temporal":
            biases = self._temporal_biases(time, times, bias_type, is_forward=True)

          elif bias_type == "spatial":

            degrees = self._get_degree_neighbors(neighbours)
            biases = self._spatial_biases(degrees)

          elif bias_type == "spatial_temporal":

            temp_biases = self._temporal_biases(time, times, "temporal", is_forward=True)

            degrees = self._get_degree_neighbors(neighbours)

            spatial_biases = self._spatial_biases(degrees, beta=1)
            #print(f"spatial: {spatial_biases}, temporal :{temp_biases}, shape: {(len(spatial_biases),len(temp_biases))}")
            #print(f"Shape: {(len(spatial_biases), len(temp_biases))}")

            #biases = np.average(temp_biases+spatial_biases)
            biases = (p*np.array(temp_biases) + q*np.array(spatial_biases))/len(temp_biases)

            #chosen_neighbour_index = self._sample(len(neighbours), biases, np_rs)

          #print(f"Len biases: {len(biases)}")
          chosen_neighbour_index = self._sample(len(neighbours), biases, np_rs)

          assert chosen_neighbour_index is not None, "biases should never be all zero"

          next_node = neighbours[chosen_neighbour_index]
          next_time = times[chosen_neighbour_index]

          print(biases, next_node, next_time)
          
          return next_node, next_time
      else:
          return None
  
  def _walk(self, src, dst, t, length, bias_type, p, q, np_rs):
      walk = [src, dst] # normalmente é o mesmo nó 
      node, time = dst, t

      for _ in range(length - 2):
          # seleciono
          result = self._step(node, time=time, bias_type=bias_type, p=p, q=q, np_rs=np_rs)

          if result is not None:
              node, time = result
              walk.append(node)
          else:
              break

      return walk
                  
def open_walk(file_name, convert_str=False):

  #path_file = root + file_name

  walks = []
  with open(file_name, 'r') as f:
    #walks = [ast.literal_eval(line) for line in f] # tira os comillas e deixa ['2', '4']
    for line in f:
      walk = ast.literal_eval(line)

      if convert_str:
        walk = [ str(w) for w in walk]

      walks.append(walk)

  return walks


class TemporalRandomWalk():


    def __init__(
        self,
        graph,
        cw_size=None,
        max_walk_length=80,
        initial_edge_bias=None,
        walk_bias=None,
        p_walk_success_threshold=0.01,
        seed=None,
    ):
        #super().__init__(graph, graph_schema=None, seed=seed)
        self.graph = graph
        self.cw_size = cw_size
        self.max_walk_length = max_walk_length
        self.initial_edge_bias = initial_edge_bias
        self.walk_bias = walk_bias
        self.p_walk_success_threshold = p_walk_success_threshold
   
        #self._check_seed(seed)
        #self._random_state, self._np_random_state = random_state(seed) # comentar para paralelo
        self.graph_schema = self.graph.create_graph_schema()

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Use the class's random state
            return self._random_state, self._np_random_state
        # seed the random number generators
        return random_state(seed)
     
    def run_walk(self, 
        num_cw,
        cw_size=None,
        max_walk_length=None,
        walk_bias=None,
        num_walks_per_node =10,
        is_forward=True,
        lamb = 0.8,
        beta = 1,
        model = None,
        debug=False,
        workers=10,
        parallel_exploration='edges'):

        self.lamb = lamb
        self.beta = beta
        
        self.model = model
        self.model_alpha = model['alpha']
        self.model_id = model['id']

        self.num_cw = num_cw
        self.cw_size = cw_size
        self.max_walk_length = max_walk_length
        self.walk_bias = walk_bias
        self.is_forward = is_forward
        self.seed = None
        self.initial_edge_bias=None
        self.num_walks_per_node = num_walks_per_node


        #data_to_process = self.chunk_nodes(workers) # funciona
        data_to_process = self.chunk_edges(workers) # funciona
  
        self.mapping_degrees = self.graph.node_degrees()
        eps = 1e-10

        self.inv_sqrt_degree = {
            n: 1 / np.sqrt(deg + eps)
            for n, deg in self.mapping_degrees.items()
        }
        
        walks = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if parallel_exploration == 'edges':
                futures = [executor.submit(self.generate_walks_for_edges, s, d, t ) for s, d, t in data_to_process] #  funciona OK porem parece que buga no While
            elif parallel_exploration == 'nodes':
                futures = [executor.submit(self.generate_walks_for_nodes, s, d, t ) for s, d, t in data_to_process] # 

            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunk processed "):
                result = future.result()

                if result is not None:
                    walks.extend(result)             
           
        return walks

    def chunk_edges(self, workers):

        sources, targets, _, times = self.graph.edge_arrays(include_edge_weight=True)
        chunk_size = len(sources) // workers
        
        sources_chunks = [sources[i:i + chunk_size] for i in range(0, len(sources), chunk_size)]
        targets_chunks = [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]
        times_chunks = [times[i:i + chunk_size] for i in range(0, len(times), chunk_size)]

        print(f"Each cpu is processing: {chunk_size} edges of {len(sources)}, {len(sources_chunks)} total edges")

        data = [(source, target, time) for source, target, time  in zip(sources_chunks, targets_chunks, times_chunks)]
        return data
    
    def chunk_nodes(self, workers):
    
        nodes = list(self.graph.nodes())

        chunk_size = len(nodes) // workers

        node_chunks = [(nodes[i:i + chunk_size]) for i in range(0, len(nodes), chunk_size)]

        print(f"Each cpu is processing: {chunk_size} nodes of {len(nodes)}, {len(node_chunks)}")

        return node_chunks
       
    def generate_walks_for_edges(self, sources, targets, times):
        
        def not_progressing_enough():
            # Estimate the probability p of a walk being long enough; the 95% percentile is used to
            # be more stable with respect to randomness. This uses Beta(1, 1) as the prior, since
            # it's uniform on p
            posterior = stats.beta.ppf(0.95, 1 + successes, 1 + failures)
            return posterior < self.p_walk_success_threshold


        #_, np_rs = self._get_random_state(None)
       
                
       # initial_edge_bias is None que significa distribuicao uniforme
        if self.is_forward:
            edge_biases = self._temporal_biases(None, times, bias_type=self.initial_edge_bias, is_forward=True, lamb=1)
        else:
            edge_biases = self._temporal_biases(None, times, bias_type=self.initial_edge_bias, is_forward=False, lamb=1)            
 
        walks = []
        num_cw_curr = 0
        
        successes = 0
        failures = 0
          
        # loop runs until we have enough context windows in total
        while num_cw_curr < self.num_cw:
            # retorna uma amostra, de os menores tempos
            first_edge_index = self._sample(len(times), edge_biases)

            src = sources[first_edge_index]
            dst = targets[first_edge_index]
            t = times[first_edge_index]

            remaining_length = self.num_cw - num_cw_curr + self.cw_size - 1
            #remaining_length = np.min(self._get_degree(src) , self.max_walk_length) + 1
           
            walk = self._walk(
                src, dst, t, min(self.max_walk_length, remaining_length), self.walk_bias, self.is_forward, debug=False
            )

            if len(walk) >= self.cw_size:
                walks.append(walk)
                num_cw_curr += len(walk) - self.cw_size + 1
                successes += 1
            else:
                failures += 1
                if not_progressing_enough():
                    raise RuntimeError(
                        f"Discarded {failures} walks out of {failures + successes}. "
                        "Too many temporal walks are being discarded for being too short. "
                        f"Consider using a smaller context window size (currently cw_size={self.cw_size})."
                    )

        return walks

    def generate_walks_for_nodes(self, sources, targets, times):
        def not_progressing_enough(successes, failures):
            # Estimate the probability p of a walk being long enough; the 95% percentile is used to
            # be more stable with respect to randomness. This uses Beta(1, 1) as the prior, since
            # it's uniform on p
            #posterior = stats.beta.ppf(0.95, 1 + successes, 1 + failures)
            #return posterior < self.p_walk_success_threshold
            return successes / (successes + failures)  
        
        # initial_edge_bias is None que significa distribuicao uniforme
        # retorna um tempo probable dentro de times
        #if self.is_forward:
        #    edge_biases = self._temporal_biases(None, times, bias_type=self.initial_edge_bias, is_forward=True, lamb=1)
        #else:
        #    edge_biases = self._temporal_biases(None, times, bias_type=self.initial_edge_bias, is_forward=False, lamb=1)            
        
        walks = []
        
        successes = 0
        failures = 0
        #num_samples = len(sources)

        #for _ in num_samples:
        # retorna uma amostra, de os menores tempos na primeira seleção 
        #first_edge_index = self._sample(len(times), edge_biases)
        #src, dst, t = sources[first_edge_index], targets[first_edge_index], times[first_edge_index]
            
        for src, dst, t in zip(sources, targets, times):
            for _ in range(self.num_walks_per_node):
                walk = self.temporal_walk(src, dst, t, self.walk_bias, self.is_forward, debug=False)

                if len(walk) >= self.cw_size:
                    walks.append(walk)
                    successes += 1
                else:
                    failures += 1
                    if not_progressing_enough(successes, failures):
                        raise RuntimeError(
                            f"Discarded {failures} walks out of {failures + successes}. "
                            "Too many temporal walks are being discarded for being too short. "
                            f"Consider using a smaller context window size (currently cw_size={self.cw_size})."
                        )
                   
        #print(f"Discarted {discarted_walks} walks in this batch")    

        return walks
    
    #def _exp_biases(self, times, t_0, decay):
    #    # t_0 assumed to be smaller than all time values
    #    return softmax(t _0 - np.array(times) if decay else np.array(times) - t_0)
    
    def _exp_biases(self, t_0, times, decay, lamb=1):
        # t_0 : assumed to be smaller than all time values
        # t_0 : current time
        # times : next times of neighbors
        # lambda :
        #       small lambda - weak temporal effect
        #       large lambda - strong preferences for recent edges    

        # decay is_backward = True by default
        # is_backward = True  is ascending example : 1  2 3 4 5    t_i+1 > t_0  t' > t
        # is_backward = False is descending example : 5, 4, 3, 2, 1 t_i-1 < t_0  t' < t
        # is looking backward in the time:  9, 8, 7, 6 . . .
        #  
        #print(f"decay {decay}")
        #return softmax(t_0 - np.array(times) if decay else np.array(times) - t_0)
        #return softmax(-lamb*(t_0 - np.array(times)) if decay else np.array(times) - t_0)
        return softmax(-lamb*(np.array(times) - t_0) if decay else -lamb*(t_0 - np.array(times)))
             
    def _spatial_biases(self, degrees, beta=1):
        degrees = np.array(degrees, dtype=float)
        scores = -beta / (degrees + 1e-10)   # -beta strong dominance by hubs and avoid div by zero
        return softmax(scores)    
        #return softmax(np.array(degrees)) 

    def _spatial_biases_hub(self, degrees, gamma=1.0):
        '''
        gamma = 0 → uniform random walk
        gamma > 0 → hub-biased
        gamma >> 1 → strongly hub-dominated
        '''
        degrees = np.asarray(degrees, dtype=float)

        scores = np.power(degrees, gamma)
        probs = scores / scores.sum()
        return probs
    
    def _spatial_biases_with_normalization(self, degree, degrees, beta=1):
      
        #return softmax(beta/(np.sqrt(np.array(degree) * np.array(degrees))))
        scores = []
        for d in degrees:
            score = beta / np.sqrt(degree*d)
            scores.append(score)

        scores = np.array(scores)
        probs = scores/scores.sum()

        return probs
    
    def _spatial_biases_with_normalization(self, degree, degrees, beta=1):
        degrees = np.asarray(degrees)
        scores = beta / np.sqrt(degree * degrees)
        return scores / scores.sum()

    def _spatial_biases_with_normalization_optimized(self, node, neighbours):
               
        inv_deg_node = self.inv_sqrt_degree[node]
        #inv_deg_neighbors = np.array([self.inv_sqrt_degree[n] for n in neighbours])
        inv_deg_neighbors = np.fromiter(
            (self.inv_sqrt_degree[n] for n in neighbours),
            dtype=float
        )
        
        if len(inv_deg_neighbors) == 0:
            return None
        
        #scores = self.beta * inv_deg_node * inv_deg_neighbors
        scores = inv_deg_node * inv_deg_neighbors

        den = scores.sum()
        if den == 0 or np.isnan(den):
            return None

        return scores / den        

    def exp(self, x):
        exp_x = np.exp(x) 
        return exp_x

    def temporal(self, t_0, times, is_forward, lamb=1):
        # vizinhos que sao cercanos em tempo sao assinados com a maior probabilidade
        if is_forward:      
            diff = np.array(times) - t_0 # t_{i+1} > t - 2 3 4 5 6 ...
        else:      
            diff = t_0 - np.array(times) # t > t_{i-1} - 6 5 4 3 2 1 ...

        diff = -lamb*diff
        
        return np.exp(diff) # nao softmax porque calcula os padroes struturais 
    
    def spatial_normalization(self, degree, degrees, beta=1):
    
        scores = []
        for k in degrees:
            score = beta / np.sqrt(degree*k)
            scores.append(score)

        scores = np.array(scores)

        #probs = scores/scores.sum()

        return scores
    
    def _temporal_biases(self, time, times, bias_type, is_forward, lamb):
        if bias_type is None:
            # default to uniform random sampling
            return None

        # time is None indicates we should obtain the minimum available time for t_0
        if is_forward:
            t_0 = time if time is not None else min(times)
        else:
            t_0 = time if time is not None else max(times)           

        if bias_type == "temporal":
            # exponential decay bias needs to be reversed if looking backwards in time
            # By default is_forward is True
            return self._exp_biases(t_0, times, decay=is_forward, lamb=lamb)
        else:
            raise ValueError("Unsupported bias type")

    def _sample(self, n, biases):
        '''
          retorna uma amostra, aleatoria a partir de um vetor de biases
          n: numero de tempos   
          biases: probabilidades
          np_rs: numpy random state
        '''
        
        if biases is not None:
            assert len(biases) == n
            # passa um random state global e todas as probabilidades
            return naive_weighted_choices(biases)

        else:
          # gera um numero entre n
            #return np_rs.choice(n)
            return np.random.choice(n)#from stellargraph import StellarGraph
    

    def _get_degree(self, node):
        #nodes = list(graph.nodes())
        #idx = int(nodes.index(node))
        return self.mapping_degrees.get(node)

    def _get_degree_by_time(self, node, time):

        #neighbors = self.graph.neighbor_arrays(node)
        #print(f"Debug: node: {node}, size:{len(neighbors)}")
        #result = [self._get_degree(n) for n in neighbors]

        # return number of nodes for each node taking time 
        # aqui vai sair error se encontra valores menores que time , result vai ser uma lista vazia
  
        neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)
        neighbours = neighbours[times > time]

        return neighbours.size

    def _get_degree_neighbors(self, neighbors, time=None):

        #neighbors = self.graph.neighbor_arrays(node)
        #print(f"Debug: size neighbors:{len(neighbors)}")

        result = []
        if time is None:
            result = [self._get_degree(n) for n in neighbors]
        else:
            result = [self._get_degree_by_time(n, time) for n in neighbors]

        return result

    def _step(self, node, time, bias_type, is_forward, debug):
        """
        Perform 1 temporal step from a node. Returns None if a dead-end is reached.

        """
        neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)
        neighbors = self.graph.neighbors(node)
        
        #print(f"Debug neighoburs len: {len(neighbours)}")
        #print(type(times), type(neighbours), type(time))
        #print(f"Debug1: node: {node}, neighbors: {neighbours}, times: {times}, time: {time}")

        if is_forward:
            mask = times >= time
        else:
            mask = times <= time

        neighbours = neighbours[mask]
        times = times[mask]
        
        #print(f"Debug2: node: {node}, neighbors: {neighbours}, times: {times}, time: {time}")

        biases = None
        degrees = None
        
        if len(neighbours) == 0:
            return None

        if bias_type == "temporal":
            
            biases = self._temporal_biases(time, times, bias_type, is_forward=is_forward, lamb=self.lamb)

        elif bias_type == "spatial":
            
            # vuelvo a calcular os neighbors independiente do tempo ja que é caminhada spacial e nao considera o tempo
            neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)

            # grau dos vizinhos
            degrees = self._get_degree_neighbors(neighbours)

            if len(degrees) == 0:
                return None
            
            biases = self._spatial_biases(degrees, beta=self.beta)
        elif bias_type == "spatial_hub":
            
            # vuelvo a calcular os neighbors independiente do tempo ja que é caminhada spacial e nao considera o tempo
            neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)

            # grau dos vizinhos
            degrees = self._get_degree_neighbors(neighbours)

            if len(degrees) == 0:
                return None
            
            biases = self._spatial_biases_hub(degrees)

        elif bias_type == "spatial_normalization":
            # inclui os vizinhos todos sem considerar o tempo, so quando é spatial
            neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)

            degrees = self._get_degree_neighbors(neighbours)
            degree = self._get_degree(node)
            
            if len(degrees) == 0:
                return None

            #biases = self._spatial_biases_with_normalization(degree, degrees, beta=self.beta)
            biases = self._spatial_biases_with_normalization_optimized(node, neighbours)

        elif bias_type == "spatial_temporal":
        
            # lambda give more priorize to recent edges
            # beta is node influence 
            # is_forward time ordering asceding: 1, 2 , 3 , 4 
            temporal_biases = self._temporal_biases(
                time, times, "temporal",
                is_forward=is_forward, lamb=self.lamb
            )            
            
            # select all neighbors , da problema de tamanho na hora de promediar
            #neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)

            # Se usamos time pode criar problemas
            degrees = self._get_degree_neighbors(neighbours, time)
            #degrees = self._get_degree_neighbors(neighbours)
    
            if len(degrees) == 0:
                return None
                    
            spatial_biases = self._spatial_biases(degrees, beta=self.beta)
            
            #print(f"spatial: {spatial_biases}, temporal :{temp_biases}, shape: {(len(spatial_biases),len(temp_biases))}")
            # print(f"Shape: {(len(spatial_biases), len(temp_biases))}")

            #assert len(temporal_biases) != len(spatial_biases) , "biases prob are diferent size"
                    
            #
            if self.model_id == 'alpha':
                biases = self.model_alpha * np.array(temporal_biases) + (1-self.model_alpha)*(np.array(spatial_biases))
            else:
                biases = (np.array(temporal_biases) + np.array(spatial_biases))/2
                
        elif bias_type == "spatial_temporal_normalization":
            
        
            # lambda give more priorize to recent edges
            # beta is node influence 
            # is_forward time ordering asceding: 1, 2 , 3 , 4 
            temporal_biases = self._temporal_biases(
                time, times, "temporal", 
                is_forward=is_forward, lamb=self.lamb
            )
                        
            # select all neighbors
            #neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)

            # Se usamos time pode criar problemas
            #degrees = self._get_degree_neighbors(neighbours, time)
            #degree = self._get_degree(node)

            #if len(degrees) == 0:
            #    return None
                    
            spatial_biases = self._spatial_biases_with_normalization_optimized(node, neighbours)

            #spatial_biases = self._spatial_biases_with_normalization(degree, degrees, beta=self.beta)
           
            if self.model_id == 'alpha':
                biases = self.model_alpha * np.array(temporal_biases) + (1-self.model_alpha)*(np.array(spatial_biases))
            else:
                biases = (np.array(temporal_biases) + np.array(spatial_biases))/2            

        elif bias_type == "spatial_temporal_normalization_mult":
        
            temporal_biases = self.temporal(time, times, is_forward=is_forward, lamb=self.lamb)
            
            # Se usamos time pode criar problemas
            degrees = self._get_degree_neighbors(neighbours, time)
            #degrees = self._get_degree_neighbors(neighbours)
            degree = self._get_degree(node)

            if len(degrees) == 0:
                return None
                    
            spatial_biases = self.spatial_normalization(degree, degrees, beta=self.beta)
        
            #assert spatial_biases.all() == 0, f"spatial biases should not been zero {spatial_biases}"
            # por algum motivo retorno todos zero que nao deveria, se multiplica com temporal biases vai zerar tudo
            if temporal_biases.all()== 0 and spatial_biases.any()!=0:
                biases = (np.array(spatial_biases))
            elif temporal_biases.any()!= 0 and spatial_biases.all()==0:
                biases = (np.array(temporal_biases))
            elif temporal_biases.any()!= 0 and spatial_biases.any()!=0:
                biases = (np.array(spatial_biases*temporal_biases))

            if biases.any() != 0:
                biases = biases/sum(biases) # RuntimeWarning: invalid value encountered in divide
            else:
                biases = None

        else:
            raise ValueError("Unsupported bias type")

            
        idx = self._sample(len(neighbours), biases)

        #assert idx is not None, "biases should never be all zero"

        if idx is None:
            return None
        
        next_node = neighbours[idx]
        next_time = times[idx]
        
        if debug and bias_type != "temporal":
            print(f"curr time: {time} curr node: {node} neighbors: {neighbours} times {times} degrees: {degrees} biases: {biases} sum: {sum(biases)} next_node: {next_node} next_time: {next_time}")
        elif debug and bias_type == "temporal": 
            print(f"curr time: {time} curr node: {node} neighbors: {neighbours} times {times} biases: {biases} sum: {sum(biases)} next_node: {next_node} next_time: {next_time}")
        
        return next_node, next_time

    def _walk(self, src, dst, t, length, bias_type, is_forward, debug):
        walk = [src, dst] # normalmente é o mesmo nó 
        times = []
        node, time = dst, t
        for _ in range(length - 2):
            # retorna next_node, next_time
            result = self._step(node, time=time, bias_type=bias_type, is_forward=is_forward, debug=debug)

            if result is not None:
                node, time = result
                walk.append(node)
                times.append(time)
            else:
                break
        
        if debug:
            print(f"walk seq: {walk}")

        return walk
    
    def temporal_walk(self, src, dst, t, bias_type, is_forward, debug):
        walk = [src, dst] # normalmente é o mesmo nó 
        times = []
        node, time = dst, t
        for _ in range(self.max_walk_length  - 2):
            # retorna next_node, next_time
            result = self._step(node, time=time, bias_type=bias_type, is_forward=is_forward, debug=debug)

            if result is not None:
                node, time = result
                walk.append(node)
                times.append(time)
            else:
                break
        
        if debug:
            print(f"walk seq: {walk}")

        return walk
