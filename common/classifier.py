from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
#import lightgbm as lgb

import random
import numpy as np
import logging

#logger = logging.getLogger(__name__)


def link_examples_to_features(link_examples, transform_node, binary_operator):
  op_func = (
      operator_func[binary_operator]
      if isinstance(binary_operator, str)
      else binary_operator
  )
  return [
      op_func(transform_node(src), transform_node(dst)) for src, dst in link_examples
  ]

def link_examples_to_features(link_examples, transform_node, binary_operator):
  results = []
  try:
    results = [binary_operator(transform_node(src), transform_node(dst)) for src, dst in link_examples]
  except:
    print(f"(src, dst):{src}, {dst}")
  return results

def link_examples_to_features(link_examples, select_embedding, binary_operator):
  '''
    applica um operador a duas embeddindgs  
  '''

  results = []
  try:
      for src, dst in link_examples:
        results.append(binary_operator(select_embedding(src), select_embedding(dst)))
  except Exception as e:
    print(e)
    print(f"(src, dst):{src}, {dst}")
  return results

def create_classifier(max_iter=500, model='logistic', seed=1):

  if model=='logistic':
    #clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegressionCV(
      Cs=10,                  # busca automática de C
      cv=5,                   # validação cruzada interna
      scoring='roc_auc',      # otimiza ROC-AUC
      max_iter=max_iter,
      solver='lbfgs',
      random_state=seed,
      n_jobs=-1
    )
  elif model == 'mlp':
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=seed
    )
  elif model == 'lgbm':
    #clf = lgb.LGBMClassifier()
    pass

  return Pipeline(steps=[("sc", StandardScaler()), ("clf", clf)])

def labelled_links(positive_examples, negative_examples):

  return (
      positive_examples + negative_examples,
      np.repeat([1, 0], [len(positive_examples), len(negative_examples)]),
  )

def positive_and_negative_links(g, edges):
  pos = list(edges[["source", "target"]].itertuples(index=False))
  neg = sample_negative_examples(g, pos)
  return pos, neg

def sample_negative_examples(g, positive_examples):
  positive_set = set(positive_examples)

  def valid_neg_edge(src, tgt):
      return (
          # no self-loops
          src != tgt
          and
          # neither direction of the edge should be a positive one
          (src, tgt) not in positive_set
          and (tgt, src) not in positive_set
      )

  possible_neg_edges = [
      (src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)
  ]
  
  # selecionamos o mesmo numero
  #assert len(positive_examples) > len(possible_neg_edges), "The quantity of possitive examples is greater than negative edges"
  
  # get k samples from a list in possible_neg_edges
  if len(positive_examples) > len(possible_neg_edges):
    #print(f"Debug  len(possible_neg_edges) = {len(possible_neg_edges)}")
    #print(f"Debug  len(positive_examples) = {len(positive_examples)}")
  
    return random.sample(possible_neg_edges, k=len(possible_neg_edges))
  else:
    return random.sample(possible_neg_edges, k=len(positive_examples))

def create_positive_negative_data(g, edges_train, edges_test):

  pos, neg = positive_and_negative_links(g, edges_train)
  pos_test, neg_test = positive_and_negative_links(g, edges_test)

  link_examples, link_labels = labelled_links(pos, neg)
  link_examples_test, link_labels_test = labelled_links(pos_test, neg_test)

  return link_examples, link_labels, link_examples_test, link_labels_test

def evaluate_roc_auc(clf, link_features, link_labels):
  predicted = clf.predict_proba(link_features)

  # check which class corresponds to positive links
  positive_column = list(clf.classes_).index(1)
  return roc_auc_score(link_labels, predicted[:, positive_column])

def run_link_prediction(binary_operator, model_name, 
                        get_embedding,
                        graph, edges_newer,
                        test_subset, n_runs=5):

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=test_subset, stratify=y, random_state=seed
    #)

    X_train, X_test = train_test_split(edges_newer, test_size=test_subset)
    X_train, y_train, X_test, y_test = create_positive_negative_data(graph, X_train, X_test)

    X = X_train + X_test   
    y = np.concatenate((y_train, y_test), axis=0)

    auc_scores = []
    ap_scores = []
    
    for seed in range(n_runs):
      X_train, X_test, y_train, y_test = train_test_split(
        X, y , test_size=test_subset, stratify=y, random_state=seed
      )

      link_features_train = link_examples_to_features(
        X_train, get_embedding, binary_operator
      )

      link_features_test = link_examples_to_features(
        X_test, get_embedding, binary_operator
      )

      clf = create_classifier(max_iter=700, model=model_name, seed=seed)     
      #treino
      clf.fit(link_features_train, y_train)
      #clf.fit(X_train, y_train)
      
      # check which class corresponds to positive links
      positive_column = list(clf.classes_).index(1)
      y_scores = clf.predict_proba(link_features_test)[:, positive_column]
      
      auc = roc_auc_score(y_test, y_scores)
      ap = average_precision_score(y_test, y_scores)
      
      auc_scores.append(auc)
      ap_scores.append(ap)

      #score = evaluate_roc_auc(clf, link_features_test, link_labels_test)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores, ddof=1)

    mean_ap = np.mean(ap_scores)
    std_ap = np.std(ap_scores, ddof=1)

    mean_auc, std_auc = round(mean_auc*100, 2), round(std_auc*100, 2)
    mean_ap, std_ap = round(mean_ap*100, 2), round(std_ap*100, 2)

    str1 = f"{mean_auc:.2f} ± {std_auc:.2f}"
    str2  = f"{mean_ap:.2f} ± {std_ap:.2f}"
    
    return {
        "Classifier": model_name,
        "Binary-operator": binary_operator,
        "ROC-AUC": str1,
        "ROC-AUC-scores": auc_scores,
        "AP": str2,
        "AP-scores": ap_scores
    }

def operator_hadamard(u, v):
    r = None
    try:
      r = u * v
    except:
      print("Rised an error")
      print(f"u, v: {u}, {v}")

    return r

def operator_l1(u, v):
    r = None
    try:
      r = np.abs(u - v)
    except:
      print("Rised an error")
      print(f"u, v: {u}, {v}")

    return r

def operator_l2(u, v):
    r = None
    try:
      r = (u - v) ** 2
    except:
      print("Rised an error")
      print(f"u, v: {u}, {v}")

    return r

def operator_avg(u, v):
    return (u + v) / 2.0

# -----------------------------
# Time Encoding (sin/cos + log)
# -----------------------------
def time_encoding(delta_t, dim=4):
    """
    delta_t: (batch, 1)
    """
    delta_t = np.log1p(delta_t)  # normalize

    enc = []
    for i in range(dim // 2):
        freq = 1 / (10000 ** (2 * i / dim))
        enc.append(np.sin(delta_t * freq))
        enc.append(np.cos(delta_t * freq))

    return np.concatenate(enc, axis=1)  # (batch, dim)

# -----------------------------
# Temporal-aware operator
# -----------------------------
def temporal_operator(z_u, z_v, delta_t):
    hadamard = z_u * z_v
    l1 = np.abs(z_u - z_v)
    t_feat = time_encoding(delta_t)

    return np.concatenate([hadamard, l1, t_feat], axis=1)
