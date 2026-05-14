import numpy as np
import pandas as pd

def load_dataset(path, sep=" ", ascending=True):

  edges = pd.DataFrame()
  if sep == ' ':
    df = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=["source", "target", "time"],
        usecols=["source", "target", "time"],
    )

    #edges[["source", "target"]] = edges[["source", "target"]].astype(int)
    edges[['source', 'target']] = df[['source', 'target']].astype(str)
    #edges['time'] = df.time.values
    edges[['time']] = df[['time']].astype(float)

  elif sep == ',':

    df = pd.read_csv(path)
    # ,u,i,ts,label,idx

    df[["u", "i"]] = df[["u", "i"]].astype(str)
    #df[['ts', 'label']] = df[['ts', 'label']].astype(int)
    df[['ts']] = df[['ts']].astype(int)
   

    edges['source'] = df.u.values
    edges['target'] = df.i.values
    edges['time'] = df.ts.values

  if ascending:
    edges.sort_values(by='time', ascending=ascending)

  return edges
