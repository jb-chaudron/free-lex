import hddm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1 - Ouverture des fichiers
path = "~/wiki_clust/data_ress/data_Chinese.csv"
df = pd.read_csv(path)


# 2 - Définition des fonctions pour le Pipeline
def low(data):
  data.columns = list(map(str.lower,data.columns))
  data.loc[:,list(data.dtypes == object)] = data.loc[:,list(data.dtypes == object)].applymap(str.lower)
  return data


def hddm_var(data):
  # 1 - On remplace le nom des variables déjà bonnes
  dic_change = {"participant" : "subj_idx"}
  data.columns = [x if not x in dic_change.keys() else dic_change[x] for x in data.columns]

  return data

def aj_phonsem(dat):
  data = dat[["phondist","semdist"]].to_numpy()
  for i in range(data.shape[1]):
      data[:,i] = (data[:,i] - data[:,i].min()) / (data[:,i].max() - data[:,i].min())

  sq = data ** 2
  dist = np.sqrt(data[:,0] + data [:,1])
  print(dist.shape)
  dat["phonsem"] = dist
  return dat

def trans_data(data,col,qt):
  fct = lambda x : [0 if y < x.quantile(qt) else (1 if y > x.quantile(1-qt) else 2) for y in x]


  return data.groupby("subj_idx")[col].transform(fct)

def codage(data,dim="all",qt=0.35):
  #fct = lambda a : [0 if x < a.quantile(0.35) else (1 if x > a.quantile(0.65) else 2) for x in a ]
  #resp = lambda a,b : [0 if x <= data[a].quantile(b) else (1 if x >= data[a].quantile(1-b) else 2) for x in data[a]]
  if dim == "all":
    #data["response"] = resp("phonsem",qt)
    data["response"] = trans_data(data,"phonsem",qt)
  elif dim == "phon":
    data["response"] = trans_data(data,"phondist",qt)
  elif dim == "sem":
    data["response"] = trans_data(data,"semdist",qt)
  else:
    pass

  return data

def flot(data):
  data.loc[:,list(data.dtypes != object)] = data.loc[:,list(data.dtypes != object)].astype(float)
  return data

def sumbsamp(data,col="phonsem"):
  data = data[data[col] < 2]

  return data

# 3 - Préprocessing

df_phon = (df.pipe(low)
    .pipe(hddm_var)
    .pipe(aj_phonsem)
    .pipe(codage,dim="phon")
    .pipe(sumbsamp,col="phondist")
    .pipe(flot))

df_sem = (df.pipe(low)
    .pipe(hddm_var)
    .pipe(aj_phonsem)
    .pipe(codage,dim="sem")
    .pipe(sumbsamp,col="semdist")
    .pipe(flot))

df_all = (df.pipe(low)
    .pipe(hddm_var)
    .pipe(aj_phonsem)
    .pipe(codage,dim="all")
    .pipe(sumbsamp,col="phonsem")
    .pipe(flot))

# 4 - Production du modèle

## 4.1 - Purement Phonologique
m_phon = hddm.HDDM(df_phon)
m_phon.find_starting_values()
m_phon.sample(2000, burn=20,dbname='traces_phon.db', db='pickle')
m_phon.save("model_phon")

## 4.2 - Purement Semantique
m_sem = hddm.HDDM(df_sem)
m_sem.find_starting_values()
m_sem.sample(2000, burn=20,dbname='traces_sem.db', db='pickle')
m_sem.save("model_sem")

## 4.3 - Les deux à la fois
m_all = hddm.HDDM(df_all)
m_all.find_starting_values()
m_all.sample(2000, burn=20,dbname='traces_tt.db', db='pickle')
m_all.save("model_tt")
