from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import main
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask.distributed import Client, progress
import joblib

def getLocalDaskCLusterLenov():
    return Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')


def createRamdomForestGrid():
    # numero árboles
    n_estimators = [int(x) for x in np.linspace(start = 200, stop =400 , num = 2)]
    # numero features
    max_features = ['auto', 'sqrt']
    # Número máximo de niveles en el árbol
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_split = [2, 5, 10]
    # Numero mínimo de muestras requeridas para cada nodo
    min_samples_leaf = [1, 2, 4]
    # Metodo de selección de muestras para entrenar cada árbol
    bootstrap = [True, False]# Crear el cuadro
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    return random_grid


def searchBestForest(params,client):
    c=client
    print(c)
    data=dd.read_csv('../ignore/dataPrepared.csv').drop(columns='Unnamed: 0')
    X_train, X_test, y_train,y_test = train_test_split(data.drop(columns='price'), data.price, test_size=0.2)
    [ele.compute() for ele in [X_train, X_test, y_train,y_test]]
        
    with joblib.parallel_backend('dask'):
        model=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
                        max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        bestMod={'model':model,'R2_score':r2_score(y_test, y_pred)}
        contador=1
        print(bestMod)
        for estimators in params['n_estimators']:
            for features in params['max_features']:
                for dep in params['max_depth']:
                    for samples in params['min_samples_split']:
                        for samplesL in params['min_samples_leaf']:
                            for boot in params['bootstrap']:
                                model=RandomForestRegressor(bootstrap=boot, criterion='mse', max_depth=dep,
                                max_features=features, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=samplesL, min_samples_split=samples,
                                min_weight_fraction_leaf=0.0, n_estimators=estimators,
                                n_jobs=None, oob_score=False, random_state=None,
                                verbose=0, warm_start=False)
                                model.fit(X_train,y_train)
                                y_pred=model.predict(X_test)
                                r2=r2_score(y_test, y_pred)
                                if r2>bestMod['R2_score']:
                                    bestMod={'model':model,'R2_score':r2}
                                    print(bestMod)
                                del model
