import pandas as pd
from pca import convertToNumber
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



def getData():
    return pd.read_csv('../ignore/data.csv').drop(columns=['x','y','z',])

def getDataTest():
    return pd.read_csv('../ignore/test.csv').drop(columns=['id','x','y','z',])

def getDataForTraining(data):
    nDf= convertToNumber(data[['cut','color','clarity']])
    for col in nDf.columns:
        data[col]=nDf[col]
    data.carat=data.carat.apply(lambda n:n*10**5)

    return data

def trainScoreNpredict(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='price'), data.price, test_size=0.1)
    """el modelo que inicializo lo saqué con el código del archivo forest.py, es un código durete
    a nivel de recursos y además no se debería usar en un ordenador con menos de 8 hilos y menos de 12gb de
    ram"""
    model= RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=110,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=10,
                      min_weight_fraction_leaf=0.0, n_estimators=333,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    return {'model':model,'R2_score':r2_score(y_test, y_pred)}

def main():
    """Este es el método principal, aquí no uso dask porque solo pruebo 1 modelo (el inicializado en la función
    anterior,este método se puede ejecutar en cualquier pc desde 6gb de ram)"""
    data=getDataForTraining(getData())
    dictRes=trainScoreNpredict(data)
    print(dictRes)
    test= getDataForTraining(getDataTest())
    submissionPred=dictRes['model'].predict(test)
    subir=pd.DataFrame(submissionPred)
    subir.columns=['price']
    subir.index.name='id'
    subir.price=subir.price.apply(lambda p:int(round(p,0)))
    subir.to_csv('../ignore/sample_submission.csv')
    
    
