from sklearn.decomposition import PCA
import pandas as pd

def getPCAoneCom(pdcolumn):
    columns = pd.get_dummies(pdcolumn)
    pca = PCA(n_components=1)
    pca.fit(columns)
    return pca.transform(columns).reshape(1, -1)[0]

def convertToNumber(dataObj):
    for col in list(dataObj.columns):
        dataObj[col]=getPCAoneCom(dataObj[col]).astype('float32')
        print(col,'converted')
    return dataObj
  
    

