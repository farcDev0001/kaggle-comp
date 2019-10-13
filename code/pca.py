from sklearn.decomposition import PCA
import pandas as pd

def getPCAoneCom(pdcolumn,label):
    columns = pd.get_dummies(pdcolumn)
    if label=='cut':
        columns=cutWeight(columns)
    elif label=='clarity':
        columns=clarityWeight(columns)
    elif label=='color':
        columns=colorWeight(columns)

    pca = PCA(n_components=1)
    pca.fit(columns)
    return pca.transform(columns).reshape(1, -1)[0]

def convertToNumber(dataObj):
    for col in list(dataObj.columns):
        dataObj[col]=getPCAoneCom(dataObj[col],col).astype('float32')
        print(col,'converted')
    return dataObj

def cutWeight(dfCutColumns):
    factor=0.2
    for ele in ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']:
        dfCutColumns[ele]=dfCutColumns[ele].apply(lambda n:n*factor)   
        factor+=0.2
    return dfCutColumns

def clarityWeight(dfClarityColumn):
    factor=0.125
    for ele in ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']:
        dfClarityColumn[ele]=dfClarityColumn[ele].apply(lambda n:n*factor)   
        factor+=0.125
    return dfClarityColumn

def colorWeight(dfColorColumn):
    factor=1
    for ele in ['D', 'E', 'F', 'G', 'H', 'I', 'J']:
        dfColorColumn[ele]=dfColorColumn[ele].apply(lambda n:n*factor)   
        factor-=0.125
    return dfColorColumn