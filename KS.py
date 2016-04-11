
import pandas as pd
from bins import qcutnew

def KS(labelCol,predictedCol,nBins=10,retBins=False):
    '''Returns the KS Statistic of the given predicted probabilities and labels'''
    assert len(labelCol)==len(predictedCol)
    assert labelCol.ndim==labelCol.ndim
    assert set(labelCol)==set([0,1])
    inpDF=pd.DataFrame({'Label':labelCol,'Predicted':predictedCol})
    inpDF['bin'],binsofcut=qcutnew(inpDF['Predicted'],nBins,retbins=True)
    binsofcut[0]=-1*float('inf')
    binsofcut[-1]=float('inf')
    ksTable=pd.crosstab(inpDF['bin'],inpDF['Label'])
    ksTable['minScore']=binsofcut[:-1]
    ksTable['maxScore']=binsofcut[1:]
    ksTable['cumonespct']=(ksTable[1].cumsum()/ksTable[1].sum())
    ksTable['cumzerospct']=(ksTable[0].cumsum()/ksTable[0].sum())
    ksTable['dvrate']=(ksTable[1]/(ksTable[0]+ksTable[1]))
    ksTable['cumdvrate']=(ksTable[1].cumsum()/(ksTable[1].cumsum()+ksTable[0].cumsum()))
    ksTable['KS']=(ksTable['cumonespct']-ksTable['cumzerospct'])*100.0
    if retBins:
        return max(ksTable.KS,key=abs),ksTable,binsofcut
    else:
        return max(ksTable.KS,key=abs),ksTable

def tableFormatter(inpksTable):
    newTable=inpksTable.copy()
    for col in ['minScore','maxScore','cumonespct','cumzerospct','dvrate','cumdvrate']:
        newTable[col]=newTable[col].apply('{0:.2}'.format)
    newTable['KS']=newTable['KS'].map(lambda x:round(x,1))
    return  newTable

if __name__=='__main__':
    import numpy as np
    np.random.seed(3)
    l=np.random.randint(2,size=100)
    p=np.random.uniform(0,1,100)
    maxks,kstable,bins=KS(l,p,retBins=True)
    print maxks
    print kstable
    #print bins
    #print kstable.cumdvrate[-1]
    #print tableFormatter(kstable)
    print kstable.loc[kstable.KS==maxks,'maxScore'][0]

    datadirjan='/Users/schidara/data/returnpath/ReturnPathJan26/'
    datadir='/Users/schidara/data/returnpath/ReturnPathApr04/'
    trainFile=datadirjan+'features_final_gopla_inst_corr.csv'
    testFile=datadir+'features_final_goldplatinum.csv'

    trainDf=pd.read_csv(trainFile)
    dvMap={'Active':0,'Churn':1}
    trainDf['dv']=trainDf['Stage'].apply(lambda x:dvMap[x])

    maxks,kstable,bins=KS(trainDf['dv'],trainDf['Activity_New_avg'],retBins=True)
    print 'From DataFrame'
    print maxks
    print tableFormatter(kstable)