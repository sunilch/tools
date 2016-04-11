import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import KS

class KSModel(object):
    '''Model to evaluate the ks and also to predict based on the ks Table'''

    def __init__(self,scoreName,n_bins=10):
        '''
        Intialization
        :param scoreName: Name of the model for which the KS tables are being built
        :param n_bins: Number of bins in the ks Table
        '''
        self.nBins=n_bins
        self.ksTable=None
        self.maxKSAbs=None
        self.maxKS=None
        self.modelName=scoreName
        self.scoreCutoff=None
        self.whichValue=None
        self.binsList=None

    def fit(self,labels,predictions):
        '''
        fits the KS model for the given prediction and the labels
        :param labels:
        :param predictions:
        :return: maxKS and ksTable for the data and prediction
        '''
        self.maxKS,self.ksTable,self.binsList=KS.KS(labels,predictions,retBins=True)
        self.maxKSAbs=abs(self.maxKS)
        baseRate=self.ksTable.cumdvrate[-1]
        self.whichValue=1 if self.maxKS==self.maxKSAbs else 0
        self.scoreCutoff=self.ksTable.loc[self.ksTable.KS==self.maxKS,'maxScore'][0]
        return self.maxKS,self.ksTable

    def formattedTable(self):
        '''
        returns the table in a formatted manner for good display
        :return: formatted Table
        '''
        return KS.tableFormatter(self.ksTable)

    def predict(self,inpScoreList):
        '''
        :param inpScoreList: List of model scores
        :return: list of predictions
        '''
        try:
            inpScoreList=pd.Series(inpScoreList)
        except:
            print 'not in the list format'
            return False
        return list(inpScoreList.map(lambda x:(1-self.whichValue) if x>=self.scoreCutoff else self.whichValue))

    def scorekS(self,valLabels,valPredictions):
        '''
        write the code to build the val KS table
        :param valLabels: Labels for val data
        :param valPredictions: Model score for the val data
        :return: ksTable and maxKS for the val
        '''
        pass

if __name__=='__main__':
    def tests():
        '''
        Defines the tests for the above function that need to pass
        :return: True if all the tests pass
        '''
        np.random.seed(35)
        l=np.random.randint(2,size=100)
        p=np.random.uniform(0,1,100)
        randomOne=KSModel('seed35')
        maxks,kstable=randomOne.fit(l,p)
        print randomOne.formattedTable()
        assert list(randomOne.predict([0.51845,0.2,0.8,-1,2]))==[0,1,0,1,0]
        np.random.seed(3)
        l=np.random.randint(2,size=100)
        p=np.random.uniform(0,1,100)
        randomTwo=KSModel('seed3')
        maxks,kstable=randomTwo.fit(l,p)
        assert list(randomTwo.predict([0.397916,0.2,0.9,1,0.399,0.397]))==[0,0,1,1,1,0]
        print 'tests pass'
        return True

    tests()