# Source 1: https://github.com/twmeggs/anfis
# Source 2: https://youtu.be/ubfHsQsLMc8
# Source 3: https://github.com/cbarkinozer/DataScience/tree/main/MachineLearning/anfis
from anfis import anfis_model
from anfis import mfDerivs
from anfis import membershipfunction
import numpy
ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])
X = ts[:,0:2]
Y = ts[:,2]
mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]
mfc = membershipfunction.MemFuncs(mf)
anf = anfis_model.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249: print('test is good')
print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
