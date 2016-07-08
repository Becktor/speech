import scipy
import scipy.io
import numpy as np
import batch
#print 'started'
trainSet = scipy.io.loadmat('TrainBatch.mat')['arr']
testSet = scipy.io.loadmat('TestBatch.mat')['arr']
#print 'data loaded'
train = batch.Batch(trainSet)
#print 'train batched'
test = batch.Batch(testSet)
#print 'data batched'
print trainSet.shape
print trainSet[0][0].shape
