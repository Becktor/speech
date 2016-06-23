import scipy.io
import numpy as np

def batch_to_minibatches(batches):
    batch=batches[:,1:]
    label=batches[:,0]
    labels=np.zeros((9145,6))
    for i in range(9145):
        labels[i][label[i]]=1.0
    return (batch,labels)

class Batches:
    def __init__(self, batch, labels):
        self.batch = batch
        self.labels = labels
        self.index_pos = 0

    def next_batch(self, batch_size):
        start = self.index_pos
        end = self.index_pos + batch_size
        d = self.batch[start:end,:]
        l = self.labels[start:end,:]
        self.index_pos = end
        return (d, l)

if __name__ == '__main__':
    trainSet = scipy.io.loadmat('TrainBatch.mat')['arr']
    testSet = scipy.io.loadmat('TestBatch.mat')['arr']
    m = batch_to_minibatches(trainSet)
    batches = Batches(m[0], m[1])
    for i in range(100):
        batch=batches.next_batch(50)
        print batches.index_pos
        print batch[0].shape
        print batch[1].shape
