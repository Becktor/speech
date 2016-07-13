import scipy.io
import numpy as np

def batch_to_minibatches(batches):
    size = batches.shape[0]
    batch=batches[:,1:]
    label=batches[:,0]
    labels=np.zeros((size,6))
    for i in range(size):
        labels[i][label[i]]=1.0
    return (batch,labels)

class Batch:
    def __init__(self, dataset):
        self._spec, self._labels = batch_to_minibatches(dataset)
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = self._spec.shape[0]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._spec = self._spec[perm]
            self._labels = self._labels[perm]
            #print self._spec.shape
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._spec[start:end], self._labels[start:end]

    @property
    def num_examples(self):
        return self._num_examples
    @property
    def images(self):
        return self._spec

    @property
    def labels(self):
        return self._labels

if __name__ == '__main__':
    trainSet = scipy.io.loadmat('TrainBatch.mat')['arr']
    testSet = scipy.io.loadmat('TestBatch.mat')['arr']


    batches = Batch(testSet)
    #for i in range(10000):
      #  batch=batches.next_batch(10)
       # print batch[0][0]
        #print batches.num_examples
       # perm = np.arange(batches.num_examples)
        #print perm
        #np.random.shuffle(perm)
        #print batches.index_pos
        #print batch[0].shape
        #print batch[1].shape
