import os
import sys
import subprocess
import scipy.io
import numpy as np
def buildBatch():
    path='/home/Jobe/Git/speech/' ### <-- path to base folder
    subject = range(1, 42)
    feeling = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    sentence = range(1, 6)
    cntr = 0
    feels = ['an', 'di', 'fe', 'ha', 'sa', 'su']
    dataset = np.zeros((13065,4019))
    for sub in subject:
        tmpAudio = 'enfDB_Audio/subject '+str(sub)
        subPathAudio = path+tmpAudio
        for feel in feeling:
            tmp = '/'+feel
            feelPathAudio = subPathAudio + tmp
            for numb in sentence:
                tmp = '/sentence '+str(numb)+'/'
                numbPathAudio = feelPathAudio + tmp

                #try:
                if os.listdir(numbPathAudio) != [] :
                    filt_files = lambda s: not 'wav' in s
                    files = [file for file in os.listdir(numbPathAudio)if filt_files(file)]
                    #print len(files)
                    for f in files :
                        #item = np.zeros((98,41))
                        item = scipy.io.loadmat(numbPathAudio+f)['out']

                        if item.shape == (98,41):
                            for i in range(len(feels)):
                                if feels[i] in f:
                                #new = np.zeros(4019)
                                #print feels[i]
                                #print f
                                    new = np.append(np.array(i), np.hstack(item))
                                    #print new
                                    dataset[cntr,:] = new
                                    cntr+=1

    #print dataset.shape
    train=dataset[:9145,:]
    test=dataset[9145:13065,:]
    #print test
    scipy.io.savemat('TrainBatch.mat', mdict = {'arr':train})
    scipy.io.savemat('TestBatch.mat', mdict = {'arr':test})
    return (train,test)

if __name__ == "__main__":

    (train,test)=buildBatch()
    trainLoad = scipy.io.loadmat('TrainBatch.mat')['arr']
    testLoad = scipy.io.loadmat('TestBatch.mat')['arr']
    print np.all(trainLoad == train)
    print np.all(testLoad == test)
