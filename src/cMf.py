import os
import sys
import subprocess

path='C:/Users/Jobe/Documents/speech/'
subject = range(1, 42)
feeling = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
sentence = range(1, 6)
for sub in subject:
    tmpAudio = 'enfDB_Audio/subject '+str(sub)
    subPathAudio = path+tmpAudio
    for feel in feeling:
        tmp = '/'+feel
        feelPathAudio = subPathAudio + tmp
        for numb in sentence:
            tmp = '/sentence '+str(numb)+'/'
            numbPathAudio = feelPathAudio + tmp
            if os.listdir(numbPathAudio) == []:
                print "yes"
                print numbPathAudio
