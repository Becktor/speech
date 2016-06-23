import os
import sys
import subprocess

path='/home/Jobe/Git/speech/'
subject = range(1, 42)
feeling = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
sentence = range(1, 6)
for sub in subject:
    tmpAudio = 'enfDB/subject '+str(sub)
    subPathAudio = path+tmpAudio
    for feel in feeling:
        tmp = '/'+feel
        feelPathAudio = subPathAudio + tmp
        for numb in sentence:
            tmp = '/sentence '+str(numb)+'/'
            numbPathAudio = feelPathAudio + tmp
            try:
                if os.listdir(numbPathAudio)[2] == "Thumbs.db":
                    os.remove(numbPathAudio+"Thumbs.db")
                    print "yes"
                    print numbPathAudio
            except:
                #print numbPathAudio
                pass
