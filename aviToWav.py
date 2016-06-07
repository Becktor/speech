import os
import sys
import subprocess
import scipy as sc


def createFolder ( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)

createFolder("enfDB_Audio")
path='/home/jobe/Git/speech/'
subject = range(1, 42)
feeling = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
sentence = range(1, 6)
progress100 = abs((42*6*6)/100)
progress10 = abs((42*6*6)/10)
devnull = open(os.devnull, 'w')
p=1
j=1
cntr=1
for sub in subject:
    cntr+=1
    name = 's'+str(sub)
    tmpAudio = 'enfDB_Audio/subject '+str(sub)+''
    tmpVideo = 'enfDB/subject '+str(sub)+''
    subPathAudio = path+tmpAudio
    subPathVideo = path+tmpVideo
    createFolder(subPathAudio)
    for feel in feeling:
        cntr+=1
        nameF = name +'_'+ feel[:2]
        tmp = '/'+feel
        feelPathAudio = subPathAudio + tmp
        feelPathVideo = subPathVideo + tmp
        createFolder(feelPathAudio)
        for numb in sentence:
            cntr+=1
            nameN = nameF +'_'+str(numb)
            tmp = '/sentence '+str(numb)+'/'
            numbPathAudio = feelPathAudio + tmp
            numbPathVideo = feelPathVideo+tmp
            createFolder(numbPathAudio)
            try:
                if os.listdir(numbPathVideo) :
                    nameOld=os.listdir(numbPathVideo)[0]
            except:
                pass

            command = "ffmpeg -i \""+numbPathVideo+nameOld+"\" -y -ab 128 -ac 1 -ar 8000 -vn \""+numbPathAudio+nameN+"\".wav"
            subprocess.call(command,shell=True, stdout=devnull, stderr=devnull)
            sample_rates = [0.8, 0.9, 1.1, 1.2]
            for k in range(0,4):
                ar= 8000*sample_rates[k]
               #print numbPathAudio
                cmd = "sox \""+numbPathAudio+nameN+".wav\" -r"+str(ar)+" \""+numbPathAudio+nameN+"_"+str(sample_rates[k])+"\".wav"
                subprocess.call(cmd,shell=True,stdout=devnull, stderr=devnull)

            if(progress100*p<cntr and p<101):
                sys.stdout.write("\r{0}>".format(str(p)+"% "+"="*j))
                sys.stdout.flush()
                p+=1
                if(progress10*j<cntr):
                    j+=1
print "\n[DONE]"
