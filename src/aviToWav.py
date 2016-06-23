import os
import sys
import subprocess
import scipy as sc
import preProcess
import progress

def createFolder ( directory ):
    if not os.path.exists(directory):
        os.makedirs(directory)

def avi_to_wav(oldpath, newpath, sr ,nName, oName, time):
    command1 = "ffmpeg -i \""+oldpath+oName+"\" -y -ab 128k -ac 1 -ar "
    command2 = str(sr)+" -vn \""+newpath+nName+"\""
    command = command1+command2

    subprocess.call(command,shell=True, stdout=devnull, stderr=devnull)
    #subprocess.call(command,shell=True)
    #preProcess.process(newpath+nName,1 , newpath+nName)
    try:
        preProcess.process(newpath+nName,time , newpath+nName)
    except:
        #print except
        pass

    sample_rates = [0.8, 0.9, 1.1, 1.2]
    for k in range(0,4):
        ar= sr*sample_rates[k]
        #print numbPathAudio
        cmd1 = "sox \"" + newpath+nName+"\" -r" + str(ar) + " \""
        cmd2 = newpath + str(sample_rates[k])+"_"+nName+"\""
        cmd = cmd1 + cmd2
        subprocess.call(cmd,shell=True,stdout=devnull, stderr=devnull)
        #subprocess.call(cmd,shell=True)
        try:
            preProcess.process(newpath + str(sample_rates[k])+"_"+nName ,time , newpath + str(sample_rates[k])+"_"+nName)
        except:
            pass



if __name__ == "__main__":
    createFolder("enfDB_Audio")
    path='/home/Jobe/Git/speech/'
    feeling = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    sentence = range(1, 6)
    subject = range(1, 42)
    max = 42 * 6 * 6
    pBar=progress.progressBar(max)
    devnull = open(os.devnull, 'w')
    sample_rate=8000
    cntr=1
    for sub in subject:
        cntr += 1
        name = 's' + str(sub)
        tmpAudio = 'enfDB_Audio/subject ' + str(sub)
        tmpVideo = 'enfDB/subject ' + str(sub)
        subPathAudio = path+tmpAudio
        subPathVideo = path+tmpVideo
        wav_length = 3
        createFolder(subPathAudio)
        for feel in feeling:
            cntr += 1
            nameF = name + '_' + feel[:2]
            tmp = '/' + feel
            feelPathAudio = subPathAudio + tmp
            feelPathVideo = subPathVideo + tmp
            createFolder(feelPathAudio)
            for numb in sentence:
                cntr += 1
                nameN = nameF + '_' + str(numb) + '.wav'
                tmp = '/sentence ' + str(numb) + '/'
                numbPathAudio = feelPathAudio + tmp
                numbPathVideo = feelPathVideo + tmp
                createFolder(numbPathAudio)
                try:
                    if os.listdir(numbPathVideo) :
                        nameOld=os.listdir(numbPathVideo)[0]
                except:
                    pass

                wavs=avi_to_wav(numbPathVideo, numbPathAudio,
                                sample_rate, nameN, nameOld, wav_length)

                pBar.p100(cntr)
    print "\n[DONE]"
