import librosa
import numpy
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
#/private/etc/apache2/httpd.conf: uncomment #loadmodule cgi
#Library/WebServer/Documents"
#sudo apachectl start (or restart)
#python -m SimpleHTTPServer 8000
#python -W ignore b.py
#brew install ffmpeg, pip install librosa
#from pyspark.mllib.tree import RandomForest, RandomForestModel
#from pyspark.mllib.util import MLUtils

numpy.set_printoptions(threshold=numpy.nan)

#y, sr = librosa.load("hello.mp3")
#filename = librosa.util.example_audio_file()
#print(filename);




chordList1 = ["N","C:maj","C:min","C#:maj","C#:min","D:maj","D:min","D#:maj","D#:min","E:maj","E:min","F:maj","F:min","F#:maj","F#:min","G:maj","G:min","G#:maj","G#min","A:maj","A:min","A#:maj", "A#:min", "B:maj","B:min"];
chordList2 = ["N","C:maj","C:min","Db:maj","Db:min","D:maj","D:min","Eb:maj","Eb:min","E:maj","E:min","F:maj","F:min","Gb:maj","Gb:min","G:maj","G:min","Ab:maj","Ab:min","A:maj","A:min","Bb:maj","Bb:min","B:maj","B:min"];


import glob
filelist = glob.glob("music/*.lab")
#filelist = ['music/b1.2']

for file in filelist:
	
	file = file.replace(".lab","")
	if '.' in file:
		delim = ' '
	else:
		delim = '\t'
	labellist = [];
	f = open(file+".lab", 'r')
	for line in f:
		#print line.rstrip().split(' ')[2]
        	try:
			labellist.append(line.rstrip().split(delim)[2]);
		except:
			a=1


	arr = []

	arr = genfromtxt(file+".lab", delimiter=delim)

	r, c = arr.shape;

	for i in range(0, r):
        
		try:	
	
			off = arr[i,0]
			dur =arr[i,1]-arr[i,0]
			
			y, sr = librosa.load(file+".ogg",sr=22000,offset=off,duration=dur)
			y_harmonic, y_percussive = librosa.effects.hpss(y)

			chroma = librosa.feature.chroma_stft(y_harmonic,sr)
			
			if i==-1:
				librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
				plt.show()
                		#plt.savefig("temp.png")
				exit()
		
				
            		cr,cc = chroma.shape;
		
			try:
            			sys.stdout.write(str(chordList1.index(labellist[i])));
			except:
				try:
					sys.stdout.write(str(chordList2.index(labellist[i])));
				except:
					try:
						sys.stdout.write(str(chordList1.index(labellist[i]+":maj")));

					except:
						sys.stdout.write(str(chordList2.index(labellist[i]+":maj")));

            		for j in range(0, cr):
				sys.stdout.write(" ")
                		sys.stdout.write(str(j+1))
				sys.stdout.write(":")
                		#print(j+1,end=":")
                		sys.stdout.write(str(numpy.mean(chroma[j])))
				
            		sys.stdout.write("\n");

        	except:
			one=1

