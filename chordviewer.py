#!/usr/bin/python -W ignore


import librosa
import os
import numpy
import matplotlib.pyplot as plt
import sys
import cgi, cgitb
cgitb.enable()
import tempfile;

os.environ['SPARK_HOME']="/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7"
sys.path.append("/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7/python")
sys.path.append("/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7/python/lib/py4j-0.10.3-src.zip")
#os.environ['PYTHONPATH']="/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7/python:/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7/python/lib/py4j-0.10.3-src.zip:"
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

print "Content-type: text/html"
print
print "<html><head>"
print "<style>"
print "table, th, td { border: 1px solid black}"
print "</style>"
print "</head><body>"

#/private/etc/apache2/httpd.conf: uncomment #loadmodule cgi
#Library/WebServer/Documents"
#sudo apachectl start (or restart)
#python -m SimpleHTTPServer 8000
#python -W ignore b.py
#brew install ffmpeg, pip install librosa
#from pyspark.mllib.tree import RandomForest, RandomForestModel
#from pyspark.mllib.util import MLUtils

chordList1 = ["N","C:maj","C:min","C#:maj","C#:min","D:maj","D:min","D#:maj","D#:min","E:maj","E:min","F:maj","F:min","F#:maj","F#:min","G:maj","G:min","G#:maj","G#min","A:maj","A:min","A#:maj", "A#:min", "B:maj","B:min"];
chordList2 = ["N","C:maj","C:min","Db:maj","Db:min","D:maj","D:min","Eb:maj","Eb:min","E:maj","E:min","F:maj","F:min","Gb:maj","Gb:min","G:maj","G:min","Ab:maj","Ab:min","A:maj","A:min","Bb:maj","Bb:min","B:maj","B:min"];

newFile = "upload.mp3"
actualName  = "";

try:
	form = cgi.FieldStorage()
	filedata  = form['filename']
	if not filedata.file: 
		print "Error"
		exit()
	actualName = filedata.filename	
	with file(newFile,'w') as outfile:
		outfile.write(filedata.file.read())
except:
	a=1
	
off  = -2
feats = ""

tf = tempfile.NamedTemporaryFile(delete=True)




for i in range(0,30):

	try:
		off = off+2
                dur = 2

                y, sr = librosa.load(newFile,sr=22000,offset=off,duration=dur)
                y_harmonic, y_percussive = librosa.effects.hpss(y)

                chroma = librosa.feature.chroma_stft(y_harmonic,sr)

                if i==-1:
                	librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
                        plt.show()
                        #plt.savefig("temp.png")
                        exit()


                cr,cc = chroma.shape;

                feats +=  "0";

		for j in range(0, cr):
                	feats += " "
                        feats += str(j+1)
                        feats +=  ":"
                        feats += str(numpy.mean(chroma[j]))
		feats += "\n";
	except:
        	one=1
#print(feats)

tf.write(feats);
tf.flush()



sc = SparkContext(appName="PythonRandomForestClassificationExample")

data = MLUtils.loadLibSVMFile(sc, tf.name)
#.collect()
tf.close()
sameModel = RandomForestModel.load(sc, "rf.model")
predictions = sameModel.predict(data.map(lambda x: x.features))
real_and_predicted = data.map(lambda lp: lp.label).zip(predictions)
real_and_predicted=real_and_predicted.collect()
print("<H1>PREDICTED CHORDS FOR " + actualName + "</h1>")
print("<table cellpadding='5px'  style='border: 1px solid black;'><tr><th>START</Th><Th>END</Th><Th>CHORD</Th></Tr>")
j = 0
for value in real_and_predicted:
	print("<tr><td>" + str(j)  + "</td><td>" + str(j+2) + "</td><td>" + chordList1[int(value[1])] + "</td></tr>")
	j += 2
print("</table>")
print("</body></html>")
exit()

