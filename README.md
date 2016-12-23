# Machine_Learning_for_Chord_Recognition
Team 201612-44 for Big Data Analytics
<br /> Reva Abramson ra2659 CVN

##Overview
Without an ear for music it is difficult to play the songs you like correctly.  This `Spark` and `Python` project uses machine learning to recognize chords.

##Usage

### Getting the Data
* Download annotated (labled with chords) data from isophonics.net (`*.lab` files)
* Find the matching audio file on youtube and convert it to mp3 format
* put all this data in a `music` directory

### Preparing the Feature Vectors
* install `Spark over Hadoop`
* install the Python library librosa `pip install librosa`
* install ffmpeg `brew install ffmpg`
* run `features.py` and direct the output to a file
* the results should be in the format of the file `output.txt` which is what Spark expects

### Creating the Model
* run `train.py` in the same directory as the output.txt file
* a model will be created called `rf.model`

### Predicting with the Model
* run `chordviewer.py` in the same directory as the mp3 song you want to predict and in the same directory as the model `rf.model`
* hardcode the name of the song in chordviewer.py line 41, or alternatively name your song upload.mp3

###Setting up the Web Server and CGI Application
* on a mac run `sudo apachectl start` in order to start up your apache web server at localhost
* navigate to `/private/etc/apache2/httpd.conf` and uncomment `#loadmodule cgi`
* move the model `rf.model` and `chordviewer.py` into `/Library/WebServer/CGI-Executables`
* move `index.html` into `/Library/WebServer/Documents`
* navigate to localhost, upload a song, and the chords will be predicted!









