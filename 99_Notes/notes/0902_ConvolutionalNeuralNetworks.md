# Convolutional Neural Networks

Wie neuronale Netze Bildinformationen verarbeiten.

## ImageNet Challenge

öffentliche Bildsammlung mit annotieerten Bilder. 22k Kategorien. Zur Klassifizierung wird
Error-Rate verwendet. Top-5 bedeuetet, Fehlerrate bei einem Bild min. eines von 5 labels entdecken (
Bild mit Auto und Ampel).

## What an Image really is

Bild in drei Kanälen (Matrizen) RGB, mit Zahlen welche die Farbintensität repräsentiert.

## The Naive Approach

Pixelwerte direkt ins Netz ein neuronales Netz eingeben? Unmöglich, weil bereits ein kleines Bild
sehr viele Werte hätte (240x240x3). Dazu wären die gleiche Zahl Neuronen nötig. Ein typischer Hidden
Layoer hat 1024 Neuronen. Man bräuchte mehrere 100 Mio. Gewichte um ein sehr kleiner Bild zu
verarbeiten.

### Tackled the Problem in the Old Times

Durch Feature Engineering konnte Anzahl Features reduziert werden. Durch Kantenextraktion konnten
die Features gewählt werden. Die Peerformance ist aber nur so gut, wie die von menschen gewählten
Filteer sind.

### Invariance to Position, Scaling, Rotation

Bilder könenn skaliert, gedreht, usw. werdene. Ein Klassifier muss also das selbe Resultat erziehlen
ob das Bild Original entspricht oder skalieert wurde.

#### The MNIST Dataset

Beerühmtes Set für die Erkennung der Handschrift.

## Convolutions & Pooling

Werden für Klassifizierung verwendet

### Filter Matrices

Pixelwerte sind im Zusammenhang mit ihren Nachbarn am informativesten. Mit einem filter, werden
diese also zusammen verarbeitet. Mathematisch benötigt diese viele Operationen, weil die Pixel- und
Matrixwerte elementweise multipliziert und addiert werden. Manchmal wird zusätzlich eine
Aktivierungsfunktion genutzt.

#### Images and Filters have different Size

Der Filter wird nun von links nach rechts über das Bild geschoben. Die Versatz der Verschiebung
wird *stride* genannt (wenn grösser, wird Bild kleiner). Das Bild wird aber kleiner. Um das zu
vermeiden, muss *gepaddet*  werden.

#### Effect of Convolutions

Mit Filter kann man Ecken detektieren (z.B. Sobel Filter). Die blenden gewisse Eigenschaften ein,
bzw. aus.

### Convolutional Layers

Ein Conv-Layer wendet viele Filter parallel an. Durch das Anwenden des Filters, schrumpft das Bild
ein wenig. Es werden acht Filter angewendet, die alle 3x3 gross sind. Dadurch erhalten wir $3*3*8=
28$ Gewichte, und das Netz muss die Gewichte lernen. Obwohl viele Inputs und Output-Werte hat es nur
wenige Gewichte in diesem System

![Transformation to Convolutional Layers](images/convlayer.png){width=50%}

#### Conv as Neural Nets

Filtergrösse von 3 weil jeder Layer input von drei hat. Conv1D(1,3) bedeuteet, dass es ein 1D
convolutional Layer mit 1 Filter und Filtergrösse 3 ist. Stride ist auch eins, würde separat als
«Parameter» angegeben.

Die Filter reduzieren die Anzahl Parameter für die nächste Schicht, weil die Gewichte jeweils gleich
sind.

### Pooling

Nachbarspixel sind jeweils sehr ähnlich und auch das convolution von Nachbarspixel würde ähnliche
Pixelwerte ergeben (hohe Redundanz). Wenn wir Objekte entdecken wollen, müssen wir diese aus der
Distanz betrachten - ein Auto können wir nicht anhand wenige Pixel erkennen.

*Pooling* nimmt nun aus z.B. 3x3 Fläche den max, min, oder avg der Pixelnachbar und wendet diese an.
Die Grösse wird über den Pool-Size Parameter bestimmt.

![Pooling](images/pooling.png){width=50%}

Pooling bewirkt einem Zoom out Effekt.

#### Pooling Layers

Pooling dividiert Höhe und Breite durch die Pool Size. Im Bild \ref{poolinglayers} wird die pool
size 2 angewendeet.

![Pooling Layers\label{poolinglayers}](images/poollayer.png){width=50%}

## Model Architectures

![The Big Picture of CNN](images/cnn_bigpicture.png){width=50%}

### Convolutions on RGB Images

Mit einem 3D-Filterl werden die Farbräume zusammengemergt. Der Filter enthält 27 Werte. Der Filter
wird über Bild gelegt und jeder überdeckte Punkt (27 Stk.) wird mit dem Filter multipliziert und
danach aufaddiert. Dies ist der neue Wert. Der Filter wird über das ganze Bild verschoben und
jeweils neu berechnet.

Das Resultat ist ein 2D mit Werten der Korrelationen.

![The Big Picture of CNN](images/convrgb.png){width=50%}

### Computer Vision Disciplines

Nicht nur was sondern auch *wo* ist das Objekt!
Reihenfolge nach Schwierigkeit links oben, rechts oben, links unten, recht unten.

Bei Segmentierung wird jeder Pixelpunkt zugeordnet.

![The Big Picture of CNN](images/cvd.png){width=50%}
