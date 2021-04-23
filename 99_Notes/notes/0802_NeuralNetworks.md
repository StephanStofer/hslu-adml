# Neural Networks

Eine der wichtigsten Methoden um ML zu betreiben.

## The Perceptron - An Artificial Neuron

Ein Neuron ist eine Zelle, die elektrisch aktiviert werden kann und mit anderen über *Synapsen*
kommunizieren kann. Ein Neuron besteht aus dem Zellkörper (Soma), *Dendriten* (Rezeptoren) und einem
einzigen *
Axon* (haben Synapsen). Die Dendriten und das Axon zweigen vom Körper ab.

![The Biological Neuron](images/biologicalneuron.png){width=40%}

Neuronen erhalten ein Inputsignal via die Dendriten und senden ein Output-Signal an das Axon. Über
Axon-Terminals kann das Signal via die Synapsen an andere weiter geleitet werden. Der Signalprozess
ist teilweise *elektrisch*, teilweise *chemisch*. Wenn die Spannung in einem kleinen Interval stark
ändert, generiert das Neuron einen *All-or-Northing* elektro-chemischen Impuls welcher *Action
Potential* genannt wird. Dieses Potential travesiert entlang der Axone und *aktiviert* mehr
synapsische Verbindungen. Neuronen-Wechsel ist sehr schnell $<1ms$. Face Recognition benötigt
ungefähr 100 Schritte und das Gehirn kann dies unter 100ms erledigen.

### How do we learn?

Wir lernen indem Synapsen gleichzeitig Impulse senden. Die Verbindung zwischen den Synapsen wird
dadurch stärker. Wenn das passiert, lernt man.

> Synapses that fire together, wire together

Das Gehirn hat verschiedene Areal, wo unterschiedliche Fähigkeiten gelernt werden.

### The Perceptron

Frank Rosenblatt implementierte als erster die Idee ein Model von Neuronen in den Computer zu
übertragen - es wurde Perceptron genannt.

#### The Perceptrion as a Logic Unit

* *Inputs* $X_j$ von vorherigen Axone werden an den Synapsen multipliziert mit den Gewichten $w_j$
* Der *Bias* $b$ macht extra tuning des Aktivierungslayers möglich
* Die *Aktivierung* eines einfachen Perceptrons ist ein Boolean ``True`` wenn ein Threshold erreicht

![The Perceptrion as a Logic Unit](images/perceptronaslogicalunit.png){width=40%}

Die Funktion $g(z)$ wird *Activation Function* genannt. Sie produziert ein *Action Potential* oder *
Activation* $a$ Als Aktivierungsfunktion können z.B. Sigmoid Funktion oder Soft-Max Funktion
verwendet werden. Als Aktivierung wird ein *Hard Threshold* verwendet.

Mit der Sigmoid-Funktion erhalten wir eine [Logistic Regression][] als Aktivierungsfunktion. Damit
kann der Perceptron als linearen Klassifier verwendet werden.

#### Logic using a Single Layer Perceptron

Siehe Wahrheitstabelle.

![Logic using a Single Layer Perceptron](images/singlelayerperceptron.png){width=40%}

#### Linear vs. Non-Linear Models

Eine Single linear Network kann kein XOR implementieren, weil *Non-linear*. Mit einem multi linear
wäre es möglich.

![Linear vs. Non-Linear Models](images/linearnonlinear.png){width=40%}

## Feed Forward Neural Networks

Wenn Daten in einem Netzwerk von links nach rechts fliessen ist es ein *feed-forward network*.

### Add a second Neuron

Schreibweise in Vektor bzw. Matrixform.

![Add a second Neuron](images/addinga2ndneuron.png){width=40%}

### Add a third Neuron

3 Inputs und drei Outputs!!

![Add a third Neuron](images/AddathirdNeuron.png){width=40%}

### Single Layer Neural Network

Generalisiert zu einer $M x N$ und löst bereits jedes linear separierbares Klassifikationsproblem.
Falls $g(z)$ eine logistische Funktion ist, kann jeder Output $a_j$ als *One-vs-All Klassifizierung*
betrachtet werden. Bedeutet dass $a_1$ aus Klasse 1 oder nicht, $a_2$ aus Klasse 2 oder nicht, ...

![Add a third Neuron](images/genericneuralnetwork.png){width=40%}

### Introducing a Hidden Layer

Einem Netzwerk können mehr Neuronen hinzugefügt werden. Bei einem Input-, Hidden und Output-Layer
spricht man von einem *two-layer*-Network. Der Input-Layer wird nicht berücksichtigt, weil da keine
Gewichte berechnet weerden. Die Aussgabe eines Neurons formen den Input des nächsten Neurons -
Simuliert das Gehirn.

#### Hidden Layers Create new Features

![Hidden Layers Create new Features](images/hiddenlayernewfeature.png){width=40%}

#### Universal Approximators

* Es können jede Art von Non-Linear Klassifizierung mit Multilayered Networks gemacht werden
* Hidden Layer lernen neue Features, welche evtl. nicht identifiziert wurden
* Training ist eher langsam, Implementierung aber schnell und einfach
* Multi-Layer ANN sind univesale Approximationen
* Führt aber eher zu overfitting
* Als Gegenmittel [Regularization][]
* sehr Fehlerresistent und Robust

### Deep Learning

Hidden Layers sind die Core-Idea in Deep Learning. Sie lernen ihre Features selber mit den Hidden
Layers. Dazu braucht man kein Expertenwissen um Features zu extrahieren, aber viele viele Daten.

## Neural Network Training by Gradient Descent and Back-Propagation

### Feed Forward Error Calculation

In Supervised Learning schmeissen wir die gelabelten Daten in das Netzwerk und schauen am Ende was
kommt raus. Falls das nicht dem entsprechenden entspricht, werden die Parameter $W$ und $b$ so
angepasst, dass das Resultat stimmt.

### Choosing a Cost Function

Wir brauchen eine Kostenfunktion, welche die Performance des Models misst. Die ist abhängig vom
aktuellen Problem:

* Regression
    - Mean Squared Error
    - Mean Absolute Error
    - Mean Absolute % Error
* Binary Classification
    - Cross Entropy
    - Hinge Loss
    - Squared Hinge Loss
* Multi Classification
    - Multilabel Cross Entropy
    - Kullback Leibler Divergence

### Recap - Feed Forward Neural Net

![Feed Forward Neural Net](images/ffnn.png){width=40%}

![Labeled Data to drive Gradient Descent](images/ffnngd.png){width=40%}

### Back Propagation

Back Propagation mit Hilfe der Kettenregel $\delta^{[3]}$.

![Back Propagation](images/backprogagation.png){width=40%}

![NN all together](images/backpropall.png){width=40%}

![NN all together in Words](images/backpropallwords.png){width=40%}

### Training a Feed Forward Neural Network

![Training a Feed Forward Neural Network](images/trainingnn.png){width=40%}

## Activation Functions and the Soft-Max Classifier

Um den Output zu erhalten benötigen wir eine Aktivierungsfunktion.

### Choosing an Activation Function

Normalerweise möchten wir als Output eine W'keit ($0 < \alpha < 1$). Weil wir Gradient Descent für
das Training verwenden sollte die Aktivierungsfunktion *stetig* und *differierbar* sein. Wenn die
Funktion auf flach ist, verschwinden die Gradienteten. Dies verlangsamt das Training markant und
sollte vermieden werden. Wenn die Funktion *monotonic*, die Fehlerebene in einem single-layer model
ist *konvex*. Eine geeignete ist die Sigmoid, weitere werden nachfolgend behandelt.

#### The Rectified Linear Unit (ReLU)

Ist stückweise linear und wird oft bei Hidden Layers verwendet. Sofern Wert grösser 0, wird sofort
zurückgegeben, sonst 0. Vermindert dass Gradienten verschwinden. Standard Activation Function für
viele Typen von Neural Networks. Modelle mit ReLU sind einfach zu trainieren und erzielen oft
bessere Resultate.

![The Rectified Linear Unit (ReLU)](images/relu.png){width=40%}

Allerdings führen sie zum *dying units* Problem, weil der Gradient für negative Werte immer null ist
und nicht mehr aktiviert wird.

#### The Leaky ReLU

Wird, oder Varianten davon, von den meisten DL Models genutzt, weil verschwinden der Gradienten
nicht auftritt. Führt *non-zero activation* für negative Werte ein. Verhindert Auslöschung des
Gradienten und dying unit Problem. Gute Performance, aber Resultate sind nicht immer konsistent.

![The Leaky ReLU](images/leakyrelu.png){width=40%}

Sind aber generell eine gute Wahl für Hidden Layer Activation.

#### Soft-Max for Multi-Label Classification

Wird für Output bei multi-labels verwendet. Damit kann W'keit bestimmt werden über alle Inputwerte,
egal wie die Aussehen (postiv, negativ, grösser 1, usw.). Softmax modifiziert ein Vektor von $n$
reelen Werten so dass die Summe 1 gibt.

Die Softmax-Funktion ist eine Generalisierung der Logistic Function für mehrere Dimensionen und wird
auch in *multinomial* Logisitc Regression verwendet. Wird normalerweise bei der letzten
Aktivierungsfunktion eines NN verwendet. Die Inputs werden *logits* genannt. Die Funktion *
normalisiert* die Outputs eines Netzwerks in eine Wahrscheinlichkeitsverteilung.

![Soft-Max for Multi-Label Classification](images/softmax.png){width=40%}

![Summary of Activation Functions](images/summaryactivationfunction.png){width=40%}