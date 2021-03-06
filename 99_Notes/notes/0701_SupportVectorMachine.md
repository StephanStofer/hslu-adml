# Support Vector Machine

Skalierbar, komplexe Entscheidungsgrenzen (Decision Boundaries) mit Kernels, insensitiv auf Outliers.
Es geht um Distanzen - welche zu den nahesten Punkten maximiert werden soll. Ist aber linear!

## Motivation

Klassifizierungsprobleme können je nach Random Seed eine unterschiedliche Gerade als Decision
Boundary ergeben.

### Uncertainty in Data

Unsicherheit in Daten. Z.B. ein Messgerät misst nicht immer gleich genau. Sie haben eine gewisse
Unschärfe. Wenn nun eine Entscheidungsgrenze sehr nahe an Punkten anliegt, könnte es sein, dass die
Unschärfe dazu führt, dass ein Punkte die Grenze überschreiten würde.

![Uncertainty in Data](images/uncertainity-in-data.png){width=40%}

### Large Margin Classifier

Die Support Vector State Machine ist anders. Die Optimierung der Gerade wird so gemacht, dass die
Gerade so liegt, dass sie einen möglichst grossen Abstand zu beiden Seiten hat. Der Abstand zeigt
an, wie sicher der Punkt *nicht* auf die andere Seite springt.

Lineare Regressionsverfahren setzen die Gerade in Abhängigkeit der Geraden beliebig.

### Support Vectors

Support Vektoren sind die Punkte im Diagramm, welche am nahesten zur Gerade liegen. Die *Margin* ist
eine symbolische Gerade die parallel zur Decision Boundary verläuft und durch die Support Vektoren
geht.

Weil die Descision Boundary von den Support Vektoren abhängt, ist die SVM insensitiv gegenüber
Outliers. Nur wenn Support Vektoren ändern, ändert die Decision Boundary.

## The Scalar Product

Das Skalar oder Dot-Produkt ist eine Nummer. Dazu werden die Vektoren elementweise miteinander
multipliziert und summiert.

$$\vec{p}\cdot\vec{q}=x_{1}x_{2}+y_{1}y_{2}+z_{1}z_{2}$$

### Scalar Product and Vector Length

Die $L^2$-Norm (siehe [Euclidiean Distance or $L^2$-Norm][]) ist die Wurzel der Quadrate der
Komponenten. Diese aufsummieren und Wurzel ziehen.

$$||\mathbf{x}||=\sqrt{x^2_1+...+x_n^2}$$

Ein Einheitsvektor ist es dann, wenn die Länge 1 ist. Zum normieren, ist der Vektor durch seine
Länge zu dividieren.

### Vector Triangles

Vektoren haben eine Länge und Richtung. Aber keine Position. Deshalb ergibt sich aus $\vec{x}$,
$\vec{y}$ und $\vec{x} -\vec{y}$ ein Dreieck (wenn $0 < \theta < 90°$).

#### The Cosine Formula

Sofern $\vec{x}$ und $\vec{y}$ nicht trivial (nicht null) sind, gilt

$$\vec{x} \cdot \vec{y} = ||\vec{x}|| * ||\vec{y}|| * cos(\theta)$$

![Vector Triangles](images/vector-triangles.png){width=40%}

#### Scalar Product as Projection

Wenn $\vec{y}$ ein Einheitsvektor ist, entspricht das Skalarprodukt $\vec{x} \cdot \vec{y}$ der *
Projektion* von $\vec{x}$ auf $\vec{y}$

![Scalar Product as Projection](images/scalarasprojection.png){width=40%}

#### Perpendicularity Test

Das Skalarprodukt von senkrechten Vektoren ist null. Aufgabe; finde Vektor $y$ damit er senkrecht
auf $x$ steht. Folgende Funktion $x_1*y_1 + x_2*y_2 + x_3*y_3 = 0$ aufstellen und lösen.

## Hyperplanes

Eine Hyperebene ist definiert durch einen Auffahrtspunkt $P$ und einem Richtungsvektor $w$. Wir
fordern von $w$, dass er ein Einheitsvektor entspricht und rechtwinklig zur Ebene steht.

### Hessian Normal Form

![Hessian Normal Form](images/hessiannormalform.png){width=50%}

![Example Hessian Normal Form](images/hessiannormalformexample.png){width=40%}

![Distanz Hessian Normal Form](images/hessiannormalformdistance.png){width=50%}

![Bias and Offset](images/biasandoffset.png){width=50%}

### Shifting the Coordinate System

Vektoren müssen nicht verschoben werden, weil die keine Position haben, sondern nur eine Richtung.

![Shifting the Coordinate System](images/shifitingcoordinatesystem.png){width=40%}

![Distance to Hyperplane Point Q](images/hessiandistancetoq.png){width=40%}

### How to interpret signed Distances

Eine Hyperebene teilt einen Raum in einen postiven und negativen Halbraum (+/-HP - Halfplane). By
Convention zeigt der Vektor $\vec{w}$ immer in die Richtung von -HP! Die Punkte in +HP haben eine
positive Distanz und die Punkte in -HP eine negative Distanz zur Hyperebene.

![Signed Distances](images/SignedDistances.png){width=40%}

![Example Signed Distances](images/SignedDistancesexample.png){width=50%}

## Large Margin Classifier

In der binären Klassifizierung wählen wir für die Labels die Werte 0 und 1. Bei Support Vector
Machines codieren wir die Labels mit -1 und +1. Dies ist nur eine «Namensgebung», wodurch sich aber
eine elegante Formulierung ergibt.

![Label Encoding](images/labelencoding.png){width=40%}

Die Hyperebene teilt lediglich die beiden Punktefamilien. Weil wir aber eine Distanz wollen, führen
wir ein $M=1$ ein und kontrollieren lediglich die Skalierung.

![Introduction the Margin](images/intromargin.png){width=40%}

### Controlling the Margin

Die rote Linie auf der Abbildung \ref{redline} wird mit einer Konstate multipliziert. Damit
vergrössert oder verkleinert sich der Margin. Mit dieser Skalierung von $\vec{w}$ kann die Margin
kontrolliert werden. Die Bedingung bleibt immer gleich, nämlich $+/-1$, jedoch verändern wir so die
«Einheit» des Abstandes (mm, cm, m, km)

![Controlling the Margin\label{redline}](images/controllingmargin.png){width=40%}

### Elegant Problem Formulation

Weil wir im obigen Fall eine Fallunterscheidung haben (if/else), ergäbe sich eine umständliche
Formulierung für die Optimierung. Weil wir die Labels +1 und -1 verwenden, können wir die Labels in
die Formel reinmultiplizieren und erreichen damit eine allgemein gültige Bedigungung (Constraint).

![Elegant Problem Formulation\label{class}](images/class1-1.png){width=40%}

### How Big is the Margin

Die Distanz können wir aus $-b$ aus der hessischen Normanform ablesen. Wenn wir die Gleichung aber
skalieren, entspricht sie nicht mehr der Hessischen Normalenform. Um dies zu umgehen kann der Term
normiert werden. Da wir zwei Distanzen in beide Richtungen haben, ergibt sich die gesamte Margin aus
$$\frac{-(b-1)}{||\vec{w}||}+\frac{(b+1)}{||\vec{w}||}=\frac{2}{||\vec{w}||}$$

### Primal Optimization Problem

Forderungen

1. Datenpunkte müssen auf richter Seite klassifiziert sein
1. Margin muss maximiert sein

Anstatt $\frac{2}{||w||}$ zu maximieren, können wir auch einfach $||w||$ minimieren, oder noch
besser $\frac{1}{2}*||w||^2 = \frac{1}{2}w \dot w$. Es ist eine Minimierung unter Constraints,
welches ein quadratisches Optimierungsproblem ist.

Diese Variante ist anfällig auf Overfitting, weil eine Linie gefunden werden muss.

Entspricht einem *Hard Margin Classifier* mit welchem ein Binäres Klassifizierungsproblem gelöst
werden kann.

![Hard Margin Classifier](images/hardmarginclassifier.png){width=40%}

## Soft Margin Classifier

Das Problem der Hard Margin Classifier ist, dass jeder Punkt klar einer KLasse zugeordnet werden
muss. Das Model wird overfitted. Der Soft Margin Classifier erlaubt eine weichere Linie, wo auch
«Übertretungen» erlaubt sind.

### Outlier Sensitivity

Hard Margin Classifiers sind sehr sensitiv gegen Outliers. In der Abbildung \ref{sensitivityoutlier}
sehen wir, dass der graue neue Punkt eher zur Mehrheit der roten Punkte passen würde. Aber wegen der
harten Grenze zu blau zugeordnet wird. Das Model ist also overfittet und generalisiert zu wenig auf
unseen Data.

![Outlier Sensitivity\label{sensitivityoutlier}](images/outliersensitivity.png){width=40%}

### Soft Margin Classifier

Ein *Soft Margin Classifier* erlaubt die Missklassifizierung von Trainingsdaten. Es ist ein
Trade-off zwischen besserer Generalisierung auf *unseen* Data und Klassifizierungsfehler in den
Trainingsdaten. Es muss die Überlegung gemacht werden, noch mehr Punkte zu ignorieren. Dazu
Cross-Validation nutzen!

![Soft Margin Classifier\label{classificationerror}](images/classificationerror.png){width=40%}

### Slack Variables

Jeder Datenpunkt erhält eine Schlupfvariable $\varepsilon$, welche ein Mass ist, wie viel der Punkt
von der Margin abweichen darf. Mit dem Hyperparameter $C$ in \ref{slackvariable} kann ich steuern,
ob ein eher engerer Margin schlimmer ist, oder mehr Verletzungen in den Trainingsdaten erlaubt sind.
Evaluation von $C$ via Cross-Validation.

![Slack Variables\label{slackvariable}](images/slackvariable.png){width=60%}

### Regularization Parameter

Mit Hilfe vom $C$, siehe \ref{slackvariable}, können wir overfitting entgegenwirken.

* je kleiner $C$, desto weicher und eher dürfen Punkte den Margin überschreiten $\rightarrow$ führt
  zu grosser Margin
* je grösser $C$, desto harter ist die Verletung $\rightarrow$ und führt zu kleiner Margin
* mit $C=\infty$ werden keine Verletzungen akzeptiert und der Soft wird wieder zu einem Hard Margin
  Classifier

Es ist immer noch ein quadratisches Optimierungsproblem mit eindeutigen Minimum. Es ist ein
Trade-Off zwischen Trainingsfehler und Margin, kontrollierbar via Hyper-Parameter $C$. Bester
Trade-Off via Cross-Validation für $C$ finden.

![Unconstrained Optimization Problem](images/unconstrainedoptimizationproblem.png){width=40%}

## Kernel

Einer der coolsten Tricks in ML und erlauben komplexere Entscheidungsgrenzen. Transformieren Daten
in den höheren dimensionalen Raum.

### Not Linearly Separable

Wenn Daten nicht klar in zwei Gruppen unterteilt werden kann (Klassen sind kreisförmig), können
Daten in höheren dimensionalen Raum projeziert werden. Durch quadrieren werde Daten projieziert. Das
ist die Idee des Kernels.

### What is a Kernel

Berechnet Similaritätswert für zwei Datenpunkte, die in einem höheren dimensionalen Raum projieziert
wurden. Die Punkte definieren *implizit* ein Mapping in den hochdimensionalen Raum. Die Projektion
wird nur hypthetisch gemacht. Bzw. geben den Wert zurück ohne die Projektion effektiv auszuführen.
Häufig verwendet:

* Lineare Kernels
* Polynomial Kernels
* RBF Kernels
* Sigmoid Kernels

#### Lineare Kernel

Spezialform des Polynomial Kernels, nämlich gewöhnliches Skalarprodukt

#### Polynomial Kernel

Dieser Kernel bereechnet aus zwei Datenpunkten (Vektoren) aus dem Tiefdimensionalen Raum das
Skalarprodukt, addiert Konstante $r$ und rechnet das Ganze hoch $d$ $$(\vec{x} \cdot \vec{y} + r)
^d$$

* $r$ = Koeffizient
* $d$ = Grad des Polynoms

![Example Polynomial Kernel](images/polykernel.png){width=60%}

![2. Example Polynomial Kernel](images/polykernel2.png){width=60%}

![3. Example Polynomial Kernel](images/polykernel3.png){width=60%}

Output des Kernel ist das Skalarprodukt der beiden Koordinaten

#### RBF Kernel

Radial Basis Function (RBF) oder Gaussian Kernel. Entspricht der Projektion in einen unendlich
dimensionalen Raum. Verhält sich ähnlich wie ein gewichtetes Nearest Neighbor Model. Je näher die
Distanz zu einem Trainingspunkt, desto mehr Einfluss auf die Klassifizierung. $$exp(-\gamma * ||
\vec{x}-\vec{y}||^2)$$

Das Gamma entspricht einem exponentiellen Decay (Verfall) der
skalierten [Euclidiean Distance or $L^2$-Norm][] zwischen $\vec{x} \text{und} \vec{y}$

![Example RBF Kernel\label{rbfkernel}](images/rbfkernel.png){width=60%}

$f$ in \ref{rbfkernel} entspricht der Ableitung. Die Taylorentwicklung ist eine unendliche lange
Approximation von $e^{xy}$

## SVM with Kernel

Obwohl wir es nur für die Hard Margin anschauen, funktioniert es sehr ähnlich im Soft Margin
Optimierungsproblem.

Problem ist, den Kernel in die Optimierungsformel einzubetten. Eine solche Representation wird mit
der *Lagrange Methode* erreicht.

### Lagrange Methode

Siehe auch IMATH-stuff.

![Lagrange Method](images/lagrangemethod.png){width=60%}

### Lagrange Transformation

Wir berechnen die partiellen Ableitungen mit $w$, $b$ und $a_i$. Eine Einschränkung ist, dass alle
partiellen Ableitungen gleich null sind.

![Lagrange Transformation](images/lagrangetransformation.png){width=60%}

![Partial Derivatives of Lagrange Formulation](images/partiallagrange.png){width=60%}

![Towards the Dual Problem Formulation](images/partiallagrange2.png){width=60%}

![Dual Problem Formulation\label{kernelsvm}](images/partiallagrange3.png){width=60%}

### Kernel Hard-Margin SVM

Als Skalarprodukt in \ref{kernelsvm} kann nun ein beliebiger Kernel eingefügt werden. Der Kernel via
Hyperparameter Optimierung eruieren. Anstatt das primale Problem zu minimieren, *maximieren* wir
die Lagrange-Dual-Funktion. Zur Klassifizierungszeit verwenden wir

![Kernel Hard-Margin SVM](images/kernelhardmarginsvm.png){width=60%}

$K$ entspricht einem Kernel. Das Resultat ist eine Zahl (Similarität im höheren Raum).

Zur Klassifizierungszeit wird der Karsupel nicht mehr benötigt.

![Kernel Hard-Margin SVM at Classification Time](images/classtime.png){width=60%}

Trainingssamples werden nicht mehr benötigt.

## Summary

* SVM sind bei der Klassifizierung ziemlich effizient, weil fast nur ein Skalarprodukt berechnet werden muss
* Sind stabil gegenüber Outliers
* Kernels projizieren zwei Datenpunkte in einen höheren Dimensionalen Raum und geben Similarität zurück, ohne die Projektion effektiv auszuführen

## API Check

`sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,...)`

* C is the regularization parameter
* kernel choice (linear, poly, rbf, ...)
* degree is the degree d of polynomial kernel
* gamma is the RBF kernel coefficient
* coef0 is the coefficient r of polynomial kernel

[https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html][siehe online]
