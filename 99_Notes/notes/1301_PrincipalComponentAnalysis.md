# Principal Component Analysis

Dimensionsreduktion welche für Datenanalyse verwendet werden kann. Lineare Transformation mit
minimalen Informationsverlust. Die Punkte werden als linear Kombination dargestellt.

## Applications of PCA

Wird sehr oft verwendet, zur Visualisierung oder [Data Quality Assessment][] für Redundanzanalyse.
PCA kann angeben wie viel Information verloren wird (wenn angewendet).

![PCA Redundancy reduction](images/pca-redundanz.png){width=50%}

Soll nur visualisiert werden, sollte eher *t-SNE* verwendet werden.

## Principal Component Analysis

Ziel des PCA ist eine Dimensionsreduktion, mit möglichst wenig Informationsverlust, zu erreichen.

### Projections to Base Vectors

Einfache Idee ist das weglöschen einer Achse. Dies erreichen wir mit der Rotation um den Nullpunkt (
siehe IMATH). die Projektion wird im Rechtenwinkel zur Achse gemacht.

![Rotation](images/pca-rotation.png){width=60%}

Information kann man durch Streuung (Varianz) messen. Wenn Streuung gleich null ist, sind alle
Punkte beieinander. Je mehr Varianz beibehalten werden kann, umso weniger Informationen gehen
verloren.

#### Thee more Variance thee more Information

Die gestrichelte Linie in der Abbildung \ref{rotation} ist die Information die verloren geht. Je
grösser der Error, je grösser die Varianz. Die beste Projektion ist die, die den Informationverlust
minimiert oder equivalent die, die die Varianz maximiert.

![Projection Error\label{rotation}](images/pca-variance.png){width=50%}

### Data Redundancy

Ein *mean centred* Feature hat immer den gleichen Wert und somit **null Varianz**. Deshalb kann die
Info gelöscht werden. Eine weitere Art der Redundanz ist, wenn zwei Features *maximale Covarianz*
aufweisen (zum Beipsiel gleiche Daten in unterschiedlicher Einheiten (m/km)).

Features können auch teilweise Redundant sein, wenn sie einen *non-zero covariance* aufweisen.
Idealerweise entfernt die Projektion die Kovarianz, bevor sie nützliche Informationen zerstören
könnte.

#### Strategies for Dimensionality Reduction

1. Redundanz eliminieren
    - voll redundante Features löschen
    - Features kombinieren um Kovarianz zu entfernen
1. Informationen löschen (so wenig wie möglich)
    - Informationsverlust während der Projektion minimieren, indem die verbleibende Varianz minimiert
      wird.

here add some notes..
