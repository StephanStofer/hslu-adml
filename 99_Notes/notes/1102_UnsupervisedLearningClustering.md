# Unsupervised Learning: Clustering

Daten ohne Labels, keine Beziehung zu der Zielvariable herstellbar, Strukturen in Daten sichtbar
machen.

## K-Means Clustering

Für einen K-Means müssen die Daten in einem Vector Space Model vorliegen. Oft eingesetzt in
Zielgruppenmarketing (Tracking) oder Auswertung in einem Contest. Skaliert gut.

### Cluster

Bestehen aus Zuordnung der Werte in einem VSM in Gruppen (Kategorien).

### Preparation

Man wählt die Anzahl Clusters $k$, entspricht also einem Hyperparameter. Für jedes $k$ wird zufällig
eine Zahl generiert und in die Datenwolke eingefügt. Diese Punkte entsprechen den Clusterzentren.
Danach:

1. Suche für jeden Punkt den nahsten Cluster (Punkte werden Zentren zugeorndet)
1. Zentrum wird angepasst, mit Durchschnittswert aus allen Punkten neu bestimmt
1. Iteriere durch 1 und 2 bis sich Lage stabilisiert (Zentren nicht mehr verschieben), bis
   konvergiert

![k-Means Algorithm in Math](images/kmeans.png){width=50%}

### Clustering Distortion

Damit kann Qualität gemessen werden. Das **Total Distortion** wird gemessen indem die quadrierte
Distanzen zwischen allen Punkten und dem Clusterzentrum gemessen wird:
$$\sum_{i=1}^n ||x_i - \mu_{c_i}||^2$$

Mit der **Average Distortion** können Clusters über versch. Datensets verglichen werden. Die average
Distortion für jeden Datenpunkt ist:
$$\frac{1}{n}\sum_{i=1}^n ||x_i - \mu_{c_i}||^2$$

#### Convergence and Optimality

$k$-Means **approximiert** die optimale Lösung, **konvergiert immer** findet aber zwingend das
globale sondern nur lokale Minimum. Sklearn ruft intern k-Means 10 mal auf und retourniert das
Clustering mit dem minimalen Distortion.

#### Choose the number of Clusters

Normalerweise wird Anzahl Clusters aus Daten erhoben (z.B. Milch Fettzellen/Verschmutzung, oder in
Marketing k=5 wo Geld für 5 Kampagnien vorhanden ist).

Die **Elbow Method** führt zum idealen $k$. Dabei wird $k$ in einem loop ausgeführt von 1 bis
n-Datenpunkten.

![Recommended number of clusters in $k$-Means](images/elbow_method.png){width=50%}

## Agglomerative Clustering

Hierarchisches Clustering (in Marketing beliebt).

### Algorithmus

Initial entspricht jeder Datenpunkt eineem Cluster. Dann für jedes Clusterpaar die Distanz berechnen
und die zwei nächsten Clusters mergen (zusammenführen). Dies wiederholen bis Stopbedingung.

Hat drei Konfigurationsoptionen:

1. Distanzmass: kann jedes Distanz- oder Similaritätsmass einsetzen
1. Linkage: Verschmelzung der Endpunkte
1. Stopkriterium: Threshold auf Anzahl Clusters, Cluster-Dichte, usw.

### End-Points for Distance Measurement

Es gibt verschiedene Varianten um die Endpunkte (Clusterzentruen) zu berechenen, die für die
Verschmelzung verwendet werden. Ist vom Distanzmass unabhängig.

![Distance Measurement für End-Points](images/endpointdistancemeasurement.png){width=50%}

Wenn ein Cluster besonders dicht ist, sollte er nicht mit einem Cluster zusammengeführt werden,
welcher weniger dicht ist. Variante vier deckt dies ab.

### Dendrogram

Zeigt die Verschmelzungen in jedem Schritt auf. Die Distanzwerte können als Stop-Entscheid mit
Elbow-Method eruieren.

![Dendrogram](images/dendrogram.png){width=50%}

### $k$-Means $\neq$ Agglomerative Clustering

In $k$-Means kann in jedem Schritt ein Daten den Cluster "wechseln". Im Agglomerative Clustering
verbleibt er immer im gleichen.

* die Anzahl Stopkriterium ist für $k$-Means nur die Anzahl. Im AC können *verschiedene* verwendet
  werden.
* $k$-Means garantieret die konveregenz nur mit der Euklidschen Distanz, AC aber mit jedem
  Distanzmass
* dafür $k$-Means gegenüber allen Datengrössen skaliert und AC nur kleineren
* $k$-Means produziert unterschiedliche Clusters je nach Initialisierung der Zentren, AC ist
  hingegen deterministisch
* $k$-Means schwierig zu interpretieren, bei AC kann mit Hilfe des Dendrogram schön aufzeigen