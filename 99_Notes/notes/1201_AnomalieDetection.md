# Anomaly or Outlier Detection

In unsupervised Learning.

## Types of Outliers

Welche Typen von Anomalien oder Outliers können wir erkennen. Es geht nicht nur um einzelne
Datenpunkte, sondern auch um Anomaliegruppen.

Beispiele von Outliers

* Analyse TCP-Traffic zum erkennen von gehackten Computer
* Analysee MRI Bild für Krebserkennung
* Kreditkartendatenanalyse
* Trading Transaction
* Social Media, Trends sind zuerst Outliers
* Outlier Investigation im [Data Quality Assessment][]
* Outlier entfernen vor dem supervised Learning

### Drei Typen von Outliers

* Globale Outliers: unterscheiden sich markant vom Rest des Datensets (zb. Pattern von Kreditkarten
  Transaktionen)
* Contextbedingte Outliers: unterscheiden sich im Kontext (zb. Temperatur 25° im Dezember, Mai wäre
  aber normal)
* Kollektiven Outliers: Punktwolken, die als Gruppe eine Anomalie ist (zb. Bestellung: Wenn eine
  Bestellung eine Liefereverzögerung hat, ist es keine Anomalie, aber wenn 100 eine haben schon).

## Outlier Detection with ML

Mit Supervised Learning sind Outlier sehr schwierig erkennbar, weil diese auch sehr rar sind. Daten
wären auch sehr unausgewogen (disbalanced). Weshalb mit Unsupervised Learning gearbeitet wird.

### Statistical Methods

Mit Hilfe der Standardabweichung weiss man wo die Daten (99.7% davon) liegen sollten. Ist nun ein
Datenpunkt nicht innerhalb der Standardabweichung ist es ein Outlier.

### Proximity-Based Methods

Es gibt Distanz- oder Dichte-basierte Methoden. Mit Distanzbasierten Methoden können nur globale
Outliers identifiziert werden.

![Pseudo-Code Distance-Based Outliers](images/dboutliers.png){width=50%}

![Density-Based Outliers](images/diboutliers.png){width=70%}

#### Local Outlier Factor (LOF)

Die Zahl sagt aus, ob ein Datenpunkt ein Outlier ist oder nicht. $LOF(x) \approx 1$ ist kein
Outlier, wenn deutlich grösser, dann ist es ein Outlier. Kleiner 1 ist es im Density-Case ein *
inlier*

#### $k$-Distance

Die Distanz (Radius) welche k-Nachbarn einschliesst.

#### Reachability Distance

Die Reachability Distance ist das Maximum der Distanz von zwei Punkten und die k-Distanz des zweiten
Punktes. $$\text{reachability-distance}_k(A,B)=max{k-distance(B),d(A,B)}$$

Lokale Erreichbarkeit ist nicht symmetrisch! Dem Punkt 1 kann der Punkt 2 innerhalb der $k$-Distanz
sein, Punkt 1 muss aber nicht in $k$-Distanz von Punkt 2 liegen.

![Reachability Distance](images/reachdistance.png){width=70%}

### Clustering-Based Methods

Wichtig mit Datensatz ohne Outlier starten, weil Clustering sehr anfällig auf Outliers ist.

![Clustering-Based Methods](images/clusteringBased.png){width=70%}
