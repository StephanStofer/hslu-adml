# Data Classification
Daten werden in zwei Klassen unterteilt. *Numerische* und *Kategorische* Daten. Bei numerische Daten gibt es *stetige* oder *diskrete* Zahlen. Bei Kategorischen sind entweder *ordinal* oder *nominal*. Ordinale haben eine Hierarchie.

## Data Quality Assessment
Daten sind sehr wichtig, der beste ml-Algo nützt nicht, wenn Daten rubbish sind. Mögliche Fehlerquellen:

* Technische Fehler
* Qualität
* schlecht Design
* menschliche Fehler
* Input in Web-Apps (ungeprüfte Eingabefelder)
* Exporte der Daten, falsche Formate - oder Pre-Processing
* Falschangaben durch Benutzer
* Daten haben immer ein Ablaufdaten! (z.B. emailadressen, Adressen)

Ein DQA kommt immer zuerst!
Schützt auch die Reputation gegegnüber Kunden.

### Data Cleaning
Prozess um Fehler in Daten zu beheben (automatisch)/bereinigen. Duplikate entfernen, null-values entfernen, Datenformate ml-friendly aufbereiten (data wrangling). Die Änderungen müssen dokumentiert und versioniert werden, den data provider darüber informieren und die Ursache für die data quality issues untersuchen.

### Approaches to DQA
Dies ist detektiv-Arbeit. Wenn etwas verdächtig erscheint, weitergraben! Die Daten werden überprüft, ob sie vertrauenwürdig sind (plausibilieren).

* Datenquellen und vertrauenwürdigkeit prüfen
* statistische Kennzahlen interpretieren
* daten visualisieren
* Datenranges prüfen (Alter sollte unter 200 sein, Salär > 0, usw.)
* Korrelation zwischen Attributen prüfen (Tachostand und Preis eines Autos)
* Redundanz -> je weniger umso bessere Daten
* Anomalieprüfung in Syntax und Semantik
* NULL Werte und Duplikate erforschen

### Statistische Kennzahlen
Geben uns einen Fingerabruck und erste Plausibilisierung der Daten. Die wichtigsten Kennzahlen sind:

* Mittelwert - *mean* - O(n)
* Modus - *mode* die Zahl die am meisten vorkommt
* Median - *median* - O(n * log n), ist aussagekräftiger

#### Schiefheit
Der Mean, Modus und Median geben Auskunft über die Schiefheit der Daten. Wir haben eine negative, Links-Schiefe *skewness* wenn $mean - mode < 0$, wenn positiv, Rechts-Schiefe $mean - mode > 0$

#### Median
Sortiere Datenreihe. Der Median enthält 50% der Daten. Die Quantile entsprechen je 25%. Die Interquartils Differenz (IQR) entspricht Q3 - Q1.

#### Boxplots
Sehr nützlich zur grafischen Darstellung. *Outliers* sind die Werte die grösser sind als $Q3 + 1.5 * IQR$ respektive $Q1 - 1.5 * IQR$. Minimum bzw. Maximum sind die Werte, die gerade noch ind diese Grenze $1.5 * IQR$ reinpassen.

Wenn viele Outliers müssen Daten genau angeschaut werden, ob sie trotzdem plausibel sind.

#### Five Number Summary of a Data Distribution
In mit Python kann sehr einfach die $Q1, Q2, Q3$, min und max einer Datenreihe ausgegeben werden:

```python
import numpy as np
import pandas as pd

s = pd.Series(np.random.rand(100))
s.describe()
```
Auch Boxplots sind sehr einfach:

```python
import matplotlib.pyplot as plt
plt.boxplot(x = [data.Mileage, data.Price], labels=['Mileage', 'Price'])
```

#### Datenverteilung
Die Verteilung wird mit der Varianz betrachtet, wobei diese *sample variance* die Besselkorrektur ($n-1$) nutzt. Die Standardabweichung entspricht aus der $\sqrt{Var(x)}$

#### Covarianz
Die Covarianz zeigt die Variablität von zwei Datensätzen auf. Ist der Wert positiv, verhalten sich die beiden Daten ähnlich. Ist sie negativ, entsprechend nicht. Ist aber schwierig zu interpretieren, weil sie nicht normiert ist.

#### Covarianzmatrix
Die covarianzmatrix ist sehr wichtig in ML. Sie enthält alle Covarianzen aller Varianzpaare. Die Diagonale kann durch die Varianz von X ersetzt werden.

#### Pearson Korrelation
Covarianz wird durch die Standardabweichung dividiert. Deshalb ergeben sich Werte zwischen 1 (perfekte Korrelation) und -1 (perfect anti-correlation). Damit kann die Datenreihe verglichen werden. Die Korrelationsmatrix kann als Heatmap gut dargestellt werden.

## Replacement Strategies für NULL Values
Kommen immer wieder vor. ML-Algos können selten damit umgehen und müssen bereinigt werden. Je nach Datenumfang sind versch. Verfahren denkbar:

* Zeilen mit NULL Werten löschen
* Fehlende Daten manuell einsetzen
* Globale Konstanten einsetzen (UNKNOWN, $infty$)
* Tendenzen verwenden (Mittelwert für symmetrische Daten, Medien für Schiefedaten)
* Tendenzen auch pro "Klasse" (Eigenschaften) berechnen (z.B Krebskranke und gesunde Patienten)
* Regressionsmodell (sehr aufwändig und ungewohnt in Praxis)

### Feature Engineering
Features entsprechen Spalten. Null-Values können also mit ML erzeugen. Information verfügbar für ML-Algo machen.

### Vector Space Model
Entspricht einem Datenset welches ausser dem Key nur numerische Werte enthält. Kategorische Daten können sehr einfach in nummersiche Daten transformiert werden. Zum Beispiel werden die Farben alle zu Spalten und entsprechende Zugehörigkeit mit 1 bzw. 0 gekennzeichnet. Diese werden als Dummy-Variable bezeichnet.

```python
import pandas as pd
data = pd.read_csv('cars.csv')
data = pd.get_dummies(data)
```
Python code um Daten entsprechend aufzubereiten.

#### Dummy Variable Trap
Mit dem einfügen von Dummy-Variablen muss die *Multikolloniarität* im Auge behalten werden. Wenn $n$-Dummy Variablen erzeugt werden und $n-1$ Spalten alle $0$ sind, wissen wir zu $100%$, dass die $n$te Spalte 1 sein muss. Dies führt zu unterterminierten Matrizen. Die Matrix kann nicht invertiert werden.
Um das zu verhindern, muss eine Spalte gelöscht werden!
Es gibt aber Verfahren, die immun dagegen sind (z.B. Entscheidungsbäume).

## Pandas Profiling
Effizient in drei Zeilen Code!

```python
import pandas_profiling
data = pd.read_csv('cars.csv‘)
data.profile_report()
```

## Fazit
Bei jedem ML-Projekt ist in Data Quality Assessment Pflicht
