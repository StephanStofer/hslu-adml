# Data Classification
Daten werden in zwei Klassen unterteilt. *Numerische* und *kategorische* Daten. Bei numerische Daten gibt es *stetige* oder *diskrete* Zahlen. Bei Kategorischen sind entweder *ordinal* oder *nominal*. Ordinale haben eine Hierarchie.

Trick zum Überprüfen ob daten nummerisch oder kategorisch ist: Ein paar Werte nehmen und daraus den Durchschnitt bilden. Ist der Durchschnitt eine sinnvolle Zahl, sind es nummerische Daten, sonst kategorische.

## Data Quality Assessment
Daten sind sehr wichtig, der beste ml-Algo nützt nicht, wenn Daten rubbish sind. Mögliche Fehlerquellen:

* Technische Fehler
* Qualität
* schlechtes Design
* menschliche Fehler
* Input in Web-Apps (ungeprüfte Eingabefelder)
* Exporte der Daten, falsche Formate - oder Pre-Processing
* Falschangaben durch Benutzer
* Daten haben immer ein Ablaufdatum! (z.B. E-mail Adressen, Adressen)

Ein DQA kommt immer zuerst!
Schützt auch die Reputation gegenüber Kunden.

### Data Cleaning
Prozess um Fehler in Daten zu beheben/bereinigen. Duplikate entfernen, null-values entfernen oder ersetzen, Datenformate ML-friendly aufbereiten (data wrangling). Die Änderungen müssen dokumentiert und versioniert werden, den Data Provider darüber informieren und die Ursache für die Data Quality Issues untersuchen.

### Approaches to DQA (Data Quality Assessment)
Dies ist detektiv-Arbeit. Wenn etwas verdächtig erscheint, weitergraben! Die Daten werden überprüft, ob sie vertrauenswürdig sind (Plausibilitätskontrolle).

* Datenquellen auf Vertrauenswürdigkeit prüfen
* statistische Kennzahlen interpretieren (mean, mode, median, quartiles, variance, standard deviation, covariance, pearson correlation)
* daten visualisieren (zb Box Plot)
* Datenranges prüfen (Alter sollte unter 200 sein, Salär > 0, usw.)
* Korrelation zwischen Attributen prüfen (Tachostand und Preis eines Autos)
* Redundanz -> je weniger umso bessere Daten (Principal Component Analysis)
* Anomalie Prüfung in Syntax und Semantik (sind die Ausschläge erklärbar?)
* NULL Werte und Duplikate erforschen

#### Apply measure even to a small Dataset

* Duplikate löschen
* Redundante Features entfernen (mit Korrelation = 1.0)
* Datum in Jahr, Monat, Tag splitten
* Feature Typ von String in Kategorische ändern

Vorsicht vor dem ersetzen von Null/NA Values. Evtl. besser diese löschen, anstatt mit Median o.ä. füllen.

### Statistische Kennzahlen
Geben uns einen Fingerabruck und erste Plausibilisierung der Daten. Die wichtigsten Kennzahlen sind:

$\mu_X = \frac{1}{n} \sum_{x=1}^{n} x_i$

* Mittelwert - *mean* - O(n)
* Modus - *mode* die Zahl die am meisten vorkommt
* Median - *median* - O(n * log n), ist aussagekräftiger aber auch aufwändiger zu berechnen

#### Schiefheit
Der Mean, Modus und Median geben Auskunft über die Schiefheit der Daten. Wir haben eine negative, Links-Schiefe *skewness* wenn $mean - mode < 0$, wenn positiv, Rechts-Schiefe $mean - mode > 0$

#### Median
Sortiere Datenreihe. Der Median enthält 50% der Daten (die Hälfte der Daten sind kleiner als der Median, die andere Hälfte grösser). Die Quantile entsprechen je 25% (zB. Median der unteren Hälfte des Medians). Die Interquartils Differenz (IQR) entspricht Q3 - Q1.

#### Boxplots
Sehr nützlich zur grafischen Darstellung. *Outliers* sind die Werte die grösser sind als $Q3 + 1.5 * IQR$ respektive $Q1 - 1.5 * IQR$. Minimum bzw. Maximum sind die Werte, die gerade noch ind diese Grenze $1.5 * IQR$ reinpassen.

Wenn viele Outliers müssen Daten genau angeschaut werden, ob sie trotzdem plausibel sind.

#### Five Number Summary of a Data Distribution
Mit Python kann sehr einfach die $Q1, Q2, Q3$, min und max einer Datenreihe ausgegeben werden:

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

$Var(X) = \frac{1}{1-n} \sum_{i=1}^{n}(x_i - \mu_X)^2$

Die Verteilung wird mit der Varianz betrachtet, wobei diese *sample variance* die Bessel Korrelation ($n-1$) nutzt. Die Standardabweichung berechnet sich aus der $\sqrt{Var(x)}$

#### Covarianz

$Cov(X,Y) = \frac{1}{1-n} \sum_{i=1}^{n}(x_i - \mu_X)(y_i - \mu_Y)$

Die Kovarianz zeigt die Variablität von zwei Datensätzen auf (nicht normierte Streuung 2 Datenreihen). Ist der Wert positiv, verhalten sich die beiden Daten ähnlich. Ist sie negativ, entsprechend nicht. Bei Kovarianz von ~0 besteht eine unabhängige Verteilung. Ist aber schwierig zu interpretieren, weil sie nicht normiert ist.

#### Covarianzmatrix
Die Kovarianzmatrix ist sehr wichtig in ML. Sie enthält alle Kovarianzen aller Varianzpaare. Die Diagonale kann durch die Varianz von X ersetzt werden ($Cov(X,X)=Var(X)$).

#### Pearson Korrelation

$$\rho(X,Y) = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} = \frac{Cov(X,Y)}{\sqrt{Var(X)} \sqrt{Var(Y)}}$$

Kovarianz wird durch die Standardabweichung dividiert und somit normiert. Deshalb ergeben sich Werte zwischen 1 (perfekte Korrelation), -1 (perfect anti-correlation) und 0 (statistisch unabhängig). Damit kann die Datenreihe verglichen werden. Die Korrelationsmatrix kann als Heatmap gut dargestellt werden.

## Replacement Strategies für NULL Values
Kommen immer wieder vor. ML-Algorithmen können nicht damit umgehen (Ausnahme Entscheidungsbäume) und müssen bereinigt werden. Je nach Datenumfang sind versch. Verfahren denkbar:

* Zeilen mit NULL Werten löschen (bei vielen verfügbaren Daten)
* Fehlende Daten manuell einsetzen
* Globale Konstanten einsetzen (UNKNOWN, $\infty$)
* Tendenzen verwenden (Mittelwert für symmetrische Daten, Median für Schiefedaten)
* Tendenzen auch pro "Klasse" (Eigenschaften) berechnen (z.B Krebskranke und gesunde Patienten)
* Regressionsmodell (sehr aufwändig und ungewohnt in Praxis)

### Feature Engineering
Features entsprechen Spalten. Null-Values können also mit ML erzeugen. Information verfügbar für ML-Algo machen.

### Vector Space Model
Entspricht einem Datenset welches ausser dem Key nur numerische Werte enthält. Kategorische Daten können sehr einfach in nummerische Daten transformiert werden. Zum Beispiel werden die Farben alle zu Spalten und entsprechende Zugehörigkeit mit 1 bzw. 0 gekennzeichnet. Diese werden als Dummy-Variable bezeichnet.

```python
import pandas as pd
data = pd.read_csv('cars.csv')
data = pd.get_dummies(data)
```
Python code um Daten entsprechend aufzubereiten.

#### Dummy Variable Trap
Mit dem Einfügen von Dummy-Variablen muss die *Multikolloniarität* im Auge behalten werden. Wenn $n$-Dummy Variablen erzeugt werden und $n-1$ Spalten alle $0$ sind, wissen wir zu $100%$, dass die $n$te Spalte 1 sein muss. Dies führt zu nicht invertierbaren Matrizen (under-determined matrix).
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
Bei jedem ML-Projekt ist in Data Quality Assessment Pflicht!
