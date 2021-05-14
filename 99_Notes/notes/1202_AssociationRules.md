# Association Rules for Market Basekt Analysis

Transaktionsdaten sind Daten wann welcher Kunde was gekauft hat. Dazu erwarten wir Transaktionsdaten
in Binären Form.

![Transaction in binary Form](images/transactiontable.png){width=50%}

## Association Rules

Associationsregel implizieren X sodass Y. Dies wird dann auf ein beliebiges Set angewendet.

### Support of a Set of Items

Support ist ein Teil einer Tranaktion welche ein spezifisches Set an Items beinhaltet. Misst wie oft
Items zusammeen gekauft werden. $$support({i_1,...,i_n})=\frac{\#purchases of
\{i_1,...,i_n\}}{\#transactions}$$

![Support of Items](images/support.png){width=50%}

### Support of an Association Rule

Weil wir Regeln benötigen müssen wir den Support leicht anpassen.

![Support of an Association Rule](images/supportofar.png){width=50%}

#### Interpretation of Support

Messen wie oft ein Set in den Daten auftauchen. Regeln mit kleine Support sind weniger intressant
aus wirtschaftlichen Gründen. Gute Verbindung zwischen Regeln und Support sind hoch: Support =
Interestingness

### Confidence of an Association Rule

Vertrauensmass wie gut die Rolle ist. $confidence(X\rightarrow Y) = \frac{X \cup Y}{support(X)}$

### Interpretation of Confidence

Entspricht auch einer bedingten Wahrscheinlichkeit $p(Y|X)$

Confidence ist ein Gütemass einer Regel. Support misst wie interessant und Confidence wie gut eine
Regel ist. Confidence = Trustworthiness

## Apriori Algorithm

Generiert Regeln die ein gewisses Mass an Support und Confidence erreichen.

![Apriori Algorithm](images/apriori1.png){width=60%}

Der Diamant wird traversiert und da wo der Support nicht erreicht wird, wird abgeschnitten.

![Apriori Algorithm PseudoCode](images/aprioripseudo.png){width=50%}

![Apriori Algorithm Step 2](images/apriori2.png){width=60%}

## Reflection

Wir haben also eine Regel die den Support und deren Güte bewertet. Wird ein Produkt sehr oft
gekauft (Banana) und ein anderes (Anchovy Paste) sehr wenig, kann die Banane trotzdem überwiegen und
den Score des Produktpaares erhöhen. Um das zu korrigiereeen gibt es einen *Lift*.

### Lift of an Association Rule

Definiert durch $$lift(X\rightarrow Y) = \frac{X \cup Y}{support(X)*support(Y)}$$

Der Lift kann auch höher als 1 sein und ist statistisch unabhängig.

### Interpretation of Lift

Lift ist nach oben offen. Man sollte Verteilung ansehen und Daten nach Lift sortieren. Je nach
Streuung müssen Regelstärke angepasst interpretiert werden. Man kann auch Elbow-Method anwenden.

* Lift = 1, ist statistisch unabhängig
* Lift < 1, solltee nicht passieren, Regel andersrum wäre beesser
* Lift > 1, jee grösser umso beesser, je stärker deren Verbindung

Lift = Association Strength

### A Note on Business

wenn zwei Produkte oft miteinander gekauft werden, könnte folgende Überlegungen gemacht werden.

* X und Y näher oder weiter auseinander ins Regal legen
* X und Y zusammen verkaufen
* X und Y mit einem dritten schlechte laufenden Artikel verkaufen
* Rabatt nur auf einem der Produkte geben (nicht auf beide)
* Einen der Preise erhöhen (oder senken)
* einen der Artikel bewerben (nicht beide gleichzeitig)

### Limitations of Apriori and Alternatives

Apriori ist langsam für grosse Daten. FP-Growth sollte verwendet werden.

## API Check

Keine Implementation in scikit learn, dafür auf [mlextend](http://rasbt.github.io/mlxtend/)
