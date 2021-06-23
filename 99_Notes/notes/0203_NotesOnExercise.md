## Data Preparation for Recommender Systems
Folgend Notizen zur Übung.
Um die Daten (strings) verarbeiten zu können müssen diese erst vorverarbeitet werden.

### Convert Strings to lower-case format
Text in lower-case Buchstaben umwandeln damit gleiche Worte immer gleich geschrieben werden.

```python
df['description'] = df['description'].str.lower()
```

### Tokenizing
Tokenizing ist ein Verfahren um einen Text zu bereinigen. Das Resultat der Tokenisierung ist eine Liste von Tokens, die als Liste im technischen Sinn, oder als Abfolge von durch Zeilenumbrüche getrennte Tokens. Deren eine Tokenklasse angehängt wird. Während diesem Vorgang werden einige Einzelaufgaben bewältigt. Die Tokens werden auch als *StopWords* bezeichnet.

* Abkürzungen erkennen und isolieren (es gibt auch gleiche Abkürzungen für unterschiedliche Worte)
* Interpunktionen und Sonderzeichen erkennen (Problem diverse Sonderzeichen wie /, @, #, $, usw. gehören oft einem Token an und dürfen nicht isoliert werden)
* kontrahierte Formen expandieren; l'auto $\longrightarrow$ la auto; gilt das nachher als Artikel oder Pronomen? Führt zu Ambiguität.
* komplexe Tokens erkennen und isolieren; allgemeine Zahlen wie 10 000, Telefonnummer, Datum und Zeit, URLs, Vor- und Nachnamen (was ist mit Titel?)
* ggf. Tokens normalisieren
    * Abkürzungen vereinheitlichen
    * Datums, Zeit- und Massangaben vereinheitlichen
    * Zahlen
* ggf. Tokens klassifizieren (d.h. Tokenklassen bilden); Klassen wie number, date, time, abbr, currency, temp, length usw. bilden
    * Diese Schuhe haben sFr. 147.- gekostet. $\longrightarrow$ [diese,schuhe,haben,currency(147,sfr,Rp),gekostet,.]
    * Wir treffen uns am 24. April um 15 Uhr. $\longrightarrow$ [wir,treffen,uns,am,date(24,4,Jr),um,time(15,Min,Sec),.]

### Lemmatization
Die Lemmatisierung ist das Rückführen von Worten in ihre Grundform. So wie sie im Wörterbuch stehen, diese werden als *Lemma* bezeichnet. Das ursprüngliche Wort ist die *Vollform*.

### Stemming
Als Stemming wird die Stammformreduktion oder Normalformenreduktion bezeichnet. Es ist ein Verfahren um verschiedene morphologische Varianten eines Wortes auf ihr gemeinsamen Wortstamm zurückzuführen. Zum Beispiel der **Deklination** von *Wortes* oder *Wörter* zu *Wort* und **Konjugation** von *geseheen* oder *sah* zu *sehen*. Bekanntes Framework ist das `Snowball`von Martin Porter.

### Data
Daraus ergeben sich normalisierte Daten die für das Training und den Test verwendet werden. Das Set ist immer in die zwei Gruppen Test und Training aufzuteilen.
