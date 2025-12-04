In diesem Schein wird es darum gehen, den sogenannten "Titanic" Datensatz zu untersuchen, Daten zu explorieren und in Folge einen Data Mining Algorithmus darauf zu trainieren.

Bearbeitung in 2er bis 3er Gruppen.

**Für die Abgabe wird erwartet:**

1. Vorverarbeitung der Daten \& Data Mining gerechte Anwendung eines Klassifikationsalgorithmus mit dem Ziel mindestens eine der vorgegebenen Zielvariablen (Pclass, Survived, oder Sex) anhand der Anderen vorhandenen Parameter abzuleiten:

	-> Beachten Sie bitte die Sektionen weiter unten.

* Visuelle Explorative Datenanalyse (Visuell = Plots z.B. Seaborn, Interpretation der Plots als Kommentar/Markdown Block); auch "EDA" genannt
* Sinnvolle Aufteilung der Daten in "train" und "validation" Datensätze
* Anpassen eines Klassifikationsmodells auf den "train" Datensatz
* Visuelle und schriftliche Auswertung der Train/Test Performance mit einer sogenannten "Cross validation" - bitte erklären Sie wie die Ergebnisse der Cross Validation zu deuten sind.
* Abgabe als .ipynb Datei (Python Notebook) - Dokumentation von Denkprozessen in Markdown


Dashboard gerechte Aufbereitung der Daten zur interaktiven Visualisierung in Power BI (mehr zu Power BI in der VL am 15.05.2025):

* Darstellung von Datenverteilungen und Insights innerhalb eines BI Dashboards mit Filtermöglichkeiten - Sie wollen die Daten dem "Management" eines Unternehmens möglichst simpel zugänglich machen
* Mind. 3 versch. interaktive Visualisierungen (sprich, wenn ein Filter gelegt wird bzw. mit einem Diagramm interagiert wird, sollten sich die anderen Diagramme mit anpassen)
* Abgabe als .pbix Datei (Power BI Datei) // Andere Dashboard Lösungen sind auch zugelassen (Tableau, Qlik, Looker Studio, ...)
* LIVE Präsentation (15-20 min) + Rückfragen, Feedback \& Co.:
* Präsentation LIVE - Ergebnisse Data Mining \& Dashboard
* Für die LIVE Präsentationen nutzen Sie bitte Ihren eigenen Testdatensatz test.csv (bitte vorab splitten und abspeichern)



**Bewertungskriterien zum Bestehen:**

Erfüllung der Bullets aus 1. \& 2. als Gruppe

Diskussion / Beantwortung von Fragen der Zuhörer \& Dozent bei LIVE Präsentation ; ggf. 1 on 1 Interview bei unsicherer Präsentation zur finalen Beurteilung BE/NB falls große Unterschiede in der Performance zwischen Gruppenmitgliedern erkennbar sind


Erklärung der Daten:

Variable		Definition					         Key

**survival**	**Survival**					    0 = No, 1 = Yes

**pclass**		**Ticket** **class**				1 = 1st, 2 = 2nd, 3 = 3rd

**sex**			**Sex**	 

Age			    Age in years	 

sibsp			# of siblings / spouses aboard the Titanic	 

parch			# of parents / children aboard the Titanic	 

ticket			Ticket number	 

fare			Passenger fare	 

cabin			Cabin number	 

embarked		Port of Embarkation				C = Cherbourg, Q = Queenstown, S = Southampton





