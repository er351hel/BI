#Grundsetup für Data Science in Python

import pandas as pd #dataframes und datenmanipulation
import numpy as np #numerical operations

import matplotlib.pyplot as plt #visualisierung
import seaborn as sns 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Titanic Datensatz einlesen
df = pd.read_csv("train.csv")
df.head()

#Festgelegt auf Survival und Sex als Zielvariable
#Zielvariable Survived (0 = gestorben, 1 = überlebt)

#Datenüberblick und Vorbereitende Analyse
df.info()
df.describe(include="all")
df.isna().sum()

#es gibt 12 spalten und 891 einträge
#Age Cabin und Embarked weisen fehlende Werte auf
#Alter hat 177 fehlende Werte, Cabin 687 und Embarked 2
#Datentypen sind int64, float64 und object


#Vorbereitung der Daten für das Modell
#Unnötige Spalten entfernen
df_model = df.copy() #Kopie des originalen Dataframes erstellen
df_model = df_model.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

#Missing Values behandeln
#Alter: Median verwenden macht kein Sinn bei ALter, sont Unverhältnismäßig hoher Anteil der 33 Jährigen sprich 177 Einträge haben das selbe Alter
# Alle NAs löschen macht aber auch keinen Sinn da sonst 177 Einträge fehlen, und somit auch wichtige Informationen verloren gehen über andere Variablen
# Daher Gruppenmedian verwenden basierend auf Pclass und Sex
df_model["Age"] = df_model.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))

#Embarked: Modus verwenden
df_model["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Agegroup erstellen und aufteilen nach Alter
df_model["Agegroup"] = pd.cut(df_model["Age"], bins=[0, 3, 12, 20, 40, 60, 80], labels=["Infant", "Child", "Teen", "Adult", "Middle-aged", "Senior"])

#Kategorische Variablen in numerische umwandeln
df_model = pd.get_dummies(df_model, columns=["Sex", "Embarked", "Agegroup"], drop_first=True)  #Was macht.get_dummies genau?  -> Wandelt kategorische Variablen in numerische Dummy-Variablen um
df_model.info()  #Überprüfen ob alles geklappt hat
#Feature und Zielvariable definieren
y = df_model["Survived"]
X = df_model.drop(columns=["Survived"]) #Alle Variablen außer Survived



#Daten in Trainings- und Testset aufteilen 80/20
X_train, X_test, y_train, y_test = train_test_split(    #train und validation sets  # 80% training 20% test 
    X, y,
    test_size=0.2, #20% der Daten für das Testset
    random_state=42, 
    stratify=y   # sorgt dafür, dass die Klassenverteilung in beiden Splits ähnlich ist
)

#drop nans if any remain
#X_train = X_train.dropna()
#X_test = X_test.dropna()
#y_train = y_train[X_train.index]
#y_test = y_test[X_test.index]



#Logistische Regression Modell erstellen und trainieren
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

#error rate
y_pred = log_reg.predict(X_test)        
print("Accuracy:", accuracy_score(y_test, y_pred))

###

#EDA (Explorative Data Analysis)
#Übersicht der Zielvariable Survived
sns.countplot(x="Survived", data=df)
plt.title("Verteilung der Zielvariable Survived")
plt.show()

#Überblick der Geschlechterverteilung
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Überleben nach Geschlecht")
plt.show()

#Überblick der Survivalrate nach ALter
#df_plot = df[['a',  'Survived']].dropna()
#plt.figure(figsize=(8, 6))
#sns.scatterplot(data=df_plot, x='Age', y='Survived', palette='Set1', alpha=0.7)
#plt.title("Überleben nach Alter")
#plt.show()

#Passagierklasse und Überleben
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Überleben nach Passagierklasse")
plt.show()

# Zusätzlich: Pclass getrennt nach Sex (z.B. zwei nebeneinanderliegende Plots)
g = sns.catplot(x="Pclass", hue="Survived", col="Sex", data=df, kind="count",
				height=4, aspect=0.9, palette='Set1')
g.fig.suptitle("Überleben nach Passagierklasse getrennt nach Sex", y=1.02)
g.set_axis_labels("Pclass", "Anzahl")
plt.show()

#Alterverteilung
sns.histplot(df["Age"], kde=True)
plt.title("Verteilung des Alters")
plt.show()


### Neues Metting 15.12.2025

#Berechne Korrelationen zwischen numerischen Variablen
correlation_matrix = df_model.corr()
plt.figure(figsize=(10, 8))  # Größe des Plots anpassen
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm") # Heatmap der Korrelationen
plt.title("Korrelationsmatrix der numerischen Variablen") # Titel des Plots
plt.show()

#Interpretation der Heatmap
# Männer haben eine negative Korrelation mit dem Überleben (-0.54), was darauf hindeutet, dass Frauen eher überleben.
# Pclass hat ebenfalls eine negative Korrelation mit dem Überleben (-0.34), was bedeutet, dass Passagiere in höheren Klassen eher überleben.
# Fare hat eine positive Korrelation mit dem Überleben (0.26), was darauf hindeutet, dass Passagiere, die mehr bezahlt haben, eher überleben.

#Scatterplot für wichtige Korrelationen
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_model, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.7)
plt.title("Zusammenhang zwischen Alter, Fahrpreis und Überleben")
plt.xlabel("Alter")
plt.ylabel("Ticketpreis")
plt.legend(title="Überlebt (0=Nein, 1=Ja)")
plt.show()

#*# Zusammenfassung Scatterplott 1: Bei sehr niedrigen Ticketpreisen scheint die Überlebensrate 
# niedriger zu sein, insbesondere bei älteren Passagieren. Kinder und Babys haben eine sehr hohe 
# Überlebensrate, unabhängig vom Ticketpreis. 

#Scatterplot 2
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Fare",
    y="Age",
    hue="Sex",                        # Farbe = Geschlecht
    style="Survived",                 # Markerform = Überleben
    palette={"female": "pink", "male": "blue"},
    markers={0: "X", 1: "o"},         # X = gestorben, o = überlebt
    s=120,                            # Punktgröße
    alpha=0.65                        # leichte Transparenz
)

plt.xscale("log")                     # Fare hat extreme Ausreißer → log scale = viel schöner
plt.title("Fare vs. Age – nach Sex (Farbe) & Survived (Markerform)")
plt.xlabel("Fare (log scale)")
plt.ylabel("Age")
plt.grid(alpha=0.2)
plt.show()

#*# Zusammenfassung Scatterplott 2: Zur besseren Visualisierung wurde die Fare-Achse logarithmisch skaliert, 
# um die Verteilung der Ticketpreise besser darzustellen. Weibliche Passagiere (rosa Punkte) haben eine höhere 
# Überlebensrate (Kreis-Markierungen) im Vergleich zu männlichen Passagieren (blaue Punkte). 
# Besonders auffällig ist, dass viele weibliche Passagiere unabhängig vom Alter überlebt haben, 
# während männliche Passagiere, insbesondere ältere, häufiger gestorben sind.


#Pairplot zur Visualisierung von Beziehungen
cols_to_plot = ['Age', 'Fare', 'Pclass', 'Survived']
sns.pairplot(df[cols_to_plot].dropna(), hue='Survived', palette='husl', diag_kind='kde')
plt.show()

#*# Zusammenfassung Pairplot: Der Pairplot zeigt die Verteilungen und Beziehungen zwischen den Variablen
# Alter, Fahrpreis, Passagierklasse und Überleben. Es ist erkennbar, dass jüngere Passagiere und 
# solche mit höheren Fahrpreisen tendenziell eine höhere Überlebensrate aufweisen.

