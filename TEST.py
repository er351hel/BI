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

#Vorbereitung der Daten für das Modell
#Unnötige Spalten entfernen
df_model = df.copy()
df_model = df_model.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

#Missing Values behandeln
#Alter: Median verwenden
df["Age"].fillna(df["Age"].median(), inplace=True)
#Embarked: Modus verwenden
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
#fare: Median verwenden
df["Fare"].fillna(df["Fare"].median(), inplace=True)

#Kategorische Variablen in numerische umwandeln
df_model = pd.get_dummies(df_model, columns=["Sex", "Embarked"], drop_first=True)

#Feature und Zielvariable definieren
y = df_model["Survived"]
X = df_model.drop(columns=["Survived"]) #Alle Variablen außer Survived

#Daten in Trainings- und Testset aufteilen 80/20
X_train, X_test, y_train, y_test = train_test_split(    #train und validation sets
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # sorgt dafür, dass die Klassenverteilung in beiden Splits ähnlich ist
)

#drop nans if any remain
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

#Logistische Regression Modell erstellen und trainieren
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

#error rate
y_pred = log_reg.predict(X_test)        
print("Accuracy:", accuracy_score(y_test, y_pred))

###