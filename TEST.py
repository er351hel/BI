#Grundsetup für Data Science in Python

import pandas as pd #dataframes und datenmanipulation
import numpy as np #numerical operations

import matplotlib.pyplot as plt #visualisierung
import seaborn as sns 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

##
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








### Aufgabe 2: EDA Visualisierungen

#EDA (Explorative Data Analysis)
#Übersicht der Zielvariable Survived um die Verteilung zu sehen
sns.countplot(x="Survived", data=df)
plt.title("Verteilung der Zielvariable Survived")
plt.show()

#Überblick der Geschlechterverteilung um zu sehen wie viele Männer und Frauen überlebt haben bzw. ob es Unterschiede gibt
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Überleben nach Geschlecht")
plt.show()

#Überblick der Survivalrate nach ALter
#df_plot = df[['a',  'Survived']].dropna()
#plt.figure(figsize=(8, 6))
#sns.scatterplot(data=df_plot, x='Age', y='Survived', palette='Set1', alpha=0.7)
#plt.title("Überleben nach Alter")
#plt.show()

#Passagierklasse und Überleben um zu sehen ob es Unterschiede gibt bei verschiedenen Klassen (wegen Priorität bei Rettungsbooten etc.)
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Überleben nach Passagierklasse")
plt.show()

# Vorheriger Plot unter Trennung der Geschlechter um Unterschiede weiter aufzuschlüsseln (z.B. zwei nebeneinanderliegende Plots)
g = sns.catplot(x="Pclass", hue="Survived", col="Sex", data=df, kind="count",
				height=4, aspect=0.9, palette='Set1')
g.fig.suptitle("Überleben nach Passagierklasse getrennt nach Geschlecht", y=1.02) 
g.set_axis_labels("Pclass", "Anzahl")
plt.show()

#Alterverteilung um zu sehen wie das Alter der Passagiere verteilt ist
# Mithilde des Histogramms war es einfacher festzulegen wie Missing values im Alter behandelt werden sollten
sns.histplot(df["Age"], kde=True)
plt.title("Verteilung des Alters")
plt.show()


### Neues Meeting 15.12.2025
### Aufgabe 2 fortgesetzt: EDA Visualisierungen

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








##Aufgabe 3a und 3b
#Daten in Trainings- und Testset aufteilen 80/20 gemäß Vorlesungscoding
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

### Neues Meeting 21.12.2025
### Aufgabe 4: Auswahl Klassifikationsalgorhithmus
### Auswahl basiert auf den Daten und Zielen der Analyse

#Herranziehen drei verschiedener Modelle: Logistische Regression, Decision Tree, Random Forest um die beste Performance 
# zu ermitteln anhand der Accuracy und Cross Validation
# Logistische Regression Modell erstellen und trainieren
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

#error rate
y_pred = log_reg.predict(X_test)        
print("Accuracy:", accuracy_score(y_test, y_pred))
### Decision Tree 
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=5,          # verhindert Overfitting
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

### Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # Anzahl der Bäume im Wald
    max_depth=5,         # verhindert Overfitting
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

### Vergleich der Modelle "Cross-Validation"
print("\nModellvergleich mittels Cross-Validation:")
models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt,
    "Random Forest": rf
}

for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10)  

#Zur Modellbewertung wurde eine k‑fold Cross‑Validation mit unterschiedlichen
#Werten für k (5, 10, 20, 100) durchgeführt.
#Dabei zeigte sich, dass sich der Mittelwert der Accuracy ab k = 10 stabilisiert
#und höhere Werte für k keinen signifikanten Erkenntnisgewinn mehr liefern,
#sondern lediglich die Rechenzeit erhöhen.
#Daher wurde für die weitere Analyse eine 10‑fold Cross‑Validation verwendet.

#| Accuracy                 | Durchschnittliche Accuracy                  |
#| ------------------------ | ------------------------------------------- |
#| Ein einzelner Wert       | Mittelwert aus mehreren Läufen              |
#| Abhängig von einem Split | Robuster & fairer                           |
#| Kann Zufall sein         | Bessere Schätzung der echten Modellleistung |
#| Gut für schnellen Check  | Standard für Modellvergleich                |

# Entscheidung für Random Forest basierend auf der höchsten Accuracy
# Performace ist höher und stabiler als bei der Logistischen Regression 
#Robuster gegenüber Overfitting als Decision Tree

    print(f"{model_name}: {scores.mean():.4f} ± {scores.std():.4f}")

#Aufgabe 5 Training von Random Forest auf dem gesamten Trainingsset
rf_final = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42,
    min_samples_split=5 
)
rf_final.fit(X, y)

y_final_pred = rf_final.predict(X_test)
print("\nFinal Random Forest Accuracy on Test Set:", accuracy_score(y_test, y_final_pred))  

# Aufgabe 6 Modellbewertung
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_final_pred))

#Interpretation der Confusion Matrix:
#Das Modell berechnet für jede Person eine Überlebenswahrscheinlichkeit.
#Nur wenn diese hoch genug ist (Standard-Schwelle ~0.5), sagt das Modell "überlebt".
#Bei Unsicherheit entscheidet es sich eher für "nicht überlebt". 0.5 und darunter ist "nicht überlebt", darüber "überlebt".
#Dadurch erkennt das Modell Todesfälle zuverlässiger als Überlebende
#(wenige False Positives, aber mehr False Negatives).

#Aufgabe 6b Cross-Validation mit dem finalen Modell
print("\nFinal Model Cross-Validation Accuracy:")
final_scores = cross_val_score(rf_final, X, y, cv=10)
print(f"Random Forest: {final_scores.mean():.44f} ± {final_scores.std():.4f}") 

#Aufgabe 6c Fehleranalyse

#Confusion Matrix visualisieren um die Fehler besser zu verstehen
cm = confusion_matrix(y_test, y_final_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nicht Überlebt', 'Überlebt'], yticklabels=['Nicht Überlebt', 'Überlebt'])
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.title('Confusion Matrix des Random Forest Modells')
plt.show()

#Feature Importance
#Visualisierung der wichtigsten Merkmale um zu verstehen, welche Variablen am meisten zur Vorhersage beitragen
#Variablen decken sich mit den Erkenntnissen aus der EDA
importances = rf_final.feature_importances_ #Funktion zur Berechnung der Feature Importance
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6)) 
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

#Feinabstimmung des Modells durch Hyperparameter-Optimierung
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]         
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBeste Hyperparameter:", best_params) # Beste Hyperparameter ausgeben
# Punkte des Finetunings wie Feature Engineering, Bereinigungen der Kategoriencrossvalidation mit versch. k und 
# modellvergleich wurden bereits in vorherigen Schritten gemacht
# Hyperparameter waren hiermit zuletzt ermittelt worden um die Modellperformance weiter zu verbessern 
# min_samples_split': 5 ist gleich akkurat wie 10 aber weniger komplex und schneller

#Aufgabe 9
#1) Vorhersagen + Wahrscheinlichkeiten
y_pred = rf_final.predict(X_test)
y_proba = rf_final.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeit für Survived=1

# 2) PowerBI-Export-DF bauen
powerbi_df = X_test.copy()
powerbi_df["Survived_True"] = y_test.values
powerbi_df["Survived_Pred"] = y_pred
powerbi_df["Survival_Probability"] = y_proba

#3)  Fehlertyp als Text
powerbi_df["Result"] = "TN"
powerbi_df.loc[(powerbi_df["Survived_True"]==0) & (powerbi_df["Survived_Pred"]==1), "Result"] = "FP"
powerbi_df.loc[(powerbi_df["Survived_True"]==1) & (powerbi_df["Survived_Pred"]==0), "Result"] = "FN"
powerbi_df.loc[(powerbi_df["Survived_True"]==1) & (powerbi_df["Survived_Pred"]==1), "Result"] = "TP"

#4) Export als CSV
powerbi_df.to_csv("titanic_powerbi_output.csv", index=False)