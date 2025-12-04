
# This line imports the pandas library, which is essential for data manipulation.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_df = pd.read_csv('c:/HTWG/WIN7/BI/BI/train.csv')

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Pclass
sns.countplot(x='Pclass', hue='Survived', data=train_df, ax=axes[0])
axes[0].set_title('Survival Count by Pclass')
axes[0].set_xlabel('Passenger Class')
axes[0].set_ylabel('Number of Passengers')
axes[0].legend(title='Survived', labels=['No', 'Yes'])

# Plot for Sex
sns.countplot(x='Sex', hue='Survived', data=train_df, ax=axes[1])
axes[1].set_title('Survival Count by Sex')
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Number of Passengers')
axes[1].legend(title='Survived', labels=['No', 'Yes'])

plt.tight_layout()
plt.show()
