# teleco_churn_ML
### Option 1  `telco_preprocessed_full.csv`

#Full control version.  

All preprocessing steps have been applied except for **scaling the numerical features**.
The dataset has not yet been split into training and testing sets, since scaling must be performed after the split to avoid data leakage. Therefore, the full dataset cannot be scaled before training.
Use this version if you want to apply your own scaler or custom split ratio.

**code can help you scale the training set after splitting**
````python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/path/telco_preprocessed_full.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale AFTER splitting **just the training set**
numeric = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
scaler = StandardScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric]  = scaler.transform(X_test[numeric])
````

### Option 2 — `X_train / X_test / y_train / y_test`

#Ready to model version.  
Scaling is already applied ,no preprocessing needed, go straight to modeling.  
Split is done with the standard ratio: **80% training / 20% testing**.

#### How to use Option 2:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

# start modeling
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
