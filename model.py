# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor

from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import pickle



#--------------- load and traitement


df=pd.read_csv('data.csv')

types = pd.get_dummies(df['type'], prefix='type', drop_first=True)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df['nameOrig'] = label.fit_transform(df['nameOrig'])
df['nameDest'] = label.fit_transform(df['nameDest'])

df[['nameOrig', 'nameDest']]

df = pd.concat([df, types], axis=1)
df = df.drop('type', axis=1)

X = df.drop('isFraud', axis=1)
y = df['isFraud']






#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = XGBClassifier()


#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
'''model = pickle.load(open('model.pkl','rb'))'''
print('done')