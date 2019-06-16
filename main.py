#Importing Basic Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset

dataset = pd.read_csv('Match_Data.csv')
print(dataset)

dataset = dataset.dropna() #Droping all NULL values

# Droping the columns which are not needed in our dataset. 
dataset = dataset.drop(['Match_SK', 'match_id', 'match_date', 'Season_Year', 'City_Name', 'Country_Name', 'Win_Margin', 'Outcome_Type', 'ManOfMach', 'Win_Margin', 'Country_id','Win_Type'], axis = 1)
print(dataset)
# Replacing names of the team which are old to new
dataset.replace(to_replace = 'Deccan Chargers', value = 'Sunrisers Hyderabad', inplace = True)
print(dataset)
# Encoding the names of the team names.

dataset = dataset.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=False)

encode = {'Team1': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DD':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'Team2': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DD':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'Toss_Winner': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DD':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'match_winner': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DD':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12,'Draw':13}}

dataset = dataset.replace(encode, inplace=False)

# Renaming the same names written in different types to same
dataset = dataset.replace(to_replace = 'Field', value = 'field', inplace = False)
dataset = dataset.replace(to_replace = 'Bat', value = 'bat', inplace = False)

#Importing LabelBinarizer from sklearn library
from sklearn.preprocessing import LabelBinarizer

# Giving name lb to object of LabelBinarizer
lb = LabelBinarizer()

print(dataset.Toss_Name)
#Encoding the column we need to encode
dataset.Toss_Name = lb.fit_transform(dataset.Toss_Name)

#Importing LabelEncoder from sklearn library
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encoding column we need to encode
dataset.Venue_Name = labelencoder.fit_transform(dataset.Venue_Name)

# Initializing all the values to variable a except match_winner column
X=dataset.drop('match_winner',axis=1)

#Initializing match winner to variable y
y=dataset['match_winner']

#Importing train_test_split from sklearn.model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
lg_pred=lg.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score ',accuracy_score(y_test,lg_pred))

# Applying Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score ',accuracy_score(y_test,nb_pred))

# Applying Support Vector Classifier
from sklearn.svm import SVC
svc=SVC(C=0.8)
svc.fit(x_train,y_train)
svc_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score ',accuracy_score(y_test,svc_pred))

# Applying Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score ',accuracy_score(y_test,rfc_pred))