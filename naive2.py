import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None

filename = 'Heart_Disease.csv'
df = pd.read_csv(filename)

#RENAMING COLUMNS
df.rename(columns = {'Cleveland':'Hospital','63':'age','1':'sex','1.1':'cp', '145':'trestbps','233':'chol','1.2':'fbs','2':'restecg','150':'thalach','0':'exang','2.3':'oldpeak','3':'slope',
                     '0.1':'ca','6':'thal', '0.2':'num'}, inplace = True)

#DEFINING DISCRETE AND CONTINUOUS COLUMNS
discr = ['sex','cp','fbs','restecg','exang']
cont = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

#DROPPING COLUMNS AND ROWS WITH MISSING VALUES
df.drop(['slope','ca','thal','Hospital'], axis=1,inplace=True)

#SHAPE OF DATAFRAME
m,n = df.shape

copy=df.to_numpy()
for i in range(m):
  flag=0
  for j in range(n):
    if(copy[i][j]=='?'):
      flag+=1
  if(flag!=0):
    df.drop(i, axis=0, inplace=True)
    i-=1

#REMOVING OUTLIERS
temp_df = df
for col in temp_df.columns:
        #print('Working on column: {}'.format(col))
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        mean = temp_df[col].mean()
        sd = temp_df[col].std()
        temp_df = temp_df[(temp_df[col] <= mean+(3*sd))]
df = temp_df

#SHUFFLING VALUES
df = df.sample(frac = 1)#shuffling

#SPLITTING INTO TRAINING AND TEST SET
m=578
df_train = df.iloc[:m, :]
for col in cont:
    df_train[col],bins = pd.cut(df_train[col], bins = 4, labels = [1,2,3,4], retbins = True)
    if col == 'trestbps':
        print(bins)
X_train = df_train.iloc[:,0:n-1]
Y_train = df_train.iloc[:, n-1:]
print(df_train)

m=145
df_test = df.iloc[:m, :]
for col in cont:
    df_test[col],bins = pd.cut(df_test[col], bins = 4, labels = [1,2,3,4], retbins = True)
X_test = df_test.iloc[:,0:n-1]
Y_test = df_test.iloc[:, n-1:]


###########################################################################################################################
### NAIVE BAYES CLASSIFIER

#CALCULATING CLASS PROBABILITIES USING TRAINING SET Y
N_total = len(Y_train)
N_negative = 0
N_positive = 0
for i in range(0, N_total):
    if Y_train.iloc[i]['num'] == 0:
        N_negative+=1
    else:
        N_positive+=1

#CLASS PROBABILITIES
Prob_negative = N_negative/N_total
Prob_positive = N_positive/N_total

#CALCULATING FEATURE PROBABILITIES

mask_negative = df_train['num'] == 0
mask_positive = df_train['num'] > 0

df_negative = df_train[mask_negative]
print(df_negative['trestbps'])
df_positive = df_train[mask_positive]

negative = {}
positive = {}
for col in df_negative.columns:
    if col in discr:
        temp_dict = {} #Initialzing a dict like cp : {1: ,2: ,3: ...}
        unique_vals = df_negative[col].unique() #finding unique vals in cp
        col_length = len(df_negative[col]) #length of values in column
        for i in unique_vals: #calculating prob for each unique value
            unique_count = 0
            for elem in df_negative[col]:
                if elem == i:
                    unique_count +=1
            temp_dict[int(i)] = unique_count/col_length
        negative[col] = temp_dict
    elif col in cont:
        temp_dict = {}
        unique_vals = [1,2,3,4]
        col_length = len(df_negative[col])
        for i in unique_vals:
            unique_count = 0
            for elem in df_negative[col]:
                if elem == i:
                    unique_count +=1
            if unique_count == 0:
                unique_count = 1
            temp_dict[int(i)] = unique_count/col_length
        negative[col] = temp_dict

for col in df_positive.columns:
    if col in discr:
        temp_dict = {} #Initialzing a dict like cp : {1: ,2: ,3: ...}
        unique_vals = df_positive[col].unique() #finding unique vals in cp
        col_length = len(df_positive[col]) #length of values in column
        for i in unique_vals: #calculating prob for each unique value
            unique_count = 0
            for elem in df_positive[col]:
                if elem == i:
                    unique_count +=1
            temp_dict[int(i)] = unique_count/col_length
        positive[col] = temp_dict
    elif col in cont:
        temp_dict = {}
        unique_vals = [1,2,3,4]
        col_length = len(df_positive[col])
        for i in unique_vals:
            unique_count = 0
            for elem in df_positive[col]:
                if elem == i:
                    unique_count +=1
            if unique_count == 0:
                unique_count = 1
            temp_dict[int(i)] = unique_count/col_length
        positive[col] = temp_dict

print(negative)
print(positive)

#CALCULATING ACCURACY USING TEST SET
outcome = np.zeros((len(Y_test),1))

for i in range(0, len(outcome)):
    test = X_test.iloc[i, :] #test example
    neg_prob = Prob_negative
    pos_prob = Prob_positive
    for feature,value in test.items():
            try:
                neg_prob = neg_prob * (negative[feature][value])
                pos_prob = pos_prob * (positive[feature][value])
            except:
                print(feature, value) 
    if neg_prob > pos_prob:
        outcome[i] = 0
    elif pos_prob > neg_prob:
        outcome[i] = 1

print(outcome)
Y_test.loc[Y_test["num"] > 0 , "num"] = 1
Y_test = Y_test.to_numpy()

sum = 0
for i in range(0, len(Y_test)):
    if Y_test[i] == outcome[i]:
        sum += 1

#print(outcome)
print("Accuracy: " , (sum/len(Y_test)))