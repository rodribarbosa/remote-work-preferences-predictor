import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor



class Model:
    
    def __init__(self, categorical_features, continuous_features, label):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.features = categorical_features+continuous_features
        self.label = label
        #self.reg = RandomForestRegressor(max_depth=11, n_estimators=700, max_leaf_nodes=800)
        self.reg = HistGradientBoostingRegressor(max_depth=6, max_leaf_nodes=4)
        
    def averager(self, df):
        columns = self.features + [self.label]
        return df[columns].groupby(self.features).agg(['mean'])
        
    def fit(self, df):

        X = df.dropna(subset=self.label).drop(self.label, axis=1)[self.features]
        y = df[self.label].dropna()

        enc = TargetEncoder()
        ct = ColumnTransformer([("target_enc", enc, self.categorical_features)
                                , ("pass", 'passthrough', self.continuous_features)]
                               , sparse_threshold=0)
        X_new = ct.fit_transform(X, y)

        #comment the next four lines to use HistGradientBoostRegressor
        #imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant')
        #X_new = imp_mean.fit_transform(X_new)
        #lab = LabelEncoder()
        #y = lab.fit_transform(y)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X_new, y, random_state=42)

        self.reg = self.reg.fit(X_train,y_train)

    
    def test_score(self):
        return round(self.reg.score(self.X_test,self.y_test),2)
    
    def predict(self, X_input):
        return self.reg.predict(X_input)
    
    #def top_features(self):
    #    return self.reg.feature_importances_
    
    
    
class Averager:
    '''
    The Averager class computes averages paycut percentage over a subset of features.
    '''
    
    def __init__(self, feature):
        self.avg_paycut = defaultdict(float)
        self.feature = feature
    
    def average(self, X, y):
            
        # Store the average paycut percentage per region (state) in self.avg_paycut
        paycut_sum = defaultdict(float)
        count = defaultdict(float)
        for row, paycut in zip(X.to_dict('records'),y):
            paycut_sum[row[self.feature]] += paycut
            count[row[self.feature]] += 1
        for feature in paycut_sum:
            self.avg_paycut[feature] = paycut_sum[feature]/count[feature]
        
        l = []
        for row in X.to_dict('records'):
            if row[self.feature] not in self.avg_paycut.keys():
                l.append(('',0))
            else:
                l.append((row[self.feature],self.avg_paycut[row[self.feature]]))
        return sorted([*set(l)], key = lambda x: x[1])