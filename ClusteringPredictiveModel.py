from FrequencyEncoder import FrequencyEncoder
from LastStateEncoder import LastStateEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ClusteringPredictiveModel:
    
    def __init__(self,cls_features, case_id_col, event_col, label_col, timestamp_col, cat_cols, numeric_cols, n_clusters, n_estimators, random_state=22, fillna=True, pos_label="A_Pending"):
        
        # columns
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.pos_label = pos_label
        
        self.n_clusters = n_clusters
        
        self.freq_encoder = FrequencyEncoder(case_id_col, event_col)
        self.data_encoder = LastStateEncoder(case_id_col, timestamp_col, cat_cols, numeric_cols, fillna)
        self.clustering = KMeans(n_clusters, random_state=random_state)
        self.clss = [RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) for _ in range(n_clusters)]
        self.data_freqs=0
        self.cls_features=cls_features
    
    def fit(self, X, y=None):
        
        # encode events as frequencies
        data_freqs = self.freq_encoder.fit_transform(X)
        #print(len(data_freqs))
        # cluster traces according to event frequencies 
        cluster_assignments = self.clustering.fit_predict(data_freqs)
        #print(len(cluster_assignments))
        # train classifier for each cluster
        for cl in range(self.n_clusters):
            cases = data_freqs[cluster_assignments == cl].index
            tmp = X[X[self.case_id_col].isin(cases)]
            name='clusters/cluster_'+str(cl)+'.csv'
            tmp.to_csv(name, sep=';')
            tmp = self.data_encoder.transform(tmp)
            #missing columns:
            missing_columns=[col for col in self.cls_features if col not in list(tmp.columns)]
            mc=pd.DataFrame(columns=missing_columns)
            tmp=pd.concat([tmp,mc],ignore_index=True,sort=False).fillna(0)
            self.clss[cl].fit(tmp.drop([self.case_id_col, self.label_col], axis=1), tmp[self.label_col])
        #print(tmp.drop([self.case_id_col, self.label_col], axis=1))
        return self
    
    
    def predict_proba(self, X):
        
        # encode events as frequencies
        self.data_freqs = self.freq_encoder.transform(X)
        
        # calculate closest clusters for each trace 
        cluster_assignments = self.clustering.predict(self.data_freqs)
        
        # predict outcomes for each cluster
        cols = [self.case_id_col]+list(self.clss[0].classes_)
        preds = pd.DataFrame(columns=cols)
        self.actual = pd.DataFrame(columns=cols)
        for cl in range(self.n_clusters):
            
            # select cases belonging to given cluster
            cases = self.data_freqs[cluster_assignments == cl].index
            if len(cases)>0:
                tmp = X[X[self.case_id_col].isin(cases)]
                
                # encode data attributes
                tmp = self.data_encoder.transform(tmp)
                #missing columns:
                missing_columns=[col for col in self.cls_features if col not in list(tmp.columns)]
                mc=pd.DataFrame(columns=missing_columns)
                tmp=pd.concat([tmp,mc],ignore_index=True,sort=False).fillna(0)
                # make predictions
                new_preds = pd.DataFrame(self.clss[cl].predict_proba(tmp.drop([self.case_id_col, self.label_col], axis=1)))
                new_preds.columns = self.clss[cl].classes_
                new_preds[self.case_id_col] = cases
                #new_preds
                preds = pd.concat([preds, new_preds], axis=0, ignore_index=True,sort=False)
                #print('cluster ', cl)
                #print(new_preds.head())
                # extract actual label values
                actuals = pd.get_dummies(tmp[self.label_col])
                #print(actuals)
                actuals[self.case_id_col] = tmp[self.case_id_col]
                self.actual = pd.concat([self.actual, actuals], axis=0, ignore_index=True,sort=False)
                print(' ',len(new_preds),end='')
            else:
                print(' 0',end='')
        print('')
        preds.fillna(0, inplace=True)
        self.actual.fillna(0, inplace=True)
        #self.actual = self.actual[self.pos_label]
        
        #return preds[self.pos_label]
        return preds
        