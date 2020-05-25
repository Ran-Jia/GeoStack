#!/usr/bin/env python
# -*- coding: utf-8 -*-

######### General #########
import numpy as np
import pandas as pd
import os, sys, re
import tables
import itertools
import logging


######### Problem Type #########
eval_type_list = ('logloss', 'auc', 'rmse') 

problem_type_list = ('classification','regression')

classification_type_list = ('binary', 'multi-class')



######### PATH #########

FOLDER_NAME = ''
PATH = ''
DATA_PATH = 'data/'
INPUT_PATH = 'data/input/' 
OUTPUT_PATH = 'data/output/'
FEATURE_PATH = 'data/output/stacking_features/' 


SUBMIT_FORMAT = 'final_result.csv'
 




######### BaseEstimator ##########
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin

######### Keras #########
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback

######### XGBoost #########
import xgboost as xgb

import os
from time import asctime, time
import subprocess
import csv
import pandas as pd

######### Evaluation ##########
from sklearn.metrics import log_loss as ll
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error




######### create cv index #########
cv_id_name='cv_id' 
n_folds = 5

def create_cv_id(target, n_folds_ = 5, cv_id_name=cv_id_name, seed=407):
    a = StratifiedKFold(n_splits=n_folds_, shuffle=True, random_state=seed)
    cv_index = np.empty(len(target))
    for idx, i in enumerate(a.split(target,target)):
        cv_index[i[1]] = idx
    cv_index = cv_index.astype(int)
    print('Done StratifiedKFold')

    np.save(INPUT_PATH + cv_id_name, cv_index)
    return


######### Utils #########

def load_data(flist, drop_duplicates=False):

    if (len(flist['train'])==0) or (len(flist['target'])==0) or (len(flist['test'])==0):
        raise Exception('train, target, and test must be set at \
                                    least one file, respectively.')

    X_train = pd.DataFrame()
    test = pd.DataFrame()

    print('Reading train dataset')
    for i in flist['train']:
        X_train = pd.concat([X_train, pd.read_csv(PATH+i)],axis=1)

    print('train dataset is created')


    print('Reading target data')
    y_train = pd.read_csv(PATH+flist['target'][0])['target']

    print('Reading train dataset')
    for i in flist['test']:
        test = pd.concat([test, pd.read_csv(PATH+i)],axis=1)

    assert( (False in X_train.columns == test.columns) == False)
    print('train shape :{}'.format(X_train.shape))
    if drop_duplicates == True:
        #delete identical columns
        unique_col = X_train.T.drop_duplicates().T.columns
        X_train = X_train[unique_col]
        test = test[unique_col]
        assert( all(X_train.columns == test.columns))
        print('train shape after concat and drop_duplicates :{}'.format(X_train.shape))
    
    return X_train, y_train, test 


#evalation function
def eval_pred( y_true, y_pred, eval_type):
    if eval_type == 'logloss':
        loss = ll( y_true, y_pred )
        print("logloss: ", loss)
        return loss            
    
    elif eval_type == 'auc':
        loss = AUC( y_true, y_pred, multi_class='ovo')
        print("AUC: ", loss)
        return loss             
    
    elif eval_type == 'rmse':
        loss = np.sqrt(mean_squared_error(y_true, y_pred))
        print("rmse: ", loss)
        return loss




######### BaseModel Class #########

class BaseModel(BaseEstimator):

    problem_type = ''
    classification_type = ''
    eval_type = ''


    def __init__(self, name="", flist={}, params={}, fold_name=cv_id_name):

        if BaseModel.problem_type == 'classification':
            if not ((BaseModel.classification_type in classification_type_list)
                     and (BaseModel.eval_type in eval_type_list)):
                raise ValueError('Problem Type, Classification Type, and Evaluation Type\
                        should be set before model defined')

        elif BaseModel.problem_type == 'regression':
            if not BaseModel.eval_type in eval_type_list:
                raise ValueError('Problem Type, and Evaluation Type\
                        should be set before model defined')

        else:
            raise ValueError('Problem Type, Classification Type, and Evaluation Type\
                        should be set before model defined')

        self.name = name
        self.flist = flist
        self.params = params
        self.fold_name = fold_name
        
        
    @classmethod
    def set_prob_type(cls, problem_type, classification_type, eval_type):
        """ Set problem type """
        assert problem_type in problem_type_list, 'Need to set Problem Type'
        if problem_type == 'classification':
            assert classification_type in classification_type_list,\
                                            'Need to set Classification Type'
        assert eval_type in eval_type_list, 'Need to set Evaluation Type'
        
        cls.problem_type = problem_type
        cls.classification_type = classification_type
        cls.eval_type = eval_type
        
        if cls.problem_type == 'classification':
            print('Setting Problem:{}, Type:{}, Eval:{}'.format(cls.problem_type,
                                                                cls.classification_type,
                                                                cls.eval_type))

        elif cls.problem_type == 'regression':
            print('Setting Problem:{}, Eval:{}'.format(cls.problem_type,
                                                        cls.eval_type))

        return



    def build_model(self):
        return None

    def make_multi_cols(self, num_class, name):
        '''make cols for multi-class predictions'''
        cols = ['c' + str(i) + '_' for i in range(num_class)]
        cols = [x + name for x in cols]
        return cols


    def run(self):
        print('running model: {}'.format(self.name))
        X, y, test = self.load_data()
        num_class = len(set(y))

        print('loading cv_fold file')
        a = np.load(INPUT_PATH + self.fold_name + '.npy')

        clf = self.build_model()
        print("Creating train and test sets for stacking.")
        

        ############# for binary #############
        if BaseModel.problem_type == 'regression' or BaseModel.classification_type == 'binary':
            dataset_blend_train = np.zeros(X.shape[0]) 
            dataset_blend_test = np.zeros(test.shape[0]) 
    
            dataset_blend_test_j = np.zeros((test.shape[0], n_folds))
        
        ############# for multi-class #############
        elif BaseModel.classification_type == 'multi-class':
            #TODO
            #train结果保存
            dataset_blend_train = np.zeros(X.shape[0]*num_class).reshape((X.shape[0],num_class))
            #test预测结果保存
            dataset_blend_test = np.zeros(test.shape[0]*num_class).reshape((test.shape[0],num_class))


        ############## Start stacking ################
        evals = []
        for i in range(n_folds):# of n_folds
            train_fold = (a!=i)
            test_fold = (a==i)
            print("Fold", i)

            X_train = X[train_fold].dropna(how='all')
            y_train = y[train_fold].dropna(how='all')
            X_test = X[test_fold].dropna(how='all')
            y_test = y[test_fold].dropna(how='all')
            
        
            if 'sklearn' in str(type(clf)):
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, X_test, y_test)


            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':            

                if 'sklearn' in str(type(clf)):
                    y_submission = clf.predict_proba(X_test)[:,1]
                else:
                    y_submission = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                if 'sklearn' in str(type(clf)):
                    y_submission = clf.predict_proba(X_test) #Check!!
                else:
                    y_submission = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'regression':      
                y_submission = clf.predict(X_test)


            try:
                dataset_blend_train[test_fold] = y_submission
            except:
                dataset_blend_train[test_fold.values] = y_submission
            
            
            evals.append(eval_pred(y_test, y_submission, BaseModel.eval_type))

            ############ binary classification ############
            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':            
                
                if 'sklearn' in str(type(clf)):
                    dataset_blend_test += clf.predict_proba(test)[:,1]
                else:
                    dataset_blend_test += clf.predict_proba(test)

            ############ multi-class classification ############
            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':            
                
                dataset_blend_test += clf.predict_proba(test)
                pass

            ############ regression ############
            elif BaseModel.problem_type == 'regression':      
                
                dataset_blend_test += clf.predict(test)


        dataset_blend_test /= n_folds
        
        for i in range(n_folds):
            print('Fold{}: {}'.format(i+1, evals[i]))
        print('{} CV Mean: '.format(BaseModel.eval_type), np.mean(evals), ' Std: ', np.std(evals))

        
        print('Saving results')

        if (BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary') or (BaseModel.problem_type == 'regression'):
            dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=['{}_stack'.format(self.name)])
            dataset_blend_train.to_csv(FEATURE_PATH+'{}_all_fold.csv'.format(self.name),index=False)
            dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=['{}_stack'.format(self.name)])
            dataset_blend_test.to_csv(FEATURE_PATH+'{}_test.csv'.format(self.name),index=False)
            

        elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
            saving_cols = self.make_multi_cols(num_class, '{}_stack'.format(self.name))
            dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=saving_cols)
            dataset_blend_train.to_csv(FEATURE_PATH+'{}_all_fold.csv'.format(self.name),index=False)

            dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=saving_cols)
            dataset_blend_test.to_csv(FEATURE_PATH+'{}_test.csv'.format(self.name),index=False)
                

        return



    def load_data(self):
        '''
        flistにシリアライゼーションを渡すことでより効率的に
        data構造をここで考慮
        '''
        return load_data(self.flist, drop_duplicates=False )
        




class KerasClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, normalize=True, categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, X_test=None, y_test=None):
        X = X.values#Need for Keras
        y = y.values#Need for Keras

        if self.normalize:
            self.mean = np.mean(X,axis=0)
            self.std = np.std(X,axis=0) + 1 #CAUSION!!!
            X = (X - self.mean)/self.std
        if self.categorize_y:
            y = to_categorical(y)


        if X_test is not None:
            X_test = X_test.values#Need for Keras
            y_test = y_test.values#Need for Keras

            if self.normalize:
                X_test = (X_test - self.mean)/self.std
            if self.categorize_y:
                y_test = to_categorical(y_test)

            self.validation_data = (X_test, y_test)

        else:
            self.validation_data = []


        
        #set initial weights
        self.nn.set_weights(self.init_weight)


        return self.nn.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=self.validation_data, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=self.shuffle, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict_proba(self, X, batch_size=128, verbose=1):
        X = X.values#Need for Keras
        if self.normalize:
            X = (X - self.mean)/self.std
        
        if BaseModel.classification_type == 'binary':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)[:,1]#multi-class 
        elif BaseModel.classification_type == 'multi-class':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)


class XGBClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, params={}, num_round=50 ):
        self.params = params
        self.num_round = num_round

        self.clf = xgb
        
    def fit(self, X, y=[], X_test=None, y_test=None, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        
        dtrain = xgb.DMatrix(X, label=y, missing=-999)

        if X_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
            watchlist  = [(dtrain, 'train'), (dtest, 'validation')]
        
        else:
            watchlist  = [(dtrain, 'train')]
        
        self.clf = xgb.train(self.params, dtrain, self.num_round, watchlist)
        return self.clf

    def predict_proba(self, X, output_margin=False, ntree_limit=0):
        dtest = xgb.DMatrix(X,missing=-999)
        
        return self.clf.predict(dtest)




###########################################
######### Regressor Wrapper Class #########
###########################################

class KerasRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, 
            normalize=True, categorize_y=False, random_sampling=None):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        self.random_sampling = random_sampling
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, X_test=None, y_test=None):
        
        if self.random_sampling != None:
            self.sampling_col = np.random.choice(list(range(X.shape[1])),self.random_sampling,replace=False)
            X = X.iloc[:,self.sampling_col].values#Need for Keras
        else:
            X = X.values#Need for Keras

        y = y.values#Need for Keras

        if self.normalize:
            self.mean = np.mean(X,axis=0)
            self.std = np.std(X,axis=0) + 1 #CAUSION!!!
            X = (X - self.mean)/self.std

  
        if X_test is not None:
            X_test = X_test.values#Need for Keras
            y_test = y_test.values#Need for Keras
            if self.normalize:
                X_test = (X_test - self.mean)/self.std

            self.validation_data = (X_test, y_test)

        else:
            self.validation_data = []


        #set initial weights
        self.nn.set_weights(self.init_weight)
        print(X.shape)
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, validation_data=self.validation_data, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=self.shuffle, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict(self, X, batch_size=128, verbose=1):
        if self.random_sampling != None:
            X = X.iloc[:,self.sampling_col].values
        else:
            X = X.values#Need for Keras
        if self.normalize:
            X = (X - self.mean)/self.std
        
        return [ pred_[0] for pred_ in self.nn.predict(X, batch_size=batch_size, verbose=verbose)]
    

class XGBRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, params={}, num_round=50 ):
        self.params = params
        self.num_round = num_round

        self.clf = xgb
        
    def fit(self, X, y=[], X_test=None, y_test=None, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        
        dtrain = xgb.DMatrix(X, label=y,missing=-999)

        if X_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
            watchlist  = [(dtrain, 'train'), (dtest, 'validation')]
        
        else:
            watchlist  = [(dtrain, 'train')]
 

        self.clf = xgb.train(self.params, dtrain, self.num_round, watchlist)
        return self.clf

    def predict(self, X, output_margin=False, ntree_limit=0):
        dtest = xgb.DMatrix(X,missing=-999)
        
        return self.clf.predict(dtest)


