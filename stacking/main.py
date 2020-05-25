# -*- coding: utf-8 -*-

# ----- for creating dataset -----
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ----- general import -----
import pandas as pd
import numpy as np

# ----- stacking library -----
from stacking.base import FOLDER_NAME, PATH, INPUT_PATH, \
        FEATURE_PATH, OUTPUT_PATH, SUBMIT_FORMAT
# ----- utils -----
from stacking.base import load_data, create_cv_id, \
        eval_pred
# ----- classifiers -----
from stacking.base import BaseModel, XGBClassifier, KerasClassifier

# ----- tensorflow.keras -----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1, l2

# ----- scikit-learn -----
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ----- Set problem type -----
problem_type = 'classification'
classification_type = 'multi-class'
eval_type = 'auc'

BaseModel.set_prob_type(problem_type, classification_type, eval_type)




# ----- First stage stacking model-----

# FILES LISTS in Stage 1.
FEATURE_LIST_stage1 = {
                'train':(
                         INPUT_PATH + 'train.csv',
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(
                         INPUT_PATH + 'test.csv',
                        ),
                }

# need to get input shape for NN 获取神经网络的input_shape
X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=False)
assert((False in X.columns == test.columns) == False)
nn_input_dim_NN = X.shape[1:]
output_dim = len(set(y))
del X, y, test


# Models in Stage 1
PARAMS_V1 = {
        'colsample_bytree':0.80,
        'learning_rate':0.1,
        "eval_metric":"mlogloss",
        'max_depth':5, 
        'min_child_weight':1,
        'nthread':4,
        'seed':407,
        'silent':1, 
        'subsample':0.60,
        'objective':'multi:softprob',
        'num_class':output_dim,
        }
# xgboost model in stage1
class ModelV1(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=300)


PARAMS_V2 = {
            'batch_size':32,
            'nb_epoch':100,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            #'show_accuracy':True,
            'class_weight':None,
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }
# NN model in stage1
class ModelV2(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dense(64, input_shape=nn_input_dim_NN, kernel_initializer='he_normal')) # hidden layer1
            model.add(LeakyReLU(alpha=.00001))
            model.add(Dropout(0.5))
                        
            model.add(Dense(output_dim, kernel_initializer='he_normal')) # hidden layer2
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)

            model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

            return KerasClassifier(nn=model,**self.params)

PARAMS_V3 = {
             'n_estimators':500, 'criterion':'gini', 'n_jobs':8, 'verbose':0,
             'random_state':407, 'oob_score':True,
             }
# RandomForest model in stage1
class ModelV3(BaseModel):
        def build_model(self):
            return RandomForestClassifier(**self.params)


PARAMS_V4 = {
             'n_estimators':300, 'learning_rate':0.05,'subsample':0.8,
             'max_depth':5, 'verbose':1, 'max_features':0.9,
             'random_state':407,
             }
# GBDT model in stage1
class ModelV4(BaseModel):
        def build_model(self):
            return GradientBoostingClassifier(**self.params)

PARAMS_V5 = {
             'n_estimators':650, 'learning_rate':0.01,'subsample':0.8,
             'max_depth':5, 'verbose':1, 'max_features':0.82,
             'random_state':407,
             }
# SVM model in stage1
class ModelV5(BaseModel):
        def build_model(self):
            return GradientBoostingClassifier(**self.params)

# ----- END first stage stacking model -----

# ----- Second stage stacking model -----

PARAMS_V1_stage2 = {
        'colsample_bytree':0.8,
        'learning_rate':0.1,
        "eval_metric":"mlogloss",
        'max_depth':4, 
        'seed':1234,
        'nthread':8,
        'reg_lambda':0.01,
        'reg_alpha':0.01,
        'silent':1, 
        'subsample':0.80,
        'objective':'multi:softprob',
        'num_class':output_dim,
        }

class ModelV1_stage2(BaseModel):
        def build_model(self):
            return XGBClassifier(params=self.params, num_round=330)



# ----- END first stage stacking model -----

if __name__ == "__main__":
    
    # Create cv-fold index 创建交叉验证的索引，以npy保存
    target = pd.read_csv(INPUT_PATH + 'target.csv')
    create_cv_id(target, n_folds_ = 5, cv_id_name='cv_id', seed=407)

    ######## stage1 Models #########
    print('-' * 20)
    print('Start stage 1 training')
    print('-' * 20)
    print('training xgboost model in stage1')

    m = ModelV1(name="xgb_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1,
        
                )
    m.run()

    print('-' * 20)
    print('training NN model in stage1')

    m = ModelV2(name="nn_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2,
                
                )
    m.run()

    print('-' * 20)
    print('training RandomForest model in stage1')

    m = ModelV3(name="rf_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3,
                
                )
    m.run()

    print('-' * 20)
    print('training GBDT model in stage1')

    m = ModelV4(name="gbdt_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4,
                
                )
    m.run()

    print('-' * 20)
    print('training SVM model in stage1')

    m = ModelV5(name="svm_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5,
                
                )
    m.run()

    print('-' * 20)

    print('Done stage 1')
    print() 
    ######## stage2 Models #########

    print('-' * 20)
    print('Start stage 2 training')

    # FILES LISTS in Stage 2.
    
    FEATURE_LIST_stage2 = {
                'train':(INPUT_PATH + 'train.csv',
                         
                         FEATURE_PATH + 'xgb_stage1_all_fold.csv',
                         FEATURE_PATH + 'nn_stage1_all_fold.csv',
                         FEATURE_PATH + 'rf_stage1_all_fold.csv',
                         FEATURE_PATH + 'gbdt_stage1_all_fold.csv',
                         FEATURE_PATH + 'svm_stage1_all_fold.csv',
                
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(INPUT_PATH + 'test.csv',
                         
                         FEATURE_PATH + 'xgb_stage1_test.csv',
                         FEATURE_PATH + 'nn_stage1_test.csv',
                         FEATURE_PATH + 'rf_stage1_test.csv',
                         FEATURE_PATH + 'gbdt_stage1_test.csv',
                         FEATURE_PATH + 'svm_stage1_test.csv',
                                                
                        ),
                }


    print('training xgboost model in stage1')
    m = ModelV1_stage2(name="xgb_stage2",
                    flist=FEATURE_LIST_stage2,
                    params = PARAMS_V1_stage2,
                
                    )
    m.run()

    print('Done stage 2')
    print() 
    
    
    print('Saving final result')

    pred = pd.read_csv(OUTPUT_PATH + 'stacking_features/xgb_stage2_test.csv')
    ori = pd.read_csv(INPUT_PATH + 'test.csv')
    print('saving final results')
    pred.columns = ['other_proba', 'granites_proba', 'basalts_proba']
    pred = pd.concat([ori, pred], axis=1)
    pred.to_csv(OUTPUT_PATH + 'final_results.csv', index=False)

    

