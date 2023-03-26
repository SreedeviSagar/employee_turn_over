import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {'support vector classifier' : SVC(kernel='rbf'),
                      'KNeighbors classifier' : KNeighborsClassifier(), 
                     #'MultinomialNB': MultinomialNB(), 
                      'Decision Tree': DecisionTreeClassifier(),
                      'Logistic regression': LogisticRegression(), 
                      'Random Forest': RandomForestClassifier(), 
                      'AdaBoost': AdaBoostClassifier(),
                      'Bagging classifier':BaggingClassifier(), 
                      'Extra trees classifier': ExtraTreesClassifier(),
                      'Gradient boosting':GradientBoostingClassifier(),
                      'xgb classifier':XGBClassifier()
                      }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info('Best found model on both training and testing dataset')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            acc_scr=accuracy_score(y_test,predicted)
            return acc_scr,best_model
        
        except Exception as e:
            raise CustomException(e,sys)
