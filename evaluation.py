import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass

class MSE(Evaluation):   
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)->float : 
        try:
            logging.info("calculating mse")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating mse: {}".format(e))
            raise e

class R2(Evaluation):
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray)->float:
        try:
            logging.info("calculating R2")
            r2=r2_score(y_true,y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2: {}".format(e))
            raise e

class RMSE(Evaluation):
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray)->float:
        try:
            logging.info("calculating rmse")
            rmse=mean_squared_error(y_true,y_pred,squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating rmse: {}".format(e))
            raise e


