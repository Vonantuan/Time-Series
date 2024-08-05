from utils._1_Imports.tsfI import *
from utils._2_Cleaning.tsfC import *
from utils._3_ModelTraining.tsfT import *
from utils._4_ModelEvaluation.tsfE import *


data_path = 'Dataset/AAPL.csv'
dataset = load_data(data_path)
df = pre_prep_data(dataset)
check_data(df)
decompisition_analysis(df)
ar_model = train_arimamodel(df)
ypred,conf_int = forecast_using_Arima(ar_model)
dataframe_from_prediction_values(dataset,ypred,conf_int)
dfx = Bivariate_using_ExogenousVariable(dataset)
arimax = train_arimamodel_bivariate(dfx)
ex,ypred,conf_int = evaluate_arima_bivariate(arimax,dataset)
dataframe_from_predvalues_ex(dataset,ypred,conf_int)
dataYF = yfinance_dataset_API()
train,test,model1,model1_preds,features = train_XGBOOST(dataYF)
evaluation_XGBOOST(test,model1_preds)