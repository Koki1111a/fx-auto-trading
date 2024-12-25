import functions
import pandas as pd
from datetime import datetime, timezone

df_train = pd.read_pickle('./experiment/M10_trainData.pkl') # 学習用データ
df_test = pd.read_pickle('./experiment/M10_devData.pkl') # 評価用データ
T = functions.FXtrading(timeFrame='M10', forwardSteps=128, login=False, df_train=df_train, df_test=df_test)
T.trainModel(test=True, set_model=False, path='./experiment/M10_model.h5', riverse=False, custom_epochs=300)