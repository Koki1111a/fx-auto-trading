import functions
import pandas as pd
from keras.layers import BatchNormalization
from tensorflow.python.keras.models import load_model

df_test = pd.read_pickle('./experiment/M10_testData.pkl') # 評価用データ
#model_path = './experiment/M30_model.h5' #読み込むモデル
model_path = './tmp/tmp_model.h5'
T = functions.FXtrading(login=False, df_test=df_test, model_path=model_path)
T.testModel(cmx=True, prob=0.999)
