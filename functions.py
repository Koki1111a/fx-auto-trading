import math
import os
import time
import tqdm
import pickle
import talib as ta
import numpy as np
import seaborn as sn
import MetaTrader5 as mt5
import pandas as pd
import tensorflow as tf
from datetime import datetime, timezone
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.layers import Dense,  Activation, Dropout, Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import Sequential
from keras.utils import np_utils

# カレントディレクトリの設定
os.chdir(os.path.dirname(__file__))


class FXtrading:

    def __init__(
                 self, 
                 login = True, # ログインするか
                 symbol = 'USDJPY', # 取引対象
                 timeFrame = 'M10', # 時間軸の種類
                 forwardSteps = 64, # 参照する未来の時間軸の本数
                 backwardSteps = 127, # 参照する過去の時間軸の本数
                 lot_ratio = 0.000001, # 1度に発注するロット[lot]/全財産[円] 
                 magic_number = 10002, # マジックナンバー 
                 slippage = 10, # スリッページ
                 profitLine = None, # 利確ライン[pips]
                 lossLine = None, # 損切ライン[pips] 
                 date_from = datetime(2014, 5, 1, tzinfo=timezone.utc), #どの時期からのデータを取得したいか
                 date_to = None, #どの時期までのデータ取得したいか

                 df_train = None, # 学習用データ
                 df_test = None, # 評価用データ

                 model_path = None
                 ):
        
        self.login = login

        if login:
            # MT5と接続
            if not mt5.initialize():
                print(f'initialize() failed, error code = {mt5.last_error()}')

            # MT5でブローカーにログイン
            authorized = mt5.login(
                75104589,
                password='*VYLG(4O55FM7h2',
                server='XMTrading-MT5 3',
            )

            if not authorized:
                print('login failed.')
        else:
            authorized = False

        self.symbol = symbol
        self.timeFrame = timeFrame
        self.tf = self.getTF() # 時間軸の単位[分]
        self.tradeFrame = self.tf * 6 # 注文の時間の間隔
        self.forwardSteps = forwardSteps
        self.backwardSteps = backwardSteps
        self.lot_ratio = lot_ratio
        self.magic_number = magic_number
        self.slippage = slippage

        if authorized:
            self.frame = self.getFrame() # 時間軸
            self.balance = mt5.account_info().balance # 総資産
            self.price = mt5.symbol_info_tick(self.symbol).ask # 注文価格
            self.margin_level = mt5.account_info().margin_level # 証拠金維持率
            self.commission = 10 * mt5.symbol_info_tick('USDJPY').ask # 手数料[円/1lotの取引]
            self.oneLotSize = mt5.symbol_info(self.symbol).trade_contract_size # 1lotの通貨量
            self.orderLot = int(100 * self.balance * self.lot_ratio) * 0.01 # 注文ロット数
            if self.orderLot < 0.01: self.orderLot = 0.01
            self.point = mt5.symbol_info(self.symbol).point # 取引通貨の最小単位
            self.pip = self.point * 10.0
            self.positions = mt5.positions_get(group='*'+self.symbol+'*') # ポジション情報
            self.spread = mt5.symbol_info_tick(self.symbol).ask - mt5.symbol_info_tick(self.symbol).bid # スプレッド
        else:
            self.balance = None
            self.price = None
            self.margin_level = None
            self.commission = None
            self.orderLot = None
            self.point = None
            self.pip = None
            self.positions = None
            self.spread = None

        if profitLine is None:
            self.profitLine = 50.0
            self.adjustProfitLine()
        else:
            self.profitLine = profitLine
        
        if lossLine  is None:
            self.lossLine = self.profitLine * 0.8
        else:
            self.lossLine = lossLine

        self.date_from = date_from
        self.df_train = df_train
        self.df_test = df_test

        if date_to is None:
            now = datetime.now()
            self.date_to = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            if not os.path.exists(f"./data/model_{self.date_to.strftime('%Y%m%d')}.h5"):
                if now.month == 1:
                    self.date_to = datetime(now.year-1, 12, 1, tzinfo=timezone.utc)
                else:
                    self.date_to = datetime(now.year, now.month-1, 1, tzinfo=timezone.utc)
        else:
            self.date_to = date_to

        if model_path is None:
            if os.path.exists('./data/model_current.h5'):
                self.model = load_model('./data/model_current.h5', custom_objects={'BatchNormalization': BatchNormalization})
                self.model_path = './data/model_current.h5'
            else:
                self.model = None
                self.model_path = None
        else:
            self.model = load_model(model_path, custom_objects={'BatchNormalization': BatchNormalization})
            self.model_path = model_path





    def update(self): # パラメータの更新

        if self.login == False:
            print('update() : online only.')
            return

        self.balance = mt5.account_info().balance # 総資産
        self.price = mt5.symbol_info_tick(self.symbol).ask # 注文価格
        self.margin_level = mt5.account_info().margin_level # 証拠金維持率
        self.commission = 10 * mt5.symbol_info_tick('USDJPY').ask # 手数料[円/1lotの取引]
        self.oneLotSize = mt5.symbol_info(self.symbol).trade_contract_size # 1lotの通貨量
        self.orderLot = int(100 * self.balance * self.lot_ratio) * 0.01 # 注文ロット数
        if self.orderLot < 0.01: self.orderLot = 0.01
        self.point = mt5.symbol_info(self.symbol).point # 取引通貨の最小単位
        self.pip = self.point * 10.0
        self.positions = mt5.positions_get(group='*'+self.symbol+'*') # ポジション情報
        self.spread = mt5.symbol_info_tick(self.symbol).ask - mt5.symbol_info_tick(self.symbol).bid # スプレッド




    def getTF(self): # 時間軸の単位[分]の取得

        if self.timeFrame == 'M1': return 1*60
        if self.timeFrame == 'M2': return 2*60
        if self.timeFrame == 'M3': return 3*60
        if self.timeFrame == 'M4': return 4*60
        if self.timeFrame == 'M5': return 5*60
        if self.timeFrame == 'M6': return 6*60
        if self.timeFrame == 'M10': return 10*60
        if self.timeFrame == 'M12': return 12*60
        if self.timeFrame == 'M15': return 15*60
        if self.timeFrame == 'M20': return 20*60
        if self.timeFrame == 'M30': return 30*60
        if self.timeFrame == 'H1': return 60*60
        if self.timeFrame == 'H2': return 2*60*60
        if self.timeFrame == 'H3': return 3*60*60
        if self.timeFrame == 'H4': return 4*60*60
        if self.timeFrame == 'H6': return 6*60*60
        if self.timeFrame == 'H8': return 8*60*60
        if self.timeFrame == 'H12': return 12*60*60
        if self.timeFrame == 'D1': return 24*60*60
        if self.timeFrame == 'W1': return 7*24*60*60
        #if self.timeFrame == 'MN1': return False





    def getFrame(self): # 時間軸の取得

        if self.login == False:
            print('getFrame() : online only.')
            return

        if self.timeFrame == 'M1': return mt5.TIMEFRAME_M1
        if self.timeFrame == 'M2': return mt5.TIMEFRAME_M2
        if self.timeFrame == 'M3': return mt5.TIMEFRAME_M3
        if self.timeFrame == 'M4': return mt5.TIMEFRAME_M4
        if self.timeFrame == 'M5': return mt5.TIMEFRAME_M5
        if self.timeFrame == 'M6': return mt5.TIMEFRAME_M6
        if self.timeFrame == 'M10': return mt5.TIMEFRAME_M10
        if self.timeFrame == 'M12': return mt5.TIMEFRAME_M12
        if self.timeFrame == 'M15': return mt5.TIMEFRAME_M15
        if self.timeFrame == 'M20': return mt5.TIMEFRAME_M20
        if self.timeFrame == 'M30': return mt5.TIMEFRAME_M30
        if self.timeFrame == 'H1': return mt5.TIMEFRAME_H1
        if self.timeFrame == 'H2': return mt5.TIMEFRAME_H2
        if self.timeFrame == 'H3': return mt5.TIMEFRAME_H3
        if self.timeFrame == 'H4': return mt5.TIMEFRAME_H4
        if self.timeFrame == 'H6': return mt5.TIMEFRAME_H6
        if self.timeFrame == 'H8': return mt5.TIMEFRAME_H8
        if self.timeFrame == 'H12': return mt5.TIMEFRAME_H12
        if self.timeFrame == 'D1': return mt5.TIMEFRAME_D1
        if self.timeFrame == 'W1': return mt5.TIMEFRAME_W1
        if self.timeFrame == 'MN1': return mt5.TIMEFRAME_MN1


    def CanUpdateModel(self, now=datetime.now()):

        date_to = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        if os.path.exists(f"./data/model_{date_to.strftime('%Y%m%d')}.h5"): # 今月のモデルの更新をしていたら更新なし
            return False
        
        if(now.day <= 7 and now.weekday() == 6): # 第一日曜日にモデルを更新
            return True

        if now.month == 1:
            date_to = datetime(now.year-1, 12, 1, tzinfo=timezone.utc)
        else:
            date_to = datetime(now.year, now.month-1, 1, tzinfo=timezone.utc)

        if not os.path.exists(f"./data/model_{date_to.strftime('%Y%m%d')}.h5"): # 先月にモデルの更新をしていなかったら更新
            return True
        
        
        return False # 条件を満たさなければ更新なし

    
    def adjustProfitLine(self, now=datetime.now(), modelCheck=False): # 利確ラインを調節する

        if self.login == False:
            print('adjustProfitLine() : online only.')
            return
        
        # MT5からのデータ取得
        date_from = datetime(now.year-1, now.month, 1, tzinfo=timezone.utc)
        date_to = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

        if modelCheck and not os.path.exists(f"./data/model_{date_to.strftime('%Y%m%d')}.h5"): # 今月のモデルの更新をしていなかったら先月の利確ラインを求める
            if now.month == 1:
                date_from = datetime(now.year-2, 12, 1, tzinfo=timezone.utc)
                date_to = datetime(now.year-1, 12, 1, tzinfo=timezone.utc)
            else:
                date_from = datetime(now.year-1, now.month-1, 1, tzinfo=timezone.utc)
                date_to = datetime(now.year, now.month-1, 1, tzinfo=timezone.utc)
            
        rates = mt5.copy_rates_range(self.symbol, self.frame, date_from, date_to)
        df = pd.DataFrame(rates)

        # データの作成
        open = df['open'].values
        high = df['high'].values
        low = df['low'].values
        

        while True:
            group = np.full(len(df)-self.forwardSteps, 1)

            for i in range(len(group)):

                for j in range(self.forwardSteps):
                    if high[i+j+1] > open[i] + self.profitLine * self.pip:
                        group[i] = 2
                        break
                    elif low[i+j+1] < open[i] - self.profitLine * self.pip:
                        group[i] = 0
                        break

            y = np_utils.to_categorical(group)
            one_hot_sum = np.sum(y, axis=0)
            one_hot_ratio = one_hot_sum / len(y)

            if one_hot_ratio[1] < 0.33:
                self.profitLine = self.profitLine + 1.0
            elif one_hot_ratio[1] > 0.4 and self.profitLine > 20.0:
                self.profitLine = self.profitLine - 1.0
            else:
                break



    def downloadData(self, result=True, path=None, noiseCut=True, checkContinuity=True): # データをダウンロードする

        if self.login == False:
            print('downloadData() : online only.')
            return
        
        # 保存先のパス指定
        if path is None:
            self.makeDir('./data')
            path = f"./data/data_{self.date_to.strftime('%Y%m%d')}.pkl"

        def f_1(x):
            return (1 - math.exp(-x*x)) / (1 - math.exp(-1))
        def f_2(x_1, x_2):
            return (f_1(x_1) / (f_1(x_1) + f_1(x_2)) - 0.5) * f_1(x_1) * f_1(x_1) + 0.5

        # MT5からのデータ取得
        rates = mt5.copy_rates_range(self.symbol, self.frame, self.date_from, self.date_to)
        df = pd.DataFrame(rates)

        # パラメータ設定
        count = 0

        # データのフレーム作り
        colms = ['group', 'data', 'drop', 'rise', 'prob']
        df_output = pd.DataFrame(columns=colms)
        self.makeDir('./tmp')
        df_output.to_pickle('./tmp/tmp_data.pkl')

        # データの作成
        time = df['time'].values
        open = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        for ind in tqdm.tqdm(range(self.backwardSteps, len(df)-self.forwardSteps)):

            # 時間が連続でない場合はスキップ
            if checkContinuity:
                period_past = int(time[ind] - time[ind-self.backwardSteps])
                if(period_past != int(self.tf*self.backwardSteps)):
                    continue

            # 値動きの幅を記録
            drop = close[ind] - low[ind-self.backwardSteps:ind].min()
            rise = high[ind-self.backwardSteps:ind].max() - close[ind]
      
            group = 1
            for i in range(self.forwardSteps):
                if high[ind+i+1] > open[ind] + self.profitLine * self.pip: # 買うべき
                    group = 2
                    x_1 = 1.0
                    x_2 = (open[ind] - low[ind:ind+i+2].min()) / (self.profitLine * self.pip)
                    p_2 = f_2(x_1, x_2)
                    break
                elif low[ind+i+1] < open[ind] - self.profitLine * self.pip: # 売るべき
                    group = 0
                    x_1 = 1.0
                    x_2 = (high[ind:ind+i+2].max() - open[ind]) / (self.profitLine * self.pip)
                    p_2 = 1.0 - f_2(x_1, x_2)
                    break

            if i == self.forwardSteps - 1:
                x_1 = (high[ind:ind+i+2].max() - open[ind]) / (self.profitLine * self.pip)
                x_2 = (open[ind] - low[ind:ind+i+2].min()) / (self.profitLine * self.pip)
                if x_1 > x_2:
                    p_2 = f_2(x_1, x_2)
                else:
                    p_2 = 1.0 - f_2(x_2, x_1)
            
            p_2 = np.round(p_2, 3) # 上がる確率
            p_1 = np.round(1.0-p_2, 3) # 下がる確率
            p = p_1, p_2

            data_open = open[ind-self.backwardSteps:ind+1]
            data_high = high[ind-self.backwardSteps:ind+1]
            data_low = low[ind-self.backwardSteps:ind+1]
            data_close = close[ind-self.backwardSteps:ind+1]
            data = [data_open, data_high, data_low, data_close]
            frame = [group, data, drop, rise, p]
            data_output = pd.DataFrame([frame], columns=colms, index=[str(count)])
            df_output = pd.concat([df_output, data_output], axis=0)
            count += 1
            if count % 30000 == 0: # 読み込むデータ数を大きくなりすぎないように一度保存
                df_output = pd.concat([pd.read_pickle('./tmp/tmp_data.pkl'), df_output], axis=0)
                df_output.to_pickle('./tmp/tmp_data.pkl') # 保存
                df_output = pd.DataFrame(columns=colms) # 読み込み中のデータを空っぽにする

        df_output = pd.concat([pd.read_pickle('./tmp/tmp_data.pkl'), df_output], axis=0)
        os.remove('./tmp/tmp_data.pkl')

        # 外れ値の除去
        if noiseCut:
            thr_drop = df_output['drop'].quantile(0.99)
            thr_rise = df_output['rise'].quantile(0.99)
            df_output = df_output.query('drop <= @thr_drop')
            df_output = df_output.query('rise <= @thr_rise')

        if result:
            df_output.to_pickle(path)
            y = df_output['group']
            y = np_utils.to_categorical(y)
            one_hot_sum = np.sum(y, axis=0)
            one_hot_ratio = one_hot_sum / len(y)
            print()
            print(f'group0：{one_hot_ratio[0]}')
            print(f'group1：{one_hot_ratio[1]}')
            print(f'group2：{one_hot_ratio[2]}')

        # ファイル書き込み
        df_output.to_pickle(path)

    




    def setPeriod(self, date_to=None, date_from=None): # データセットの期間の設定

        if date_to is not None:
            self.date_to = date_to
        if date_from is not None:
            self.date_from = date_from
    




    def setDf_train(self, date_to=None, date_from=None): # 学習データの設定

        self.setPeriod(date_to, date_from)
        path = f"./data/data_{self.date_to.strftime('%Y%m%d')}.pkl"
        
        if not os.path.isfile(path):
            self.downloadData(result=False)
        
        self.df_train = pd.read_pickle(path) # 学習用データ

        



    def getX_current(self): #現在時刻を基準に入力データの作成

        if self.login == False:
            print('get_current() : online only.')
            return

        # MT5からデータを取得
        rates = mt5.copy_rates_from_pos(
                        self.symbol, # 銘柄 
                        self.frame, # 時間軸
                        0, # 開始バーの位置。0は現在を表す
                        self.backwardSteps+1+2, # 取得するバーの数
                    )
        df = pd.DataFrame(rates)

        '''
        # データの時系列を確認
        period_past = int(df.at[self.backwardSteps+2, 'time'] - df.at[0, 'time'])
        if period_past != int(self.tf*(self.backwardSteps+2)):
            return None
        '''
        
        sma3 = ta.SMA(df['open'].values, timeperiod=3)
        x_current = sma3[2:]        
        
        with open('./data/x_range.pkl', 'rb') as p:
            d = pickle.load(p)
        x_range = d[self.model_path]

        # 値動きが基準以上に大きい場合はスキップ
        if(x_current[-1] - np.min(x_current) > x_range[0] or np.max(x_current) - x_current[-1] > x_range[1]):
            return None
        
        x_current = (x_current - x_current[-1]) / (x_range[0] + x_range[1]) + 0.5
        x_current = x_current.astype(np.float16)
        x_current = np.reshape(x_current, (1, 1, self.backwardSteps+1, 1))

        return x_current




    def makeDecide(self, prob=0.99): # アクションを決定する

        input = self.getX_current() # 入力データを取得する

        if input is None: # 入力データが得られない場合はアクションなし
            return 1
        
        output = np.array(self.model(input))
        output[output >= prob] = 1
        output[output < prob] = 0

        if output[0,0] == output[0,-1] == 0: # 予測確率が低い場合はアクションなし
            return 1
        
        return np.argmax(output) * 2
    




    def fictitiousTrade(self, mode='chechPriceAndTime'): # 架空の取引を行う

        # 架空ポジションの情報読み取り
        if os.path.exists('./data/fictitiousPosition.pkl'):
            with open('./data/fictitiousPosition.pkl', 'rb') as p:
                position = pickle.load(p)
        else:
            position = None

        if mode == 'buy': # 架空の買い注文
            sl = self.price - self.lossLine * self.pip + self.commission / self.oneLotSize # 逆指値
            tp = self.price + self.profitLine * self.pip + self.commission / self.oneLotSize #指値
            orderTime = time.time() # 注文した時刻

            position = [mode, sl, tp, orderTime]

        elif mode == 'sell': # 架空の売り注文
            sl = self.price + self.lossLine * self.pip - self.commission / self.oneLotSize # 逆指値
            tp = self.price - self.profitLine * self.pip - self.commission / self.oneLotSize # 指値
            orderTime = time.time() # 注文した時刻

            position = [mode, sl, tp, orderTime]

        elif mode == 'chechPriceAndTime' and position is not None: # 一定時間たっている，または指値になっているかを確認

            if time.time() - position[3] > self.tradeFrame: # 架空注文してからの経過時間
                position = None

            elif position[0] == 'buy' and (self.price < position[1] or self.price > position[2]): # 買いポジションで指値になっているか
                position = None

            elif position[0] == 'sell' and (self.price > position[1] or self.price < position[2]): # 売りポジションで指値になっているか
                position = None

        
        # 保存
        self.makeDir('./data')
        with open('./data/fictitiousPosition.pkl', 'wb') as p:
            pickle.dump(position, p)

        return position


            

    

    


    def buy(self, i=None): # 買いの注文

        if self.login == False:
            print('buy() : online only.')
            return

        if i is None:

            request = {
                        'symbol': self.symbol, # 通貨ペア（取引対象）
                        'action': mt5.TRADE_ACTION_DEAL, # 成行注文
                        'type': mt5.ORDER_TYPE_BUY, # 成行買い注文
                        'volume': self.orderLot, # ロット数
                        'price': self.price, # 注文価格
                        'sl': self.price - self.lossLine * self.pip + self.commission / self.oneLotSize, #逆指値
                        'tp': self.price + self.profitLine * self.pip + self.commission / self.oneLotSize, #指値
                        'deviation': self.slippage, # スリッページ
                        'type_time': mt5.ORDER_TIME_GTC, # 注文有効期限
                        'type_filling': mt5.ORDER_FILLING_IOC, # 注文タイプ
                        }

        elif len(self.positions) > i and self.positions[i][5] == 1:

            request = {
                        'symbol': self.symbol, # 通貨ペア（取引対象）
                        'action': mt5.TRADE_ACTION_DEAL, # 成行注文
                        'type': mt5.ORDER_TYPE_BUY, # 成行買い注文
                        'volume': self.positions[i][9], # ロット数
                        'price': self.price, # 注文価格
                        'deviation': self.slippage, # スリッページ
                        'type_time': mt5.ORDER_TIME_GTC, # 注文有効期限
                        'type_filling': mt5.ORDER_FILLING_IOC, # 注文タイプ
                        'position':self.positions[i][0] # チケットナンバー
                        }
        
        else:
            return # スキップ

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print('Order failed with error code:', result.retcode)
            print(result.comment)
        else:
            print('Order placed with ticket:', result.order)

        return result



    def sell(self, i=None): # 売りの注文

        if self.login == False:
            print('sell() : online only.')
            return
        
        if i is None:
            request = {
                        'symbol': self.symbol, # 通貨ペア（取引対象）
                        'action': mt5.TRADE_ACTION_DEAL, # 成行注文
                        'type': mt5.ORDER_TYPE_SELL, # 成行買い注文
                        'volume': self.orderLot, # ロット数
                        'price': self.price, # 注文価格
                        'sl': self.price + self.lossLine * self.pip - self.commission / self.oneLotSize, #逆指値
                        'tp': self.price - self.profitLine * self.pip - self.commission / self.oneLotSize, #指値
                        'deviation': self.slippage, # スリッページ
                        'type_time': mt5.ORDER_TIME_GTC, # 注文有効期限
                        'type_filling': mt5.ORDER_FILLING_IOC, # 注文タイプ
                        }
            
        elif len(self.positions) > i and self.positions[i][5] == 0:

            request = {
                        'symbol': self.symbol, # 通貨ペア（取引対象）
                        'action': mt5.TRADE_ACTION_DEAL, # 成行注文
                        'type': mt5.ORDER_TYPE_SELL, # 成行買い注文
                        'volume': self.positions[i][9], # ロット数
                        'price': self.price, # 注文価格
                        'deviation': self.slippage, # スリッページ
                        'type_time': mt5.ORDER_TIME_GTC, # 注文有効期限
                        'type_filling': mt5.ORDER_FILLING_IOC, # 注文タイプ
                        'position':self.positions[i][0] # チケットナンバー
                        }
            
        else:
            return # スキップ

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print('Order failed with error code:', result.retcode)
            print(result.comment)
        else:
            print('Order placed with ticket:', result.order)

        return result 





    def get_positionTime(self, i): #インデックスが i のポジションの経過時間を返す (iが大きいほど新しい)

        if self.login == False:
            print('get_positionTime() : online only.')
            return

        start_time = datetime.fromtimestamp(self.positions[i].time)
        last_bar = mt5.copy_rates_from_pos(self.symbol, self.frame, 0, 1)[0]
        now = datetime.fromtimestamp(last_bar[0])

        return (now - start_time).total_seconds()





    def cal_weight(class_id_list): # 学習データの重みを調節する

        amounts_of_class_array = np.zeros(len(class_id_list[0]))

        for class_id in class_id_list:
            amounts_of_class_array = amounts_of_class_array + class_id
        mx = np.max(amounts_of_class_array)

        class_weights = {}
        for q in range(0,len(amounts_of_class_array)):
            class_weights[q] = round(float(tf.math.pow(amounts_of_class_array[q]/mx, -1)),2)

        return class_weights
    
    
    def addDictionary(self, key, data, path):

        # 保存データの更新
        if os.path.exists(path):
            with open(path, 'rb') as p:
                d = pickle.load(p)
            d[key] = data
        else:
            d = {key:data}
        
        # 保存
        with open(path, 'wb') as p:
            pickle.dump(d, p)






    def trainModel(self, test=True, set_model=False, path=None, pretrained=None, riverse=False, custom_epochs=None): #VGG11でモデル作成
        
        # 保存先のパス指定
        if path is None:
            self.makeDir('./data')
            path = f"./data/model_{self.date_to.strftime('%Y%m%d')}.h5"

        # 学習データの読み込み
        df_train = self.df_train
        x_tmp = df_train['data'].values
        width = len(x_tmp[0][0])
        data_num = len(x_tmp)

        # 正規化用の数値を計算，保存
        x_range = [np.max(df_train['drop'].values), np.max(df_train['rise'].values)]
        self.addDictionary(key=path, data=x_range, path='./data/x_range.pkl')
        self.addDictionary(key='./tmp/tmp_model.h5', data=x_range, path='./data/x_range.pkl')

        # 正規化，リサイズ
        x_train = np.empty((data_num, 4, width))
        ran = max(x_range)
        for i in range(data_num):
            for j in range(4):
                x_train[i,j,:] = (x_tmp[i][j] - x_tmp[i][3][-1]) / (2 * ran) + 0.5
        x_train = x_train.astype(np.float16)
        x_train = np.reshape(x_train, (data_num, 4, width, 1))

        # ラベルの準備
        y_tmp = df_train['prob'].values
        y_train = np.empty((data_num, 2))
        for i in range(data_num):
            y_train[i,:] = y_tmp[i]
        y_train = y_train.astype(np.float16)

        if(test == True and self.df_test is not None): # 検証データを作成する
            df_test = self.df_test
            thr_drop = x_range[0]
            thr_rise = x_range[1]
            df_test = df_test.query('drop <= @thr_drop')
            df_test = df_test.query('rise <= @thr_rise')
            x_tmp = df_test['data'].values
            data_num = len(x_tmp)
            width = len(x_tmp[0][0])
            x_test = np.empty((data_num, 4, width))
            for i in range(data_num):
                for j in range(4):
                    x_test[i,j,:] = (x_tmp[i][j] - x_tmp[i][3][-1]) / (2 * ran) + 0.5
            x_test = x_test.astype(np.float16)
            x_test = np.reshape(x_test, (data_num, 4, width, 1))
            y_tmp = df_test['prob'].values
            y_test = np.empty((data_num, 2))
            for i in range(data_num):
                y_test[i,:] = y_tmp[i]
            y_test = y_test.astype(np.float16)
            validation_data = (x_test, y_test)
        else: # 検証データを作成しない
            validation_data = None

        batch_size = 64 # ミニバッチサイズ

        '''
        while True:
            if len(x_train) / 500 > batch_size * 2:
                batch_size = batch_size * 2
            else:
                break
        '''

        # データ拡張
        def custom_augmentation(data):
            '''
            last_element = data[len(data)-1]
            r = np.max(data) - np.min(data)
            data = data + np.random.uniform(-0.01*r, 0.01*r, size=data.shape)
            data[(data > 1.0)] = 1.0
            data[(data < 0.0)] = 0.0
            data[len(data)-1] = last_element
            '''
            return data
        if riverse:
            x_riverse = 1.0 - x_train
            y_riverse = np.flip(y_train.copy(), axis=1)
            x_train = np.concatenate((x_train, x_riverse), axis=0)
            y_train = np.concatenate((y_train, y_riverse), axis=0)
        datagen = ImageDataGenerator(preprocessing_function=custom_augmentation)
        generator = datagen.flow(x_train, y_train, batch_size=batch_size)

        
        # モデル構造
        model = Sequential()
        input_shape = x_train.shape[1:]
        kernel_size = 3

        # 4, w, 1
        model.add(Conv2D(16, (4, kernel_size), input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((2,2), padding='same'))

        # 1, w/2, 16
        model.add(Conv2D(32, (2, kernel_size), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, (2, kernel_size), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((2,2), padding='same'))

        # 1, w/4, 32
        model.add(Conv1D(32, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv1D(32, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((1,2), padding='same'))

        # 1, w/8, 32
        model.add(Conv1D(64, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv1D(64, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((1,2), padding='same'))

        # 1, w/16, 64
        model.add(Conv1D(64, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv1D(64, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((1,2), padding='same'))

        # 1, w/32, 64
        model.add(Conv1D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv1D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D((1,2), padding='same'))
        

        # 1, w/64, 128
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid')) #　第1引数：分類させる数

        # モデルのコンパイル
        model.compile(loss='binary_crossentropy', # 誤差関数
                    optimizer='adam', # 最適化手法
                    metrics=['accuracy'])
        

        if pretrained is not None and os.path.exists(pretrained): # 追加学習
            model = load_model(pretrained,custom_objects={'BatchNormalization': BatchNormalization})
            epochs = 100
        else: # 初めから学習
            epochs = 1000

        if custom_epochs is not None: # エポック数を指定するとき
            epochs = custom_epochs

        # チェックポイントによるモデルの保存 (検証誤差が向上した場合のみ保存する)
        self.makeDir('./tmp')
        checkpoint = ModelCheckpoint(
                                    epoch_per_save=1,
                                    filepath='./tmp/tmp_model_{epoch:03d}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True
                                    )

        #学習
        model.fit_generator(
                  generator=generator,
                  steps_per_epoch=len(x_train) // batch_size,
                  epochs=epochs,
                  class_weight=FXtrading.cal_weight(y_train),
                  callbacks=[checkpoint],
                  validation_data=validation_data,
                  )
            
        #モデルを保存する
        model = load_model('./tmp/tmp_model.h5', custom_objects={'BatchNormalization': BatchNormalization}) #読み込むモデル
        model = model.save(path)
        os.remove('./tmp/tmp_model.h5')

        if set_model:
            self.model = load_model(path, custom_objects={'BatchNormalization': BatchNormalization})
            self.makeDir('./data')
            self.model.save('./data/model_current.h5')
            self.model_path = './data/model_current.h5'
            self.addDictionary(key=self.model_path, data=x_range, path='./data/x_range.pkl')

        return



    def makeDir(self, dirpath): # ディレクトリを作る

        DIR = dirpath
        if not os.path.exists(DIR):  # ディレクトリが存在しない場合、作成する。
            os.makedirs(DIR)

        return DIR


    def testModel(self, cmx=False, mode='BoS', prob=0.999): #モデルを評価する

        with open('./data/x_range.pkl', 'rb') as p:
            d = pickle.load(p)
        x_range = d[self.model_path]

        df_test = self.df_test.query('group != 1')
        thr_drop = x_range[0]
        thr_rise = x_range[1]
        df_test = df_test.query('drop <= @thr_drop')
        df_test = df_test.query('rise <= @thr_rise')
        x_tmp = df_test['data'].values
        y_test = df_test['group'].values
        y_test[y_test==2] = 1
        
        data_num = len(x_tmp)
        width = len(x_tmp[0])
        x_test = np.empty((data_num, width))
        for i in range(data_num):
            x_test[i,:] = (x_tmp[i] - x_tmp[i][-1]) / (x_range[0] + x_range[1]) + 0.5
        x_test = x_test.astype(np.float16)
        x_test = np.reshape(x_test, (data_num, 1, width, 1))

        y_test = y_test.astype(np.int64)
        y_test = np_utils.to_categorical(y_test)
        
        output = np.array(self.model(x_test))
        output[output >= prob] = 1
        output[output < prob] = 0
        sum = np.sum(output, axis=0)
        temp = np.array(y_test)
        temp2 = temp - np.flip(temp, axis=1) # 予想が外れたとき，評価値がマイナスになるように
        result = np.sum(temp2*output, axis=0) # 勝ち越し

        print('「売り」の回数：%d' %sum[0])
        print('「売り」の勝ち越し：%d' %result[0])
        print(f'「売り」の成功率：{(sum[0]+result[0])/(2*sum[0])}')
        print('「買い」の回数：%d' %sum[-1])
        print('「買い」の勝ち越し：%d' %result[-1])
        print(f'「買い」の成功率：{(sum[-1]+result[-1])/(2*sum[-1])}')
        print(f'score：{(sum[0]+sum[-1]+result[0]+result[-1])/(2*sum[0]+2*sum[-1])}')
        print(f'取引しなかった回数：{output.shape[0]-(sum[0]+sum[-1])}')

        if cmx: # 混同行列
            predict_classes = np.argmax(self.model.predict(x_test), axis=-1)
            true_classes = np.argmax(y_test, axis=1)
            FXtrading.print_cmx(true_classes, predict_classes)    




    def print_cmx(y_true, y_pred): # 混同行列の表示
        labels = sorted(list(set(y_true)))
        cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
        df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True)
        plt.show()