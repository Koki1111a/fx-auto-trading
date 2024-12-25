from datetime import datetime, timezone
import functions
import time


T = functions.FXtrading()
T.adjustProfitLine(modelCheck=True) # 利確ラインの調節
modelUpdateFlag = False # モデルを更新するかどうか
waitFlag = [False, False] # 待機中を示すフラグ
period = T.forwardSteps * T.tf # ポジションの時間制限[秒]


while 1:
    time.sleep(1) # パソコンの負担を減らす
    T.update() # パラメータの更新
    positions = T.positions # ポジション情報の取得
    now = datetime.now() # 現在時刻
    fictitiousPosition = T.fictitiousTrade() # 架空ポジションの情報

    if(waitFlag[0] and not waitFlag[1]):
        print('> 待機中', end='\r', flush=True)
        waitFlag[1] = True

    if not modelUpdateFlag:
        modelUpdateFlag = T.CanUpdateModel(now) # モデルを更新するか

    if modelUpdateFlag:
        print('> モデルの更新')
        date_to = datetime(now.year, now.month, 1, tzinfo=timezone.utc) # どの時期までのデータ取得したいか
        print('>> 利確ラインの更新')
        T.adjustProfitLine()
        print(f'利確ラインは{T.profitLine}pipsです．')
        print('>> 学習データの更新')
        T.setDf_train(date_to) # 学習データの更新
        print('>> 学習')
        T.trainModel(set_model=True, pretrained='./data/model_current.h5')
        modelUpdateFlag = False
        waitFlag[0] = False
        waitFlag[1] = False

    act = T.makeDecide()

    if(T.spread <= 1.0 * T.pip # スプレッドが十分小さいときのみ取引可能
       and fictitiousPosition is None # 架空の取引を行っていないか(買いポジションを持っているのに売りポジションがほしいときなどは，代わりに架空のポジションを持つ)
       and len(positions) < 5 # ポジションを持ちすぎないように
       and (len(positions) == 0 # ポジションがない (1)
            or (T.get_positionTime(-1) > T.tradeFrame # 最後にポジションを持ってからの時間が一定以上 (2-1)
                and T.margin_level > 300.0 # 証拠金維持率が一定以上 (2-2)
                )
            )
       ):
        
        # 持っている直近のポジションと，持ちたいポジションが矛盾するとき
        if len(positions) > 0 and act + (1-positions[-1][5])*2 == 2:

            # 一番指値から遠いポジションを手放す
            worstProfit = positions[0][15] / T.positions[0][9] # 利益/ロット数
            worstPosition = 0 # 最も利益が低いポジションのインデックス
            for i in range(1, len(positions)):
                profit = positions[i][15] / T.positions[i][9]
                if worstProfit > profit:
                    worstProfit = profit
                    worstPosition = i
            if act == 0: # 売りの架空注文
                print('> 買いポジションの解消 (売りの架空注文)')
                T.fictitiousTrade(mode='sell')
                T.sell(worstPosition)
                waitFlag[0] = False
                waitFlag[1] = False
            if act == 2: # 買いの架空注文
                print('> 売りポジションの解消 (買いの架空注文)')
                T.fictitiousTrade(mode='buy')
                T.buy(worstPosition)
                waitFlag[0] = False
                waitFlag[1] = False

        else:

            if act == 0: # 売りの注文
                print('> 売り注文')
                T.sell()
                waitFlag[0] = False
                waitFlag[1] = False
            elif act == 1: # なにもしない
                waitFlag[0] = True
            elif act == 2: # 買いの注文
                print('> 買い注文')
                T.buy()
                waitFlag[0] = False
                waitFlag[1] = False


    else:
        waitFlag[0] = True
    

    
    for i in range(len(positions)): # 持っているポジションを手放すか判断

        order_type = (1-positions[i][5])*2 # sellかbuyか取得

        if(T.get_positionTime(i) > period # 一定時間経っているか
           and ( (positions[i][15] > T.commission * T.positions[i][9] and act == 1) # 手数料分の利益があり，これ以上利益が望めない場合 (1)
                or (act != 1 and act != order_type) # ポジションに逆行する動きが予測される場合 (2)
                )
           ):
            

            if order_type == 0: # sellポジション
                print('> 売りポジションの解消')
                T.buy(i)
                waitFlag[0] = False
                waitFlag[1] = False
            elif order_type == 2: # buyポジション
                print('> 買いポジションの解消')
                T.sell(i)
                waitFlag[0] = False
                waitFlag[1] = False