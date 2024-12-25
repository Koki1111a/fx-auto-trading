import functions
from datetime import datetime, timezone


date_from = datetime(2017, 5, 1, 0, tzinfo=timezone.utc) #どの時期からのデータを取得したいか
date_to = datetime(2022, 5, 30, tzinfo=timezone.utc) #どの時期までのデータ取得したいか
T = functions.FXtrading(timeFrame='M10', forwardSteps=128, backwardSteps=256, profitLine=40, date_from=date_from, date_to=date_to)
#T.adjustProfitLine(now=date_to)
print(f'利確ライン：{T.profitLine}')
T.downloadData(path='./experiment/M10_trinData.pkl', noiseCut=True, checkContinuity=True)
#T.downloadData()