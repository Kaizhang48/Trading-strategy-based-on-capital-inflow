# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy  as np
import dealwithdata as dwd
from backtesting_class import Backtest
import datetime

starttime=datetime.datetime.now()
#----------------------------------------------------导入数据
c=pd.read_csv('C:\csvdata\python_ZZcontnetCLOSE.csv')
bench=pd.read_csv('C:\csvdata\python_ZZdaliydata.csv')
mkv=pd.read_csv('C:\csvdata\python_ZZcontnetMarketValue.csv')
h=pd.read_csv('C:\csvdata\python_ZZcontnetHIGH.csv')
l=pd.read_csv('C:\csvdata\python_ZZcontnetLOW.csv')
o=pd.read_csv('C:\csvdata\python_ZZcontnetOPEN.csv')
pre_superinflow=pd.read_csv('C:\csvdata\python_ZZcontnetSUPERINFLOW.csv')
pre_superoutflow=pd.read_csv('C:\csvdata\python_ZZcontnetSUPEROUTFLOW.csv')
to=pd.read_csv('C:\csvdata\python_ZZcontnetTURNOVER.csv')
uwo=pd.read_csv('C:\csvdata\python_ZZcontnetOPEN(unweight).csv')
uwc=pd.read_csv('C:\csvdata\python_ZZcontnetCLOSE(unweight).csv')
rsuperinflow=dwd.dealwithdata(pre_superinflow).values.T
rsuperoutflow=dwd.dealwithdata(pre_superoutflow).values.T
rsupernetflow=rsuperinflow-rsuperoutflow

#-----------------------------------------------------导入数据完毕
zhangkai=Backtest(o,h,l,c,uwc,uwo,to,mkv,bench,[3],[0.3,0.4,0.3],[0]) 
zhangkai.KDJ() 
zhangkai.storagedesign()

KDJthreshhold=80
netthreshhold=0.18
stoploss=0.5
maxcheckdate=20
maxdecrease=0.09
pre_spclose=np.zeros(zhangkai.CLOSE.shape)
spclose=pre_spclose.copy()
pre_spbenchclose=spclose.copy()
spbenchclose=spclose.copy()
finalclose=spclose.copy()
for i in range(len(zhangkai.CLOSE[:,0])):
    loc=np.where(zhangkai.CLOSE[i,:]==0)
    benchclose=zhangkai.BENCHMARK['close']
    rbenchclose=benchclose.values.T
    if len(loc[0])==0:        #已经上市
        pre_spclose[i,1:]=np.diff(zhangkai.CLOSE[i,0:])/zhangkai.CLOSE[i,0:-1]
        spclose[i,0:]=np.cumprod(1+pre_spclose[i,0:])
        pre_spbenchclose[i,1:]=np.diff(rbenchclose[0:])/rbenchclose[0:-1]
        spbenchclose[i,0:]=np.cumprod(1+pre_spbenchclose[i,0:])
        finalclose[i,0:]=spclose[i,0:]-spbenchclose[i,0:]
    else :
        b=loc[0][-1]
        pre_spclose[i,b+2:]=np.diff(zhangkai.CLOSE[i,b+1:])/zhangkai.CLOSE[i,b+1:-1]
        spclose[i,b+1:]=np.cumprod(1+pre_spclose[i,b+1:])
        pre_spbenchclose[i,b+2:]=np.diff(rbenchclose[b+1:])/rbenchclose[b+1:-1]
        spbenchclose[i,b+1:]=np.cumprod(1+pre_spbenchclose[i,b+1:])
        finalclose[i,b+1:]=spclose[i,b+1:]-spbenchclose[i,b+1:]         
cishu=0
for date in range(zhangkai.observe,len(zhangkai.tradedate)-1):
    cishu=cishu+1  
    print cishu
    
    zhangkai.update_account(date)
 #------------------------------------将不同市值大小的公司分开   
    zhangkai.divid_universe_by_marketvalue(date)
#--------------------------------------分离完毕
    
#--------------------------------------构建salelist
    salelist=np.array([])
    storage_num=zhangkai.storage_num
    storage_room=zhangkai.storage_room
    storage=zhangkai.storage
    buy=zhangkai.buy
    
    for i in range(storage_num):
        for p in range(storage_room[i]):
            if storage[i,0,p,date+1]!=-1:
                stock=storage[i,0,p,date+1]
                stock=int(stock)
                temp=np.where(buy[i,0,stock,:date+1]!=0)
                buydate=temp[-1][-1]
                if date>=buydate+zhangkai.changelength-1 and ((zhangkai.K[stock,date]>KDJthreshhold and zhangkai.D[stock,date]>KDJthreshhold and zhangkai.K[stock,date-1]>zhangkai.D[stock,date-1] and zhangkai.K[stock,date]<zhangkai.D[stock,date]) or zhangkai.CLOSE[stock,date]/zhangkai.CLOSE[stock,buydate]<=1-stoploss):
                    salelist=np.append(salelist,stock)
#----------------------------------------salelist构建完毕
 #---------------------------------------下卖单                   
    if len(salelist)!=0:
        for stock in salelist:
            zhangkai.sell_to(stock,0,date)
 #----------------------------------------下卖单
 #---------------------------------------构建buylist
    buylist=np.array([])
    stockmktvalue=np.array([])
  
    for j in range(2,-1,-1):
        for stock in zhangkai.universe[j,0,0]:#遍历股票池
            stock=int(stock)
            if stock!=-1:  
                threshdate=0
                for checkdate in range(1,maxcheckdate+1):
                     if finalclose[stock,date]/finalclose[stock,date-checkdate]<=1-maxdecrease:
                         threshdate=date-checkdate                                                  
                if threshdate!=0:
                    if sum(rsupernetflow[stock,threshdate:date+1])/sum(zhangkai.TURNOVER[stock,threshdate:date+1])>=netthreshhold: 
                        
                        buylist=np.append(buylist,stock)
                        stockmktvalue=np.append(stockmktvalue,zhangkai.Usymbol[j])
    if len(buylist)!=0:
        buylist=np.vstack([buylist,stockmktvalue])
#---------------------------------------------buylist构建完毕
#---------------------------------------------下买单
    if len(buylist)!=0:
        for j in range(2,-1,-1): 
            fbuylist=buylist[0,:][buylist[1,:]==zhangkai.Usymbol[j]]#先买同一市值水平的
            if len(fbuylist)==0:           
                continue
            for stock in fbuylist: 
                for i in range(storage_num):#0,1,......,storage_num-1
                    open_apv=0
                    apv_num=np.array([])
                    if len(apv_num)<=storage_num :#如果仍与仓位没有开,就继续检查是否允许开仓
                        if i!=0:#如果不是第一个仓
                            loc=np.where(storage[i-1,0,0,:]!=-1)
                            if len(loc[-1])==0:                                       
                                open_apv=np.Inf#下一个仓自然不许开,
                            else:
                                open_apv=loc[-1][0]
                                apv_num=np.append(apv_num,i)
                        else:#第一个仓的apv就是observe
                            open_apv=zhangkai.observe 
                            apv_num=np.append(apv_num,i)
                        
                    if date>=open_apv+zhangkai.build_stgap[i]: 
                        zhangkai.checkempty(storage[i,0,:,date+1])
                        if zhangkai.leftspace!=0:#有空位去买股票
                            amount=zhangkai.cash[i,0,0,date+1]/(zhangkai.leftspace*zhangkai.OPN[stock,date+1])
                            zhangkai.order(i,j,stock,amount,date)
#--------------------------------------------------下买单                            
                
    zhangkai.summarize_account(date)
zhangkai.visualization()
endtime=datetime.datetime.now()
interval=(endtime-starttime).seconds
print "the running time is "+str(interval)    
