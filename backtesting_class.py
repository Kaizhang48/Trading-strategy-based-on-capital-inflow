# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 12:40:05 2016
#夏普率2.11版本
@author: KaiZhang
"""
import pandas as pd
import numpy  as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import volume_control as vc
import dealwithdata as dwd
import maxdrawdown_maker as maxdd
class Backtest:
    def __init__(self,opn,high,low,close,uwclose,uwopn,turnover,mktvlu,benchmark,\
        storage_room,level_pct,build_stgap,cash_total=2000000,stptax_rate=0.001,\
        commision_rate=0.0005,lowest_commision=5,changelength=1,limit_coeff=0.1,observe=99): 
        self.OPN=dwd.dealwithdata(opn).values.T
        self.HIGH=dwd.dealwithdata(high).values.T
        self.LOW=dwd.dealwithdata(low).values.T
        self.CLOSE=dwd.dealwithdata(close)
        self.stkname=self.CLOSE.columns
        self.tradedate=self.CLOSE.index
        self.CLOSE=dwd.dealwithdata(close).values.T
        self.UWCLOSE=dwd.dealwithdata(uwclose).values.T
        self.UWOPN=dwd.dealwithdata(uwopn).values.T
        self.MARKETVALUE=dwd.dealwithdata(mktvlu).values.T
        self.universe_num=len(self.CLOSE[0,:])
        self.storage_room=storage_room
        self.storage_num=len(storage_room)
        self.level_pct=level_pct
        self.stk_level=len(level_pct)
        self.Usymbol=range(1,self.stk_level+1)
        self.build_stgap=build_stgap
        self.cash_total=cash_total
        self.stptax_rate=stptax_rate
        self.commision_rate=commision_rate
        self.lowest_commision=lowest_commision
        self.changelength=changelength
        self.observe=observe        
        self.BENCHMARK=benchmark
        self.stk_level=len(self.level_pct)
        self.Usymbol=range(1,self.stk_level+1)              
        [a,s]=self.CLOSE.shape       
        self.storage_name=DataFrame()        
        for i in range(self.storage_num):
            temp1=np.array([])
            temp2=np.array([])
            for j in range(self.storage_room[i]):
                temp1=np.append(temp1,'storage'+str(i+1))
                temp2=np.append(temp2,'position'+str(j+1))                
#            temp=DataFrame(np.zeros([3,s]),columns=self.tradedate,index=[temp1,temp2],dtype=str)
            temp=DataFrame(np.zeros([self.storage_room[0],s]),columns=self.tradedate,index=[temp1,temp2],dtype=str)
            self.storage_name=pd.concat([self.storage_name,temp],axis=0)
        self.holdnum=np.zeros([1,s])
        self.storage=-np.ones([self.storage_num,1,self.storage_room[0],s])
        self.storage_mktvlu=np.zeros([self.storage_num,1,self.storage_room[0],s])
        self.cash=np.zeros([self.storage_num,1,1,s])
        self.capital=self.cash.copy()
        self.cashross=np.zeros([1,s])
        self.capitalross=self.cashross.copy()
        self.cashross[0,:self.observe+1]=2000000
        self.buy=np.zeros([self.storage_num,1,a,s])
        self.sale=self.buy.copy()
        self.volume=self.buy.copy()
        self.per_volume=self.buy.copy()
        self.stamptax=self.buy.copy()
        self.sale_commision=self.buy.copy()
        self.buy_commision=self.buy.copy()
        self.each_rt=np.array([])
        self.each_cost=np.array([])
        self.first_date=0
        self.trade_num=0   
    def divid_universe_by_marketvalue(self,date):
 #       self.__creat_deeperdata()
        c=self.MARKETVALUE[:,date].copy().T
        c=np.vstack([np.arange(len(self.stkname)),c])
        c=c[:,np.lexsort(c)]
        [aa,bb]=c.shape
        cutline=[0]  
        for level in self.Usymbol:#Usymbl=[1,2,3]
            temp=len(self.stkname)*sum(self.level_pct[0:level])
            cutline.append(temp)
        cut_num=np.diff(np.array(cutline))        
        self.universe=-np.ones([self.stk_level,1,aa,max(cut_num)])
        for level in range(len(cutline[1:])):
            temp=c[:,cutline[level]:cutline[level+1]]
            self.universe[level,0,:,:len(temp[0])]=temp                
    def storagedesign(self,utype='same',s_num=1,stk_level=3,shape=[1.5,1.5,0]):
#        self.__creat_deeper()
        self.U=np.zeros([s_num,1,1,stk_level])
        if utype=='same':            
            for i in range(self.storage_num):
                self.U[i,0,0,:]=shape
        elif utype=='different':
            for i in range(self.storage_num):
                self.U[i,0,0,:]=input("please enter the number of different levels of stock that you want in the "+str(i+1)+"th storage :")                
    def checkempty(self,target):
        loc=np.where(target==-1)
        self.leftspace=len(loc[0])      
        #把order的职责单纯化,判断的东西移出去;
        
    def order(self,i,universe_level,stock,amount,date,limit='strict'):
        i=int(i)
        universe_level=int(universe_level)
        stock=int(stock)
        date=int(date)
        
        volume=self.volume
        per_volume=self.per_volume
        cash=self.cash
        buy_commision=self.buy_commision
        buy=self.buy
        storage=self.storage
        storage_name=self.storage_name
        storage_mktvlu=self.storage_mktvlu
        am=amount
#python 数值=数值,是赋值,不是指针
#        若不满仓
#        if len(loc[0])==self.storage_room[i]:#如果股票之前没有持仓
        self.checkempty(storage[i,0,:,date+1])
        
        if self.leftspace!=0:
            
            tttt=np.where(storage_mktvlu[i,0,:,date+1]==self.Usymbol[universe_level])
            if len(tttt[0])<self.U[i,0,0,universe_level]:  
               stopprice=round(self.UWCLOSE[stock,date]*(1+self.limit_coeff)*100)/100
               stopdprice=round(self.UWCLOSE[stock,date]*(1-self.limit_coeff)*100)/100
               if self.UWOPN[stock,date+1]<stopprice and self.UWOPN[stock,date+1]>stopdprice:
                   if self.TURNOVER[stock,date+1]!=0:#没有停牌
                       if limit=='strict':
                            am=vc.volume_round(am)
                            am,bc=vc.volume_limit(cash[i,0,0,date+1],am,self.OPN[stock,date+1],self.commision_rate,self.lowest_commision)
                            per_volume[i,0,stock,date+1]=am
                            volume[i,0,stock,date+1]=volume[i,0,stock,date+1]+am

#                           volume[i,0,stock,date+1],bc=vc.volume_limit(cash[i,0,0,date+1],volume[i,0,stock,date+1],self.OPN[stock,date+1],self.commision_rate,self.lowest_commision) 
                       else:
                           volume[i,0,stock,date+1]=volume[i,0,stock,date+1]+am
                           per_volume[i,0,stock,date+1]=am
                           bc=am*self.OPN[stock,date+1]*self.commision_rate
                           if bc<self.lowest_commision:
                               bc=self.lowest_commision

                               
                       if volume[i,0,stock,date+1]!=0:
                           cash[i,0,0,date+1]=cash[i,0,0,date+1]-am*self.OPN[stock,date+1]-bc
                           buy_commision[i,0,stock,date+1]=bc
                           buy[i,0,stock,date+1]=self.OPN[stock,date+1]
                           loc=np.where(storage[i,0,:,date+1]==stock)
                           if len(loc[-1])==0:#如果之前没买过
                               temp=np.where(storage[i,0,:,date+1]==-1)
                               loc=temp[0][0]
                               storage[i,0,loc,date+1]=stock
                               storage_name.loc['storage'+str(i+1)].iat[loc,date+1]=self.stkname[stock]
                               storage_mktvlu[i,0,loc,date+1]=self.Usymbol[universe_level]                               
                               
                           if self.first_date==0:
                               self.first_date=date  
    def sell_to(self,stock,to_amount,date):
  ######待做工作,改进卖出方法,从而达到卖出时,可以根据不同的买入时间来计算收益
       #在卖出时,找出不同的买入时间,从持仓最久开始卖,不同买入时间收益不同
       #算出在两次卖出之间,有几次买入
       #若之前没有卖出,则算从第一天到现在一共有几次买入
       #volume每日依前日更新,但是per_volume不是
       #buy和sale矩阵也不是每日依前一日更新
 #-----------------------------------------------------------------------------------------------------------------------
       #初步验证正确
       #需进一步验证
       #思路:每一次的买入值都记录在per_volume里, volume中对应的值是当日之前per_volume中的总和
       #每一次要卖出时,查出,卖出日之前所有天数中,有买入的天数,当天买入量,买入价格,从离当前最近的天数中开始卖,边卖边记录收益,成本,佣金,直到,所剩余的要卖出的数量<某一天买入的数量
       #这一日之后的之前又买入的天数中的买入全部被卖出,该日还剩下一些,以及这一日之前的买入还没被卖出
       #就把这一部分放在这,不用管,每一次都是从离当前最近的天数开始卖
#---------------------------------------------------------------------------
#反思:是否应该从持仓最久的部分开始卖
        per_volume=self.per_volume
        volume=self.volume
        cash=self.cash
        buy_commision=self.buy_commision
        buy=self.buy
        storage=self.storage
        storage_name=self.storage_name
        storage_mktvlu=self.storage_mktvlu
        sale=self.sale
        stamptax=self.stamptax
        sale_commision=self.sale_commision
        stock=int(stock)
        loc=np.where(storage[:,:,:,date+1]==stock)
        for temploc in zip(loc[0],loc[1],loc[2]):
            stopprice=round(self.UWCLOSE[stock,date]*(1+self.limit_coeff)*100)/100
            stopdprice=round(self.UWCLOSE[stock,date]*(1-self.limit_coeff)*100)/100
            if self.UWOPN[stock,date+1]<stopprice and self.UWOPN[stock,date+1]>stopdprice:
                if self.TURNOVER[stock,date+1]!=0:                   

                    self.b_temp=np.where(per_volume[temploc[0],temploc[1],stock,:date+1]!=0)                    
                    buydate=self.b_temp[0]
                    self.td_vol_in_period=per_volume[temploc[0],temploc[1],stock,buydate]#时间段内买入的数量
                    self.td_vol_in_period=Series(self.td_vol_in_period,index=buydate)          
                    sale[temploc[0],temploc[1],stock,date+1]=self.OPN[stock,date+1]#此次卖出价格
                    sale_volume=volume[temploc[0],temploc[1],stock,date+1]-to_amount#此次卖出数量
                    volume[temploc[0],temploc[1],stock,date+1]=to_amount
                    salevalue=sale_volume*self.OPN[stock,date+1]#此次卖出股票价值
                    stax=salevalue*self.stptax_rate#此次卖出股票所要缴纳的印花税
                    stamptax[temploc[0],temploc[1],stock,date+1]=stax
                    sale_commision[temploc[0],temploc[1],stock,date+1]=salevalue*self.commision_rate#此次卖出股票所要缴纳的佣金
                    if sale_commision[temploc[0],temploc[1],stock,date+1]<self.lowest_commision:
                        sale_commision[temploc[0],temploc[1],stock,date+1]=self.lowest_commision
                    scommision=sale_commision[temploc[0],temploc[1],stock,date+1]
                    cash[temploc[0],0,0,date+1]=cash[temploc[0],0,0,date+1]+salevalue-stax-scommision
                    temp_bc=0
                    temp_sv=sale_volume
                    temp_per_rt=0
                    temp_cost=0
                    cc=list(self.td_vol_in_period.index)
                    for jj in cc:
                        #jj是买入的天数,从最远的一天开始算                       
                        if temp_sv>=self.td_vol_in_period[jj]:
                            temp_bc=temp_bc+buy_commision[temploc[0],temploc[1],stock,jj]                            
                            temp_per_rt=temp_per_rt+(self.OPN[stock,date+1]-self.OPN[stock,jj])*self.td_vol_in_period[jj]
                            temp_cost=temp_cost+self.td_vol_in_period[jj]*self.OPN[stock,jj]
                            per_volume[temploc[0],temploc[1],stock,jj]=0
                            temp_sv=temp_sv-self.td_vol_in_period[jj]

                        else:
                            temp_bc=temp_bc+temp_sv*self.OPN[stock,jj]*self.commision_rate
                            temp_per_rt=temp_per_rt+(self.OPN[stock,date+1]-self.OPN[stock,jj])*temp_sv
                            temp_cost=temp_cost+temp_sv*self.OPN[stock,jj]
                            per_volume[temploc[0],temploc[1],stock,jj]=per_volume[temploc[0],temploc[1],stock,jj]-temp_sv
                            break
                        
                    per_rt=temp_per_rt-stax-scommision-temp_bc
                    per_rt_rate=per_rt/temp_cost
                    if to_amount==0:          
                        storage[temploc[0],temploc[1],temploc[2],date+1]=-1
                        storage_name.loc['storage'+str(temploc[0]+1)].iat[temploc[2],date+1]=str(0.0)
                        storage_mktvlu[temploc[0],temploc[1],temploc[2],date+1]=0

                    self.trade_num=self.trade_num+1 
                    self.each_rt=np.append(self.each_rt,per_rt_rate) 
                    self.each_cost=np.append(self.each_cost,temp_cost)  
                       
    def KDJ(self,KDJa=34,KDJb=3,KDJc=3):
        close=self.CLOSE
        low=self.LOW
        high=self.HIGH
        self.nhigh=np.zeros(high.shape)
        self.nlow=np.zeros(low.shape)
        nhigh=self.nhigh
        nlow=self.nlow
        nlow[:,:KDJa-1]=low[:,:KDJa-1]
        nhigh[:,:KDJa-1]=high[:,:KDJa-1]
        for i in range(KDJa-1,len(nlow[0])):        
            nlow[:,i]=low[:,(i-KDJa+1):i+1].min(axis=1)       
            nhigh[:,i]=high[:,(i-KDJa+1):i+1].max(axis=1)           
        nhigh=nhigh+0.00000001
        self.rsv=((close-nlow)/(nhigh-nlow))*100
#rsv中的high low 是a日最高最低
        self.K=np.zeros(self.rsv.shape)
        self.D=self.K.copy()
        for i in range(KDJb):
            self.K[:,KDJb-1:]=self.K[:,KDJb-1:]+self.rsv[:,KDJb-1-i:len(self.rsv[0])-i]
        self.K=self.K/KDJb
        self.K[:,:KDJb-1]=self.rsv[:,:KDJb-1]
#k是rsv b日平均
        for i in range(KDJc):
            self.D[:,KDJc-1:]=self.D[:,KDJc-1:]+self.K[:,KDJc-1-i:len(self.rsv[0])-i]
        self.D=self.D/KDJc
        self.D[:,:KDJc-1]=self.K[:,:KDJc-1]
#D是k的c日平均
        self.J=3*self.K-2*self.D       
    def update_account(self,date):
        cash=self.cash
        storage=self.storage
        storage_name=self.storage_name
        storage_mktvlu=self.storage_mktvlu
        volume=self.volume       
        for i in range(self.storage_num):
            if date==self.observe:
                cash[i,0,0,:self.observe+1]=self.cash_total/self.storage_num                
            storage[i,0,:,date+1]=storage[i,0,:,date]
            storage_name.loc['storage'+str(i+1)].ix[:,date+1]=storage_name.loc['storage'+str(i+1)].ix[:,date]
            storage_mktvlu[i,0,:,date+1]=storage_mktvlu[i,0,:,date]
            volume[i,0,:,date+1]=volume[i,0,:,date]
            cash[i,0,0,date+1]=cash[i,0,0,date]        
    def summarize_account(self,date):
        cash=self.cash
        storage=self.storage       
        volume=self.volume
        cash=self.cash       
        capital=self.capital
        capitalross=self.capitalross
        holdnum=self.holdnum
        cashross=self.cashross                
        temp=0    
        for j in range(self.storage_num):
            loc=np.where(storage[j,0,:,date+1]!=-1)
            temp=temp+len(loc[0])       
            for stock in storage[j,0,:,date+1]:
                stock=int(stock)
                if stock!=-1:
                    capital[j,0,0,date+1]=capital[j,0,0,date+1]+volume[j,0,stock,date+1]*self.CLOSE[stock,date+1]                                                     
            cashross[0,date+1]=cashross[0,date+1]+cash[j,0,0,date+1]
            capitalross[0,date+1]=capitalross[0,date+1]+capital[j,0,0,date+1] 
        holdnum[0,date+1]=temp        
    def data_analyse(self):
        each_rt=self.each_rt
        cashross=self.cashross
        tradedate=self.tradedate
        capitalross=self.capitalross        
        benchclose=self.BENCHMARK['close']
        rbenchclose=benchclose.values.T
        bench_rt=Series((rbenchclose[self.first_date:]-rbenchclose[self.first_date])/rbenchclose[self.first_date],index=benchclose.index[self.first_date:])             
        self.accountvalue=cashross+capitalross
        self.account_rt=(self.accountvalue-cashross[0,0])/cashross[0,0]
        self.account_rt=self.account_rt[0,self.first_date:]
        self.accountvalue=self.accountvalue[:,self.first_date:]
        self.account_rt=Series(self.account_rt,index=benchclose.index[self.first_date:])
        temp=np.vstack([self.account_rt.values,bench_rt.values])
        self.compare=DataFrame(temp.T,index=tradedate[self.first_date:],columns=['strategy','benchmark'])        
        per_d=np.diff(self.accountvalue[0,self.first_date:])/self.accountvalue[0,self.first_date:-1]
        self.accountvalue=Series(self.accountvalue[0,:],index=benchclose.index[self.first_date:])
        self.__annualized_rt=self.account_rt.values[-1]/len(tradedate)*242
        self.__win_rate=float(len(each_rt[each_rt>0]))/float(self.trade_num)
        self.__mean=np.mean(per_d)
        self.__std=np.std(per_d)
        self.__sharp_ratio=(self.__mean/self.__std)*16
    def min_variance_weight(self,opt_list,opt_list_vb,miu,date,hist_window,stnum):
        
        self.m=np.array([])
        self.cap=np.array([])
        self.memberofcovariance=np.array([])
        self.member_vb=np.array([])
        pre_cov=np.array([])
        st_member=np.array([])
        count=0
        for stk,vb in zip(opt_list,opt_list_vb):
            if stk!=-1:
                count=count+1
                self.member_vb=np.append(self.member_vb,vb)
                self.memberofcovariance=np.append(self.memberofcovariance,stk)
                #calculate the value of stock that I built position                
                if count==1:#其中包含仓内的股票和新选的股票
                    pre_cov=self.CLOSE[stk,date-hist_window:date+1]
                elif count>1:
                    pre_cov=np.vstack([pre_cov,self.CLOSE[stk,date-hist_window:date+1]])
                    
                temp1=(self.CLOSE[stk,date]-self.CLOSE[stk,date-hist_window])/self.CLOSE[stk,date-hist_window]
                self.m=np.append(self.m,temp1/hist_window*242)
                
        for i in self.storage[stnum,0,:,date+1]:
            if i!=-1:
                loc=np.where(self.buy[stnum,0,i,:date+1]!=0)
                buydate=loc[-1][-1]
                self.cap=np.append(self.cap,self.CLOSE[i,date]*self.volume[stnum,0,i,buydate])
                st_member=np.append(st_member,i)
        self.cap=np.vstack([st_member,self.cap])    
            
        self.tt=self.cash[stnum,0,0,date]+sum(self.cap[1,:])
        self.real_w=np.vstack([st_member,self.cap[1,:]/self.tt])
        self.real_w=pd.Series(self.real_w[1,:],index=self.real_w[0,:])
               
        self.C=np.mat(np.cov(pre_cov))
        self.m=np.mat(self.m)
        self.u=np.mat(np.ones(self.m.shape))
        temp2=np.mat([miu,1])
        self.M=np.mat([[float((self.m*self.C**(-1))*self.m.T),float((self.m*self.C**(-1))*self.u.T)],[float((self.u*self.C**(-1))*self.m.T),float((self.u*self.C**(-1))*self.u.T)]])
        temp3=np.vstack([self.m*self.C**(-1),self.u*self.C**(-1)])
        self.w=(temp2*self.M**(-1))*temp3
        self.w=pd.Series(np.array(self.w)[0],index=self.memberofcovariance)
        self.member_vb=pd.Series(self.member_vb,index=self.memberofcovariance)
        self.w_change=pd.DataFrame({'w':self.w,'real_w':self.real_w,'vb':self.member_vb}).fillna(value=0)
        self.adjust=self.w_change['w']-self.w_change['real_w']
        
        

    def visualization(self):       
        self.data_analyse()
        holdnum=self.holdnum       
        tradedate=self.tradedate        
        accountvalue=self.accountvalue.values
        account_rt=self.account_rt.values
        fig,axs=plt.subplots(2,1,figsize=[20,10])
        axs[0].plot(self.compare['strategy'].values)
        axs[0].plot(self.compare['benchmark'].values)  
        temp=len(self.CLOSE[0][self.first_date:])
        temp1=self.tradedate[self.first_date:]
        axs[0].set_xticks(np.linspace(0,temp-1,7,dtype=int))
        axs[0].set_xticklabels(temp1[np.linspace(0,temp-1,7,dtype=int)],fontsize=10)
        axs[0].bar(range(len(holdnum[0,self.first_date:])),holdnum[0,self.first_date:]*0.15,width=0.01)
        accountvalue=np.zeros([1,len(self.accountvalue.values)])
        accountvalue[0,:]=self.accountvalue.values
        bloc_of_maxdd,loc_of_maxdd,self.__maxdrawdown=maxdd.maxdrawdown_maker(accountvalue)
        axs[0].plot(range(bloc_of_maxdd[1][0],loc_of_maxdd[1][0]+1),account_rt[bloc_of_maxdd[1][0]:loc_of_maxdd[1][0]+1],color='r',linewidth=4)
        axs[0].legend(['Strategy','Benchmark','The Time Slot of Maxdrawdown','The Number of Position'],loc='best',fontsize=15)
        loc=np.where(accountvalue==accountvalue.max())
        tempdate=tradedate[self.first_date:]
        maxrt=round(account_rt.max(),2)
        axs[0].scatter(loc[-1][0],account_rt[loc[-1][0]])
        axs[0].annotate("MAX: "+str(maxrt)+", At: "+str(tempdate[loc[-1][0]]),xy=(loc[-1][0],account_rt[loc[-1][0]]+0.01),xytext=(loc[-1][0],account_rt[loc[-1][0]]+0.3),arrowprops=dict(facecolor='black',arrowstyle="->"))
        temp=account_rt[-1]*100
        temp1=self.__annualized_rt*100
        temp2=self.__win_rate*100
        temp3=self.__maxdrawdown*100
        s=len(self.CLOSE[0,:])
        m=account_rt.max()
        axs[0].text(s-100,round(m)-1,'the duration of backtestign :'+str(tradedate[self.first_date])+"-"+str(tradedate[-1]))
        axs[0].text(s-100,round(m)-1.5,'cummulative rate of return :'+"%.2f"%temp+'%') 
        axs[0].text(s-100,round(m)-2,'annualized rate of return :'+"%.2f"%temp1+'%')
        axs[0].text(s-100,round(m)-2.5,'rate of win :'+"%.2f"%temp2+'%')
        axs[0].text(s-100,round(m)-3,'the mean of return :'+"%.4f"%self.__mean)
        axs[0].text(s-100,round(m)-3.5,'the std of return :'+"%.4f"%self.__std)
        axs[0].text(s-100,round(m)-4,'sharp_ratio :'+"%.2f"%self.__sharp_ratio)
        axs[0].text(s-100,round(m)-4.5,'maxdrawdown :'+"%.2f"%temp3+"%")
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Abnormal Return')
        axs[0].set_title('Rate of Return for Strategy and Benchmark')        
        axs[1].hist(self.each_rt,bins=100)
        axs[1].set_xlabel('Rate of Return')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Abnormal Return for Every Transaction')
        plt.show()
        
