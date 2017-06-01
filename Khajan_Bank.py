import pandas as pd
import xgboost
import numpy as np
from sklearn.preprocessing import LabelEncoder
from subprocess import check_output
import os
os.chdir('/home/hduser1/Downloads/Apna_Bank')
print(check_output(["ls"]).decode("utf8"))



print('Reading the files')
df = pd.read_csv('train.csv')#train csv

atms = pd.read_csv('ATM_Info.csv')#ATM info csv

df_test = pd.read_csv('test.csv')#test csv

sub_re = pd.read_csv('Replenishment.csv')#Submission - Replenishment.csv

sub_with = pd.read_csv('Withdrawal.csv')#Submission - Withdrawal.csv

################################################################################################
#Converting to datetime through pandas and striping out day,month,year,weekday and day of year #
#Idea behind it for feature engineering and one more task we can perform from week of day or   # 
# day,we can use for taking stratergy which is the second part of the question.                #
################################################################################################
def conv_df(df):
    df = pd.merge(df, atms, on='ATM_ID', how='left')
    df['Date'] = pd.to_datetime(df.Date, format='%d-%b-%y')
    df['day']= df.Date.dt.day
    df['month'] = df.Date.dt.month
    df['year'] = df.Date.dt.year
    df['weekday'] = df.Date.dt.dayofweek
    df['day_of_year'] = df.Date.dt.dayofyear
    return df

print('Adding features...')
df_train = conv_df(df)

################################################################################################
#Drop Balance from the df_train as the Balance has no role as of now(Balance is the amount left#
#after withdrawl of amount from respective atm).                                               #
################################################################################################

df_train.drop(['Balance'],1, inplace=True)

################################################################################################
#I have found the column name of submission file of withdrawl and train data has different col-#
#umn name .I have changed the name of columns as similar to train                              #
################################################################################################

sub_with.columns = ['ID', 'ATM_ID', 'Date', 'Withdrawal']

df_test = conv_df(sub_with)

###############################################################################################
#Here I have created a feature u_limit, here I have calculated group sum of each category     # 
# i.e. AtM_ID with quantile (0.99)                                                            #
###############################################################################################

ulimit = df_train[['ATM_ID', 'Withdrawal']].groupby('ATM_ID').quantile(0.99).reset_index()

###############################################################################################
#After it,I have added a new row called ulimit in df_train                                    #
###############################################################################################

df_train = df_train.merge(ulimit, 'left', 'ATM_ID', suffixes=('', '_ulimit'))

################################################################################################
#I have eliminate all those rows which withdraw greater than ulimit,it actually recheck if any #
# value greater than categorical sum(drop noise).Actually,I found it has (4807) values drop    #   
# which has value greater than 0.99 of the categorical sum.                                    #
################################################################################################

df_train.drop(df_train.loc[df_train.Withdrawal>df_train.Withdrawal_ulimit, :].index, inplace=True)

#################################################################################################
#I have drop withdrawl_ulimit as I have created to check the values                             #
#################################################################################################

df_train.drop(['Withdrawal_ulimit'],1,inplace=True)

#################################################################################################
#Indexing the data                                                                              #
#################################################################################################

df_train.index = range(df_train.shape[0])

#################################################################################################
#Calculated the categorical mean with respect to ATM_ID                                         #
#################################################################################################

df_mean = df_train.groupby('ATM_ID', as_index=False)['Withdrawal'].mean()

#################################################################################################
#Calculated the categorical standard deviation with respect to ATM_ID                           #
#################################################################################################

df_std = df_train[['ATM_ID', 'Withdrawal']].groupby('ATM_ID').std()['Withdrawal'].reset_index()

#################################################################################################
#Calculated the quantile value of catgorical data ATM_ID                                        #
#################################################################################################

df_uq = df_train[['ATM_ID', 'Withdrawal']].groupby('ATM_ID').quantile(0.75).reset_index()
df_lq = df_train[['ATM_ID', 'Withdrawal']].groupby('ATM_ID').quantile(0.25).reset_index()

#################################################################################################
#Merge df_test and df-train                                                                     #
#################################################################################################
df_all = pd.concat([df_train, df_test])


#################################################################################################
#Added features quantile,mean,standard deviation                                                #
#################################################################################################

df_all = df_all.merge(df_mean, 'left', 'ATM_ID', suffixes=('','_mean'))
df_all = df_all.merge(df_std, 'left', 'ATM_ID', suffixes=('','_std'))
df_all = df_all.merge(df_uq, 'left', 'ATM_ID', suffixes=('','_uq'))
df_all = df_all.merge(df_lq, 'left', 'ATM_ID', suffixes=('','_lq'))

#################################################################################################
#Dealing with catgorial value with LabelEncoder()                                               #
#################################################################################################

le = LabelEncoder()
for col in df_all.columns:
    if col in ['Facility', 'Type']:
        df_all[col] = le.fit_transform(df_all[col])

#################################################################################################
#I am  done with my feature Engineering as of now but I will add one Hot Encoder like feature   #
#################################################################################################

print('Feature engineering done.')

##################################################################################################
#Using model to train the data and predit the withdrawal amount as of now I have use xgboost but #
#sometimes logistic regression and RandomForest also, give great results.                        #
##################################################################################################

xgb = xgboost.XGBRegressor(n_estimators=600, learning_rate=0.02, 
                           max_depth=10, silent=True, min_child_weight=6, 
                           subsample=0.8, colsample_bytree = 0.7, reg_lambda = 0.5)

###################################################################################################
#DataFrame.isin()                                                                                 #
#The result will only be true at a location if all the labels match. If values is a Series,       #
#that’s the index. If values is a dictionary, the keys must be the column names, which must match.# 
#If values is a DataFrame, then both the index and column labels must match.                      #
###################################################################################################

df_train = df_all.loc[df_all.ID.isin(df.ID.unique()), :]
df_test = df_all.loc[df_all.ID.isin(df_test.ID.unique()), :]

#################################################################################################
#I have segregrate all the predictor values which i will to use in the model                    #
#################################################################################################

predictors = df_train.columns.drop(['Date','ATM_ID', 'Withdrawal','ID'])

#################################################################################################
#Train the model using                                                                          #
#################################################################################################
xgb.fit(df_train[predictors], df_train['Withdrawal'])

#################################################################################################
#Predict the withdrawl value from the model                                                     #
#################################################################################################
df_test['Withdrawal'] = xgb.predict(df_test[predictors])

#################################################################################################
#Those predicted value less than zero I have to make it zero because withdrawl amount can't be  #
#less than zero.                                                                                #
#################################################################################################
df_test.loc[df_test.Withdrawal<0, 'Withdrawal'] = 0


#################################################################################################
#Now, Withdrawl amount has predicted we have to go for second part of question which is         #
#optimizing the replenishment amount through strategy.                                          #                                  #################################################################################################

print('Withdrawal predicted. Now optimizing for replenishment.')

#################################################################################################
#Now, Withdrawl amount has predicted we have to go for second part of question which is         #
#optimizing the replenishment amount through strategy.                                          #
#Costs Involved                                                                                 #
#The cost of ATM replenishment involves 3 components                                            #
#Cost of refill: Refilling ATMs involves cost, for example, transportation/labor etc.           #
#Cost of cash: Banks earn interest on money through lending. Any money kept in ATMs is not      #
#available for lending and Banks looses interests on it. So, over stocking ATM leads to loss of #
#revenue for Bank.                                                                              #
#If X% is the interest rate annually and if an ATM has funds on daily basis as 1000, 500, 300,  #
#etc. then cost of cash for that ATM for test month is calculated as                            # 
#(Sum (funds on each day) / 31 ) * (15%/12)                                                     #
#Stock out cost = If an ATM runs out of cash there is a penalty!                                #
#Cost 	Value (INR)                                                                             #
#Cost of refill 	300                                                                     #
#Cost of Cash 	        15% Annually                                                            #
#Cost of stock out 	1000 per day                                                            #
#Replenishment process:                                                                         #
#Replenishment happens at the beginning of the day                                              #
#If there is cash in the ATM at the time of replenishment then it is removed and a new box with #
#cash equivalent to replenishment amount is inserted in ATM Stock calculated at the end of day  #
#You can use one of the replenishment strategy as follows:                                      #
#Strategy 	Action                                                                          #
#0 	      do not replenish                                                                  #
#1 	      7 days per week (everyday replenish)                                              #
#2 	      Replenish alternate days (day-1, day-3, day-5, etc.)                              #
#3 	      Replenish two specific days per week - Thursday-Monday                            #
#4 	      Replenish once weekly on every Thursday                                           #
#5 	      Replenish once weekly on every Monday                                             #
#6 	      Replenish once alternate week on Thursdays                                        #
#################################################################################################

def cost_function(df_, strategy, repl_amount):
    df = df_.copy()
    #getting replenishment according to strategy
    if strategy==0:
        df['is_refilled'] = 0
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==1:
        df['is_refilled'] = 1
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==2:
        df['range_'] = range(len(df))
        df.loc[df.range_%2==0 , 'is_refilled'] = 1
        df.is_refilled.fillna(0, inplace=True)
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==3:
        df.loc[(df.weekday==0) | (df.weekday==3), 'is_refilled'] = 1
        df.is_refilled.fillna(0, inplace=True)
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==4:
        df.loc[df.weekday==3, 'is_refilled'] = 1
        df.is_refilled.fillna(0, inplace=True)
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==5:
        df.loc[df.weekday==0, 'is_refilled'] = 1
        df.is_refilled.fillna(0, inplace=True)
        df['Replenishment'] = df['is_refilled']*repl_amount
    if strategy==6:
        df.loc[df.weekday==3,'is_thursday'] = 1
        df.is_thursday.fillna(0, inplace=True)
        df['cumsum_thursday']  = df.is_thursday.cumsum()%2
        df['is_refilled'] = df.is_thursday*df.cumsum_thursday
        df['Replenishment'] = df.is_refilled*repl_amount
        df.drop(['cumsum_thursday', 'is_thursday'],1, inplace=True)
    #Calculating balance accroding to replenishment
    df['Difference'] = (df['Replenishment']-df['Withdrawal'])
    #calculating the cumulative sum accoring to startergy
    df['cumsum_is_refilled'] = df.is_refilled.cumsum()
    #calculate the cumulativee sum according to cumsum_is_refilled
    df['Balance'] = df.groupby('cumsum_is_refilled')['Difference'].cumsum()
    
    df.drop(['cumsum_is_refilled', 'Difference'], 1, inplace=True)
    
    #calculating cost
    #cost of filling0
    cost_of_filling = df['is_refilled'].sum()*300
    #cost of cash
    costoofcash = (df.loc[df.Balance>0, 'Balance'].mean())*(0.15/12)
    cost_of_cash = 0 if np.isnan(costoofcash) else costoofcash
    #cost of stockout
    cos = df.loc[df.Balance<0, 'Balance'].count()
    # if balance is zero then zero otherwise penality of 1000
    cost_of_stockout = 0 if np.isnan(cos) else int(cos)*1000
    
    total_cost = cost_of_cash+cost_of_stockout+cost_of_filling
    
    return total_cost


#################################################################################################
#Now,Dynammic Programming algorithm to find the optimised stratergy to find the one which would #
#be the best by which bank occurred the minimum loss from the penality or cost of cash or cost  #
#of refill. 										                                                              	#
#The optimisation startergy I have taken here :							                                    #
#optimum_repl -intiated a variable optimum_repl(the optimum replacement amount)                 #
#best_stratergy-it includes the intial value of stratergy(stratergy,optimum_replacement and     #
#minimum amount i.e.the amount we get from the optimum_replacement)                             #
#Min_ will get the total cost with different startergy (1 to 7).As, cost_function returns       #
#total cost according to function with taking value as 0. while, curr_cost returns the total    #
#cost when value=1000 till 200000 on the day according to stratergy.If curr_cost less than      #
#min_cost(which means all the money get withdraw in the end of day) min_ = curr_cost and        #
#optimum_value = vaue.Actually I am doing experiemnt by taking this variables curr_cost and     #
#min_ so that I will end up with best startergy this is why I have taken sub-sample of          #
#range(0,200000,1000).                                                                          #
#################################################################################################

def Optimize(df_exp):
    optimum_repl = 0  
    best_strategy = [0,0,cost_function(df_exp, 1,0)]
    for strat in range(1,7):
        min_ = cost_function(df_exp, strat, 0)
        cost = []
        for value in range(0, 200000, 1000):
            curr_cost = cost_function(df_exp, strat, value)
            if curr_cost<min_:
                min_ = curr_cost
                optimum_repl= value
           if min_<best_strategy[2]:
                best_strategy = [strat, optimum_repl, min_]
      return best_strategy

#################################################################################################
#Preparing Submission file according to format   				                # #################################################################################################

sol_re = pd.DataFrame(columns=['ATM_ID', 'Replenishment frequency', 'Replenishment amount'])

#################################################################################################
#Testing the optimise startergy and cost_function  on the ATM_ID='SRNO00279'Graph shows if we   #
#take startergy-4.How Replenishment value change with respect to Total cost.                    #
#################################################################################################
df_t = df_test.loc[df_test.ATM_ID=='SRNO00279', :]
get_ipython().magic(u'time')
cost = []
for i in range(0,200000, 1000):
    cost.append((cost_function(df_t, 4, i)))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.plot(range(0,200000,1000), cost)
plt.xlabel('Replenishment value')
plt.ylabel('Total Cost')
plt.title('Plot for strategy 4')
plt.show()

#################################################################################################
#I have started optimising the replacement amount by calling above function Optimise            #
#################################################################################################

counter=0
for atm in df_test.ATM_ID.unique():
    df_temp = df_test.loc[df_test.ATM_ID==atm, :]
    counter+=1
    print('Optimizing for ATM_ID :', atm, 479-counter, 'optimizations left')
    curr_best = Optimize(df_temp)
    atm_sol = pd.Series({'ATM_ID':atm, 'Replenishment frequency':curr_best[0],
                           'Replenishment amount':curr_best[1]})
    sol_re = sol_re.append(atm_sol,ignore_index=True)

#################################################################################################
#Submitting file in the .csv format								                                              #
#################################################################################################

sol_re.to_csv('Solution_Replenishment.csv', index=False)#should be renamed to Withdrawal.csv for submission
sol_with = df_test[['ID', 'ATM_ID','Date' ,'Withdrawal']]
sol_with.columns = [['ID', 'ATM_ID', 'DATE', 'WITHDRAWAL']]
sol_with.to_csv('Solution_Withdrawal.csv', index=False)#should be renamed to Withdrawal.csv for submission
