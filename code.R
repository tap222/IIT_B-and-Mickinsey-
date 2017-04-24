rm(list=ls())
library(caret)
library(xgboost)
library(lubridate)
tr<-read.csv("train.csv")
ATM<-read.csv("ATM_info.csv")
tr$year<-year(tr$Date)
tr$year<-year(tr$Date)
tr$month<-month(tr$Date)
tr$day<-day(tr$Date)
tr$Date<-NULL
View(tr)
tr<-merge(ATM,tr,by="ATM_ID")
View(tr)
tr$ID<-NULL
tr$WITHDRAWAL<-tr$Withdrawal
View(tr)
tr$Withdrawal<-NULL
ind<-createDataPartition(tr$WITHDRAWAL, p=0.8, list=FALSE)
valid<-tr[-ind,]
train<-tr[ind,]
train$Balance<-NULL
valid$Balance<-NULL
View(train)
View(valid)
tmatrix<-data.matrix(train[,1:6])
vmatrix<-data.matrix(valid[,1:6])
class(tmatrix)
head(tmatrix)
dtrain<-xgb.DMatrix(tmatrix,label=train$WITHDRAWAL)
dtrain
dvalid<-xgb.DMatrix(vmatrix, label=valid$WITHDRAWAL)
dvalid
watchlist <- list(train=dtrain, test=dvalid)
watchlist
model <- xgb.train(data=dtrain,                    # Must
                   max.depth=10,                    # Optional, default=6
                   eta=0.5,                        # Optional, default=0.3
                   nthread = 3,                    # Optional, default=1
                   nround=200,                      # Must
                   #subsamle= 0.75,                  # Optional, default=1
                   colsample_bytree = 0.8,         # Optional, default=1
                   #watchlist=watchlist,            # Optional, default=none
                   objective = "reg:linear")  # Must
pred <- predict(model, dvalid)
te<-read.csv("withdrawl.csv")
te$year<-year(te$DATE)
te$month<-month(te$DATE)
te$day<-day(te$DATE)
te$DATE<-NULL
View(te)
te$WITHDRAWAL1<-te$WITHDRAWAL
View(te)
te$WITHDRAWAL<-NULL
te$WITHDRAWAL<-te$WITHDRAWAL1
View(te)
te$WITHDRAWAL1<-NULL
test_matrix<-data.matrix(te)
test_pred <- predict(model, test_matrix)
test_pred
te$Withdrawl<-test_pred
write.csv(te, file="xgboost1.csv", quote= FALSE, row.names = FALSE)

tr<-read.csv("withdrawl.csv")
View(tr)
byday <- aggregate(cbind(WITHDRAWAL)~ATM_ID,data=tr,FUN=sum)
View(byday)
write.csv(byday,file="Replenishment2.csv")
tail(train,-1) - head(train, -1)
View(train)