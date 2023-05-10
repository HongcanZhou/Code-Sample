#—————————————————————————————————————— C4.5 ——————————————————————————————————————————
##清空工作环境
rm(list=ls())
##加载程序包
library(xlsx)
library(RWeka)
library(partykit)
##设置工作路径
setwd("F:/周泓璨的文件夹/3.6 挑战杯科研/代码")

##导入训练集以及测试集
dtraining_set= read.xlsx(file= "dtraining_set.xlsx", 1, header = T)
test_set= read.xlsx(file= "test_set.xlsx", 1, header = T)
#转因子：fetal_health
dtraining_set$fetal_health= as.factor(dtraining_set$fetal_health)  
test_set$fetal_health= as.factor(test_set$fetal_health)
dtraining_set$histogram_tendency= as.factor(dtraining_set$histogram_tendency)
test_set$histogram_tendency= as.factor(test_set$histogram_tendency)

#了解数据集
str(dtraining_set)

###构造C4.5决策树算法 1_no pruning
baby_tree= J48(fetal_health ~ ., data= dtraining_set)
#输出规则
baby_tree
#绘制决策树图形
plot(baby_tree, main="fetal health classification1_all num_no pruning")   #太乱
#计算预测训练集的准确率
table(predict(baby_tree), dtraining_set$fetal_health)
train_correct = sum(as.numeric(predict(baby_tree))==as.numeric(dtraining_set$fetal_health))/nrow(dtraining_set)
train_correct
#预测测试集并计算准确率
test_pre = predict(baby_tree, newdata = test_set)
table(test_pre, test_set$fetal_health)
test_correct = sum(as.numeric(test_pre)==as.numeric(test_set$fetal_health))/nrow(test_set)
test_correct
##输出predict完整结果
write.xlsx(test_pre,file= "test_pre.xlsx")
##输出混淆矩阵
table(test_set$fetal_health, test_pre, dnn=c("真实值","预测值") )  

###构造C4.5决策树算法 3_with pruning W=3
baby_tree2= J48(fetal_health ~ ., data= dtraining_set, control=Weka_control(M=50))
#输出规则
baby_tree2
#绘制决策树图形
plot(baby_tree2, main="fetal health classification1_with pruning W=3")   
#预测测试集并计算准确率
test_pre2 = predict(baby_tree2, newdata = test_set, type= "probability")
##输出predict完整结果
write.xlsx(test_pre2,file= "test_pre.xlsx")
test_pre2 = predict(baby_tree2, newdata = test_set, type= "class")
table(test_pre2, test_set$fetal_health)
test_correct2 = sum(as.numeric(test_pre2)==as.numeric(test_set$fetal_health))/nrow(test_set)
test_correct2
##输出混淆矩阵
table(test_set$fetal_health, test_pre2, dnn=c("真实值","预测值") )  


#—————————————————————————————— Random Forest ——————————————————————————————————————————
library(randomForest)
library(pROC)
library(ROCR)
train$fetal_health = as.factor(train$fetal_health)
test$fetal_health = as.factor(test$fetal_health)
train$histogram_tendency = as.factor(train$histogram_tendency)
test$histogram_tendency = as.factor(test$histogram_tendency)

p = ncol(train)-1
n = nrow(train)
OOB = c()
### 构建RF1
set.seed(1)
for (i in 1:(p-1)) {
  model = randomForest(fetal_health~., data = train, mtry = i, importance = TRUE,
                       xtest = test[,-22], ytest = test[,22])#测试集
  oob = mean(model$err.rate[,1])
  OOB = c(oob, OOB)
}
# 输出相关结果
plot(x = 1:(p-1), y = OOB, type = "b", col = "steelblue", xlab = "m", ylab = "error", lwd = 2)
grid()
OOB
M = which(OOB==min(OOB))
M

### 构建RF2
et.seed(23)
model1 = randomForest(fetal_health~., data = train, mtry = M, importance = TRUE,
                      ntree = 1000, xtest = test[,-22], ytest = test[,22])
# 输出结果
plot(model1, lwd = 2, main = "")
grid()

### 构建RF3
set.seed(45)
model2 = randomForest(fetal_health~., data = train, mtry = M, importance = TRUE,
                      ntree = 400, xtest = test[,-22], ytest = test[,22])
# 输出结果
importance(model2)
varImpPlot(model2,main = "")

#默认阈值预测结果
print(model2)
plot(test$fetal_health, model2$test$predicted, xlab = "真实值", ylab = "预测值")
table(model2$test$predicted, test$fetal_health)

#ROC&AUC
multiclass.roc(test$fetal_health, model2$test$votes)

zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==1)] = 1
roc(zero, model2$test$votes[,1], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.766

zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==2)] = 2
roc(zero, model2$test$votes[,2], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.158

zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==3)] = 3
roc(zero, model2$test$votes[,3], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.236

#调整阈值预测结果
predict_test = rep(1,length(test$fetal_health))
predict_test[which(model2$test$votes[,2]>=0.158)] = 2
predict_test[which(model2$test$votes[,3]>=0.236)] = 3

table(predict_test, test$fetal_health)
(505+84+53)/length(test$fetal_health)   #准确率
1-505/sum(test$fetal_health==1)         #1的敏感度
1-84/sum(test$fetal_health==2)          #2的敏感度
1-53/sum(test$fetal_health==3)          #3的敏感度
plot(test$fetal_health, as.factor(predict_test), xlab = "真实值", ylab = "预测值")

#—————————————————————————————— Random Forest ——————————————————————————————————————————
##SMOTE
library(DMwR)
full=read.csv('E://fetal_health.csv',header=T)
full$histogram_tendency=factor(full$histogram_tendency)
full$fetal_health=factor(full$fetal_health)

sum=1075+204+121
1075/sum
204/sum
121/sum

set.seed(5)
train_index=sample(nrow(full),1400)
train_unbalanced=full[train_index,]
test=full[-train_index,]

train_unbalanced1=train_unbalanced[which(train_unbalanced$fetal_health!=3),]
train_unbalanced1$fetal_health=factor(train_unbalanced1$fetal_health)
train_balanced1=SMOTE(fetal_health~.,
                      data=train_unbalanced1,
                      k=5,perc.over = 100)
twos=train_balanced1[which(train_balanced1$fetal_health==2),]


train_unbalanced2=train_unbalanced[which(train_unbalanced$fetal_health!=2),]
train_unbalanced2$fetal_health=factor(train_unbalanced2$fetal_health)
train_balanced2=SMOTE(fetal_health~.,
                      data=train_unbalanced2,
                      k=5,perc.over = 300)
threes=train_balanced2[which(train_balanced2$fetal_health==3),]


ones=train_unbalanced[which(train_unbalanced$fetal_health==1),]

train_balanced=rbind(ones,twos,threes)
summary(train_balanced)

write.csv(train_balanced,file='E://train_balanced.csv')
write.csv(test,file='E://test.csv')

##boosting model
library(rpart)
library(adabag)
##d=1
cv.results_d1=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=1),v=10)
  cv.results_d1=rbind(cv.results_d1,c(B,1,boost.model.cvi$error))
}

##d=2
cv.results_d2=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=2),v=10)
  cv.results_d2=rbind(cv.results_d2,c(B,2,boost.model.cvi$error))
}

##d=3
cv.results_d3=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=3),v=10)
  cv.results_d3=rbind(cv.results_d3,c(B,3,boost.model.cvi$error))
}

##d=4
cv.results_d4=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=4),v=10)
  cv.results_d4=rbind(cv.results_d4,c(B,4,boost.model.cvi$error))
}

##d=5
cv.results_d5=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=5),v=10)
  cv.results_d5=rbind(cv.results_d5,c(B,5,boost.model.cvi$error))
}

##d=6
cv.results_d6=NULL
for(B in 1:20)
{
  boost.model.cvi=boosting.cv(fetal_health~.,data=train_balanced,mfinal=B,
                              control=rpart.control(maxdepth=6),v=10)
  cv.results_d6=rbind(cv.results_d6,c(B,6,boost.model.cvi$error))
}


plot(cv.results_d2[,3],col='red',type='l',ylim=c(0,0.5),
     xlab='number of trees (B)',ylab='CV error rate',lwd=2)
points(cv.results_d1[,3],col='blue',type='l',lwd=2)
points(cv.results_d3[,3],col='green',type='l',lwd=2)
points(cv.results_d4[,3],col='purple',type='l',lwd=2)
points(cv.results_d5[,3],col='orange',type='l',lwd=2)
points(cv.results_d6[,3],col='grey',type='l',lwd=2)
legend('topright',col=c('white','blue','red','green','purple','orange','grey'),
       legend = c('depth of trees','d=1','d=2','d=3','d=4','d=5','d=6'),lty=1,lwd=2)

##选择B=10,d=5
boost.model=boosting(fetal_health~.,data=train_balanced,mfinal=10,boos=F,
                     control=rpart.control(maxdepth=5))

test.pred=predict.boosting(boost.model,newdata=test)
table(test.pred$class,test$fetal_health)

acc=(559+72+51)/nrow(test)
sen1=559/(559+18+3)
sen2=72/(72+16+3)
sen3=51/(51+2+2)

plot(test$fetal_health, as.factor(test.pred$class), 
     xlab = "真实值", ylab = "预测值")


##变量重要性
boost.model$importance
importanceplot(boost.model)

name=c('abnormal_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'histogram_mean','accelerations','prolongued_decelerations',
       'histogram_median','mean_value_of_short_term_variability',
       'uterine_contractions','histogram_mode','histogram_max',
       'baseline.value','histogram_number_of_peaks','light_decelerations',
       'mean_value_of_long_term_variability','histogram_variance',
       'fetal_movement','histogram_tendency','histogram_width',
       'histogram_min','histogram_number_of_zeroes','severe_decelerations')
importance=c(25.1663394,17.1081591,10.9549682,9.0489002,7.8063663,
             6.338575,5.6253547,5.2094049,2.823027,2.1353868,1.9833104,
             1.6396358,1.1669853,1.0674996,0.7964273,0.5964822,0.2903968,
             0.2427811,0,0,0)

par(oma=c(0,18,0,0))
plot(importance,21:1,yaxt='n',ylab='',xlab='MeanDecreaseGini')
axis(side=2,at=21:1,labels=name,las=1)
for(i in 1:21)
{
  abline(h=i,col='grey',lty=3)
}

###多分类roc（只输出AUC）
library(pROC)
response=test$fetal_health
response=as.numeric(response)
response[which(response==1)]='n'
response[which(response==2)]='s'
response[which(response==3)]='p'

pred.prob=test.pred$prob
predictor=data.frame("n"=pred.prob[,1],"s"=pred.prob[,2],
                     "p"=pred.prob[,3])
predictor=as.matrix(predictor)

roc=multiclass.roc(response=response,predictor=predictor)


###multi-roc（三张ROC图像） 
par(mfrow=c(1,3))
zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==1)] = 1
roc(zero, test.pred$prob[,1], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.599,auc=0.982

zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==2)] = 2
roc(zero, test.pred$prob[,2], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.399,auc=0.966

zero = rep(0, length(test$fetal_health))
zero[which(test$fetal_health==3)] = 3
roc(zero, test.pred$prob[,3], plot=TRUE, print.thres=TRUE, print.auc=TRUE)#标签&概率0.288,auc=0.986

avg.auc=(0.982+0.966+0.986)/3

###调整阈值&分阶段预测
new.pred=rep(1,nrow(test))
new.pred[which(test.pred$prob[,3]>=0.288)]=3
new.pred[which(test.pred$prob[,2]>=0.399 &
                 test.pred$prob[,3]<0.288)]=2
which(test.pred$prob[,3]>=0.288 & test.pred$prob[,2]>=0.399)
table(new.pred,test$fetal_health)

acc.new=(531+81+52)/nrow(test)
sen1.new=531/(531+39+10)
sen2.new=75/(8+75+8)
sen3.new=52/(52+2+1)

plot(test$fetal_health, as.factor(new.pred), 
     xlab = "真实值", ylab = "预测值")








