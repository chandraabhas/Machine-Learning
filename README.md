# Machine-Learning
PROBLEM STATEMENT
The collapse of reinforced concrete structures has caused a heavy damage to property and lives of the people in the recent decades. Investigations lead to the fact that the quality of concrete was below standards. It was therefore needed to achieve the required compressive strength and ductility of concrete in the design to attain the reliability of structures. Also, if the compressive strength of concrete greatly exceeds the specified strength, it seriously affects the ductile ratio of the structure. This also creates dead load in the structure.
Hence, we can state 3 things: 1.It is needed to achieve the required compressive strength and ductility of concrete in the design. 2. If the compressive strength of concrete greatly exceeds the specified strength, it will seriously affect the ductile ratio of the structure 3. If the deviation of compressive strength of concrete is over the limit, it causes imbalance to the ductile ratio of structure, and adversely influence the seismic capability of the structure.
Due to variability in the strength properties of concrete, the compliance with the desired specifications is not met. Variations in strength may occur due to improper mixing of materials, varied w/c ratio for different batches, change in batching plant operator, proportions of raw materials etc. The compressive strength of the concrete obtained after 28 days of ageing, goes drastically high. That excessive value creates a dead load in the structure. And ultimately takes the cost of production high, incurring loss to the company in the long run.
Here, we address this issue. This will be done using various visualisations and analysis. 
The aim is to predict compressive strength of the concrete based on a particular combination of 
values of fields/parameters. We will create models using different algorithms and choose the best, 
based on accuracy. There will be thorough inter models as well as 
intra( i.e hyperparamter tuning) model comparison. 
We will be building three classification models â Logistic regression model, 
Kernel Support Vector Machine (SVM) (Gaussian) model, and Random Forest Classification model and 
test the accuracy of prediction.Finally we will draw a comparison amongst all the models in the result 
section of the report.

# install.packages('caTools')
# install.packages('caret')
# install.packages('e1071')
# install.packages('randomForest')
library(readxl) 
library(caTools) 
library(caret)

library(e1071) library(randomForest)

DATA
The dataset is sourced from Simplex Infrastructures Ltd. which is a diversified company established in 1924 and executing projects in several sectors like Transport, Energy & Power, Mining, Buildings, Marine, Real Estate etc. It is named with the heading - âCONCRETE CUBE REGISTER FOR TOWER 5 & 6 & POD/MRSS CUBES STRENGTH DETAILSâ. Following is the link for the data named Simplex_Cube_Register.
https://drive.google.com/open?id=1_gCfyRD3TgWspTkn2CYopoD1CG1fxhL0 (https://drive.google.com/open? id=1_gCfyRD3TgWspTkn2CYopoD1CG1fxhL0)
This is a private data obtained from Mr.Rishi Kathed, enrolled in Project Management course at RMIT University. His undergraduate research thesis, that drove us to delve deeper into this data. is at the following link:
https://drive.google.com/open?id=1hPvsxyLPxNNJrcSFymAPyqQvFIkmz0MN (https://drive.google.com/open? id=1hPvsxyLPxNNJrcSFymAPyqQvFIkmz0MN)
Details about the data: 1. It contains a total of 2986 rows and 16 columns. 2. This data set contains details about the strength of different concrete grades, taken from different location and aged 7 days and 28 days respectively. The details of each concrete grade are spanned amongst 16 columns â grade, location, structure, source, weight, quantity, average strength, density, compressive strength, etc.

DATA CLEANING
Certain columns were dropped from the table which 
were not of significance for model building. 
Since the data has been imported as an excel file, 
it contains a lot of âNAâ values which needs to be cleaned. 
In excel file, some columns have rowspans because of common values among some rows. 
R reads the rowspans in a different way, suppose if there is a rowspan of n 
rows(or same value for n rows), it takes the rowspan value and assign it to 1st row and 2:n will 
get values as NA. As we can see the similiar pattern throughout the sheets, which can be fixed by 
small chunk of while code chunk, which takes the row value at first row and fill remaining rows 
with that value until we get row with already having new value. 
The target feature contained some impossible values such as Compressive strength in % 
being equal to 0 and values greater than 200 which have been removed. Concrete source and concrete grade columns contained similar labels which have been made uniform. Trimming of Qty column was done and made uniform. Encoding for concrete source and concrete grade column was performed. Age was converted to categorical feature. Missing data from certain columns were removed. 
Numerical data columns were converted double. Encoding of target feature as factor was done by splitting it into two range of values - 1) 0-100 and, 2) 100-200 and ten labelling them.

#Columns dropped
cube$Sr.No <- NULL
cube$Remark <- NULL
cube$`CUBE ID` <- NULL
cube$`Date of Testing` <- NULL
cube$`Date of Casting` <- NULL
cube$X__1 <- NULL
cube <- cube[,-(2:3),drop=FALSE]
cube <- cube[,-(8:9),drop=FALSE]
#subdivision of rowspan and replicating the rowspan value into the newly created 'NA' columnsModification <- function(colname){
i <- 1
temp <- ""
while (i < length(cube[[colname]])) {
if(!is.na(cube[[colname]][i])){ temp <- cube[[colname]][i]
} else{
cube[[colname]][i] <- temp
}
i <- i+1 } return(cube)
}
colNames <- c('Concrete Grade', 'Location', 'Structure','Age','Qty.','Concrete Source','Avg. St r
ength in N/mm2','Comp. Strength in %')
for(i in 1:length(colNames))
  cube <- columnsModification(colNames[i])
#Removing impossible values
cube<-cube[!(cube$`Comp. Strength in %`==0 | cube$`Comp. Strength in %`>=200),]
cube<-cube[!(is.na(cube$Age)>0),] #One corrupt row with NA values removed
summary(cube)

#Making the column uniform
cube$`Concrete Source` <- gsub('ACC PLANT', 'ACC', cube$`Concrete Source`)
cube$`Concrete Source` <- gsub('ACC PLANT GGBS', 'ACC', cube$`Concrete Source`)
cube$`Concrete Source` <- gsub('ACC GGBS', 'ACC', cube$`Concrete Source`)
cube$`Concrete Grade` <- gsub('M-40', 'M40', cube$`Concrete Grade`)
cube$`Concrete Grade` <- gsub('M-50', 'M50', cube$`Concrete Grade`)
cube$`Concrete Grade` <- gsub('M-70', 'M70', cube$`Concrete Grade`)
cube$`Concrete Grade` <- gsub('M-10', 'M10', cube$`Concrete Grade`)
cube$`Concrete Grade` <- gsub('M-15', 'M15', cube$`Concrete Grade`)
#Trimming of Qty column
cube$Qty. <- gsub('m3' , '' , cube$Qty.)
cube$Qty. <- gsub('M3' , '' , cube$Qty.)
trimws(cube$Qty., which = c("both", "left", "right"))

#Encoding for concrete source and concrete grade features
cube$`Concrete Source` = factor(cube$`Concrete Source`,
                                levels = c('ACC', 'Lafarge' ),
                                labels = c(0, 1))
cube$`Concrete Grade` = factor(cube$`Concrete Grade`,
                                levels = c('M10', 'M15' , 'M40' , 'M50' , 'M70'),
                                labels = c(0, 1 , 2, 3 , 4))
#Conversion of numerical feature Age to categorical
cube$Age = factor(cube$Age,levels = unique(cube$Age),labels = seq(0,length(unique(cube$Age))-1)
)

#Handling of missing data
cube$Qty. = ifelse(is.na(cube$Qty.),
ave(cube$Qty., FUN = function(x) mean(x, na.rm = TRUE)), cube$Qty.)
cube$`Weight in Kg.` = ifelse(is.na(cube$`Weight in Kg.`),
ave(cube$`Weight in Kg.`, FUN = function(x) mean(x, na.rm = TRUE)),
                   cube$`Weight in Kg.`)
cube$`Density in MT/m3` = ifelse(is.na(cube$`Density in MT/m3`),
ave(cube$`Density in MT/m3`, FUN = function(x) mean(x, na.rm = TR
UE)),
                              cube$`Density in MT/m3`)
cube$`Load in KN` = ifelse(is.na(cube$`Load in KN`),
),
ave(cube$`Load in KN`, FUN = function(x) mean(x, na.rm = TRUE) cube$`Load in KN`)
cube$`Comp. Strength in %` = ifelse(is.na(cube$`Comp. Strength in %`),
ave(cube$`Comp. Strength in %`, FUN = function(x) mean(x, na.rm = TR
UE)),
                           cube$`Comp. Strength in %`)
#
# #Conversion of integer data into double
as.double(cube$Qty.)

 as.double(cube$`Weight in Kg.`)
 
  as.double(cube$`Density in MT/m3`)
  
   as.double(cube$`Load in KN`)
   
    as.double(cube$`Comp. Strength in %`)
    
# Encoding the target feature as factor
cube$`Comp. Strength in %` <- cut(cube$`Comp. Strength in %`, breaks = c(0,100,200),
                                  labels = c("A" , "B"))
cube$`Comp. Strength in %` = factor(cube$`Comp. Strength in %`,
                         levels = c('A', 'B'),
                         labels = c(0,1))
                         
 DATA MODELLING
After retrieving and exploration (done in assignment 1) of the data, the final step is the data modelling, where we will be using three classification models - Kernel SVM model, Logistic regression model and, Random Forest Classification model. But before that, two basic steps would be performed which are splitting the data and feature scaling.

SPLITTING THE DATASET

# Splitting the dataset into the Training set and Test set # install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(cube$`Comp. Strength in %`, SplitRatio = 0.80)
training_set = subset(cube, split == TRUE)
test_set = subset(cube, split == FALSE)

FEATURE SCALING
There can be instances found in data frame where values for one feature could range between 1-100 and values for other feature could range from 1-10000000. In scenarios like these, owing to the mere greater numeric range, the impact on response variables by the feature having greater numeric range could be more than the one having less numeric range, and this could, in turn, impact prediction accuracy. The objective is to improve predictive accuracy and not allow a particular feature impact the prediction due to large numeric value range. Thus, we may need to normalize or scale values under different features such that they fall under common range. This normalization is called feature scaling and it was performed on the training set and the test set of the data.

training_set[, 5:7] = scale(training_set[, 5:7])
test_set[, 5:7] = scale(test_set[, 5:7])

KERNEL SVM MODEL
1. In order to fit the training set to Kernel SVM, we create the SVM classifier.
2. We then go onto predicting the test set using the SVM classifier.
3. We create the confusion matrix and obtain the following output: y_pred 0 1 0 227 56 1 11 169
Misclassification rate - 14.47% Accuracy - 85.52%

# Fitting kernel svm to the Training set library(e1071)
classifier = svm(formula = `Comp. Strength in %` ~ `Load in KN` + `Density in MT/m3` +
                   `Weight in Kg.` + Age ,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-8])
# Making the Confusion Matrix
cm = table(test_set$`Comp. Strength in %`, y_pred)
cm

## y_pred 
##     0  1 
## 0 227 56 
## 1 11 169

K-FOLD CROSS VALIDATION
k-fold Cross validation is performed - the samples are randomly partitioned into k sets (called fold s) of roughly equal size. A model is fit using all the samples except the first subset. Then, the predi ction error of the fitted model is calculated using the first held-out samples. The same operation is repeated for each fold and the modelâs performance is calculated by averaging the errors across th e different test sets. K is usually fixed at 5 or 10. Cross-validation provides an estimate of the test error for each model. Accuracy obtained is 86%.

folds = createFolds(training_set$`Comp. Strength in %`, k = 10) cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = `Comp. Strength in %` ~ `Load in KN` + `Density in MT/m3` +
                     `Weight in Kg.` + Age ,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
y_pred = predict(classifier, newdata = test_fold[-8])
cm = table(test_fold$`Comp. Strength in %`, y_pred)
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1]) return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy

## [1] 0.86

GRID SEARCH FOR KERNEL SVM MODEL

classifier = train(form = `Comp. Strength in %` ~ `Load in KN` + `Density in MT/m3` +
                     `Weight in Kg.` + Age,
                   data = training_set, method = 'svmRadial')
                   
 classifier
 
  classifier$bestTune
  
  ##       sigma C
## 3 0.5973066 1

LOGISTIC REGRESSION MODEL
1. Classifier was created for logistic regression and we fit the training set to it. 2. Prediction was made using this classifier.
3. Confusion matrix obtained: y_pred 0 1 0 225 58 1 13 167
Misclassification rate - 15.33% Accuracy - 84.66%

# Fitting Logistic Regression to the Training set
classifier = glm(formula = `Comp. Strength in %` ~ `Load in KN` + `Density in MT/m3` +
                   `Weight in Kg.` + Age ,
                 family = binomial,
                 data = training_set)
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-8])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
# Making the Confusion Matrix
cm = table(test_set$`Comp. Strength in %`, y_pred)
cm

## y_pred 
##     0  1 
## 0 225 58 
## 1 13 167

RANDOM FOREST CLASSIFICATION MODEL
1. In order to fit the training set to Random forest classification, we create the random forest classifier. 2. We then go onto predicting the test set using the Random forest classifier.
3. We create the confusion matrix and obtain the following output: y_pred 0 1 0 267 16 1 7 173
Misclassification rate - 4.96% Accuracy - 95.03%

library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-8],
                          y = training_set$`Comp. Strength in %`,
                          ntree = 500)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-8])
# Making the Confusion Matrix
cm = table(test_set$`Comp. Strength in %`, y_pred)
cm

## y_pred 
##     0  1
## 0 266 17 
## 1 9 171

RESULTS
1. Logistic regression model performed well with an accuracy of 84.66%.
2. Kernel SVM (Support Vector Machine) (Gaussian) model performed well too with an accuracy of 85.52%.
3. Random forest classification model gave the highest accuracy amongst all three models with an accuracy of
95.03%.
4. Grid search could not create any marginal difference in the accuracy output and hence no hyperparameter tuning
was done.
5. k-Fold cross validation gave an accuracy of 86%

CONCLUSION

All the three classification models performed well, with a good high accuracy. Amongst all the three classification models, Random forest classification model performed the best. And hence can be used for prediction of the compressive strength. We can provide a portal to the company that can help to predict and provide them with an intuition as to how much strength can be achieved by given combination of features, before actually building it. This can help them save on cost of production, time and labour involved. Fine tuning of the parameters involved in building the concrete, can help to restrict the compressive strength within limits, ultimately avoiding the creation of dead load in the structure. This will help the buildings to function and perform better in the long run, and avoiding life hazards due to weak concrete structure.




