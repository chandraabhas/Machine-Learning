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
