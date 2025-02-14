import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


#                            loading the data
cars = pd.read_csv("/Users/Hirad/Downloads/New folder/car data.csv")
hd = cars.head(10)
print(cars.shape)
print(cars.info())
print('checking the number of nun(s) in the dataset :',cars.isnull().sum())



#               replacing categorical data with numerical data
cars.replace({'Fuel_Type':{'Petrol' : 0 , 'Diesel' :1 , 'CNG' : 2}},inplace=True)
cars.replace({'Seller_Type':{'Dealer' : 0 , 'Individual' :1 ,}},inplace=True)
cars.replace({'Transmission':{'Manual' : 0 , 'Automatic' :1}},inplace=True)


#                     spliting dataset from target

x = cars.drop(['Car_Name','Selling_Price'],axis = 1)
y = cars['Selling_Price']


#                     spliting train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3
                            ,random_state=40)


#                      Loading regression model
lin = LinearRegression()


#                  fitting the data to Linearregression
lin.fit(x_train,y_train)


# predicting the training data
y_predict = lin.predict(x_test)


#                             R squared model
error_score = metrics.r2_score(y_test,y_predict)
print('R squared error :',error_score)


#          visualizing the actual price and the predicted price 
plt.scatter(y_test,y_predict)
plt.xlabel('original price')
plt.ylabel('predicted price')
plt.title('predicted price VS original price ')
plt.show()










