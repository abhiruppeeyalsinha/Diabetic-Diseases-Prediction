import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf, keras
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
data_frame = pd.read_csv("diabetes.csv")
correlation = data_frame.corr()
# print(correlation)
x_data = data_frame.drop(columns="Outcome")
# scaler = StandardScaler()
# x_data = scaler.fit_transform(x_data)
# print(x_data)
y_data = data_frame["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.25,shuffle=True,
                                                 random_state=50)
# print(f"x train: {x_train.shape},y train{y_train.shape}")


# CNN Based

# model.add(Dense(128, activation="relu", input_dim=8))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy'])
# cnn_model = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test))
# cnn_model.model.save(r"E:\Projects & Tutorial\CNN Project\Hyperparameter Tuning a Neural Network\model file\Hyper_parameter_cnn_model50.h")

dt_clf = DecisionTreeClassifier(random_state=42)
# model = dt_clf.fit(x_train,y_train)
# test_val = [8,125,96,0,0,0,0.232,54]
# test_val = np.expand_dims(test_val,axis=0)
# #print(f'test_val.shape:- {test_val.shape}, test_val:{test_val}')
# result = (model.predict(x_test))
# print(f"Train accuracy:- {np.round(accuracy_score(y_train,dt_clf.predict(x_train)),2)}")
# print(f"Test accuracy:- {np.round(accuracy_score(y_test,dt_clf.predict(x_test)),2)}")

# confusion_Matrix  = confusion_matrix(y_test, result)
# print(f"confusion_Matrix:-{confusion_Matrix}")
# sns.heatmap(confusion_Matrix, annot=True,cmap="Blues")
# plt.xlabel("predicted", fontsize =12)
# plt.ylabel("true", fontsize =12)
# plt.show()


# if result == 1:
#     print("Diabetes Detected!!")
# else:
#     print("Healthy!!")
# file_name = "dt_clf!!.sav"
# pickle.dump(model,open(file_name,'wb'))
# print(f"it's working")

# cnn_model = load_model(model_path)
# result = cnn_model.predict(test_val)
# print(result)
# result = np.nanargmax(result)
# print(result)




# hyperparameter_space = {'bootstrap':[True],
#     'max_depth':[2,3,4,6,8,10,12,15,20],
#     'max_features':[2,3],
#     'min_samples_leaf':[1,2,4,6,8,10,20,30],
#     'min_samples_split':[1,2,3,4,5,6,7,8,10],
#     'n_estimators': [100,200,300,1000]}

# GridSearchCV Hyper tunning

# gs = GridSearchCV(estimator=dt_clf,param_grid=hyperparameter_space,
#                   scoring="accuracy",
#                   n_jobs=-1, cv=10, return_train_score=True)

# gs.fit(x_train,y_train)
# hyper_result = gs.best_params_.predict(x_test)
# print(f"result:- {hyper_result}")


# print(f"Train_accuracy:- {np.round(accuracy_score(y_train,gs.predict(x_train)),2)}")
# print(f"Test_accuracy:- {np.round(accuracy_score(y_test,gs.predict(x_test)),2)}")

#Random State Hyper tunning

hyperparameter_space = {"max_depth":[2,3,4,6,8,10,12,15,20],
                        "min_sample_leaf":[1,2,4,6,8,10,20,30],
                        "min_sample_split":[1,2,3,4,5,6,8,10]}


random_search = RandomizedSearchCV(dt_clf,param_distributions=hyperparameter_space,
                                   n_iter=100,scoring = "accuracy",random_state = 0,
                                   n_jobs =-1,cv=10, return_train_score=True)


random_search.fit(x_train,y_train)
result = random_search.best_estimator_.predict(x_test)
# print(f"result:- {result}")

save_model = pickle.dump(result,open(r"random_searchModel.pt","wb"))
print("Optimal hyperparameter combination:",random_search.best_estimator_)

print("Mean cross-validation training accu scr: ",random_search.best_score_)
























