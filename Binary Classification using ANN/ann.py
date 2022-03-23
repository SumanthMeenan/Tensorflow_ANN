# Import Libraries
from imports import * 

data = pd.read_csv("Churn_Modelling.csv")
print("Data Shape" , data.shape)

input = data.iloc[:,3:-1]
output = data.iloc[:, -1].values 

gender_dict = {"Female":1, "Male":0}            
input['Gender'] = input['Gender'].apply(lambda x: gender_dict[x])    

print(input['Geography'].value_counts())

col_tranformer = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])], remainder = "passthrough")
features  = np.array(col_tranformer.fit_transform(input))

print("features shape: ", features.shape)
print(features) 

#Standardization and normalization has to been done after data splitting
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = 0.3)

std_scalar = StandardScaler() # Z-Score normalization [(x-mean)/sd]
x_train = std_scalar.fit_transform(x_train)
x_test = std_scalar.fit_transform(x_test)

print("x_train",x_train[0])

#Neural Network Architecture

#Step1: Initialize a Neural Network
sequential = tf.keras.models.Sequential()

#Step2: Create Hidden layers 
sequential.add(tf.keras.layers.Dense(units = 14, activation = 'relu'))
sequential.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
sequential.add(tf.keras.layers.Dropout(0.2))
#Step3: Create Output layer  #Binary Classification - 1 neuron
sequential.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

#Step4: Compile the NN [loss func -> cal loss b/w actual & predicted, optimizer -> cal. gradients & update weights]
sequential.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# print(sequential.summary)
#Step5: Fit the ANN on data
training_process = sequential.fit(x_train, y_train, batch_size = 16, epochs = 50)

sequential.save("ann.h5") #h5 is for NN, saves our NN as a serialized object. 

example_datapoint = [0, 1, 0, 837, 0, 53, 5, 55231, 3, 1, 1, 103234]
transformed_datapoint = std_scalar.transform([example_datapoint])
print("prediction: ", sequential.predict(transformed_datapoint))

def accuracy_graph():
    plt.plot(training_process.history['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(" ANN Model Accuracy")
    plt.legend(["Training"])
    plt.show()

accuracy_graph()

# total_data = batch_size * no of batches 
# 7000 = 16 * no of batches 
# # no of batches  = 7000/16 = 438

#why training acc becomes stagnant at 10th epoch out of 50 epochs?
# weights r not much updated, w_new = w_old - alpha * gradient, target alpha, lrscheduler 

#weight initializer https://keras.io/api/layers/initializers/ 
#layer vs batch normalization 
#saving a model vs saving weights
#predictions on UI
#overfitting and underfitting in NN

#backpropogation
#gradient descent
#optimizers
#loss functions, gradients
#metrics
#callbacks



