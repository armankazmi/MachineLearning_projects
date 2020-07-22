#Importing the modules
import pandas as pd
import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer


#Mnist Dataset with 3000 samples
df = pd.read_csv("17054.csv")
X = np.asfarray(df.drop('label',axis = 1))
y = np.asfarray(df.label)


#Uncomment to Visualise the training images

#from matplotlib import pyplot as plt 
#pixels = np.array(df.drop('label',axis = 1).loc[1], dtype='uint8')
#pixels = pixels.reshape((28, 28))
#plt.imshow(pixels, cmap="Greys")


#Normalizing X
factor = 0.99/255
X_val = (X[0:])*factor + 0.01

#Preparing the Classification dataset
data = ClassificationDataSet(784,1, nb_classes=10)  #784 features, 1 output and 10 classes

#Appending the data
for i in range(len(y)):
    data.addSample(X_val[i],y[i])


#Splitting the dataset with 25% testing dataset
testdata , traindata = data.splitWithProportion(0.25)

#Preparing the Classification dataset
test_data = ClassificationDataSet(784, 1, nb_classes=10)
training_data = ClassificationDataSet(784, 1, nb_classes=10)

for n in range(0, testdata.getLength()):
    test_data.addSample( testdata.getSample(n)[0], testdata.getSample(n)[1])
for n in range(0, traindata.getLength()):
    training_data.addSample( traindata.getSample(n)[0], traindata.getSample(n)[1])

#Converting to One-hot encoding
test_data._convertToOneOfMany()
training_data._convertToOneOfMany()


#Building the network
net = buildNetwork(training_data.indim, 200, training_data.outdim,outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, dataset=training_data, momentum=0.1,learningrate=0.01,verbose=True,weightdecay=0.01)

#For printing the network
print(net)

#Training the network on 20 epochs
trainee,validation = trainer.trainUntilConvergence(dataset=training_data,maxEpochs=20)


#Accuracy using scikit-learn
from sklearn.metrics import accuracy_score
print ("Accuracy on test set: %7.4f" % accuracy_score(trainer.testOnClassData(dataset=test_data), test_data['class'], normalize=True))

