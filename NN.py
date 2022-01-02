import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

###X=[[1.0, 2.0, 3.0, 2.5],
   ###[2.0, 5.0, -1.0, -2.3],
   ###[-1.15, 2.7, 3.3, -.05]]

def create_data(points,classes):
    X = np.zeros((points*classes,2)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for class_number in range(classes):
        ix = range(points*class_number,points*(class_number+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X,y
    
X,y=create_data(100,3)

class Layer_Dense:
    def __init__(self,n_input,n_neuron):
        self.weights =0.10*np.random.randn(n_input,n_neuron)
        self.biases =0.10*np.random.randn(1,n_neuron)
    def forward(self,inputs):
        self.outputs=np.dot(inputs,self.weights)+self.biases

class Activation_function_ReLU:
    def forward(self,inputs):
        self.outputs=np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.outputs=probabilities
"""becuase the spiral will have 2 features as X and Y coordinate 
and input layer consist of number of features"""

###categorial loss entropy
class Loss:
    def calculate(self,output,y):
        sample_loss=self.forward(output,y)
        data_loss=np.mean(sample_loss)
        return data_loss

class Loss_categoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        sample=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidence=y_pred_clipped[range(sample),y_true]
        elif len(y_true.shape)==2:
            correct_confidence=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log=-np.log(correct_confidence)
        return negative_log

layer1=Layer_Dense(2,64)
Activation_1=Activation_function_ReLU()
Activation_1_SM=Activation_Softmax()
layer1.forward(X)
###print(layer1.outputs)
Activation_1.forward(layer1.outputs)
Activation_1_SM.forward(layer1.outputs)
###print(Activation_1.outputs)
###print(Activation_1_SM.outputs.shape)

 
layer2=Layer_Dense(64,2)
Activation_2=Activation_function_ReLU()
Activation_2_SM=Activation_Softmax()

layer2.forward(layer1.outputs)

Activation_2.forward(layer2.outputs)
Activation_2_SM.forward(layer1.outputs)
print(Activation_1_SM.outputs.shape)
###print(Activation_2.outputs)     

loss_function=Loss_categoricalCrossEntropy()
loss=loss_function.calculate(Activation_2_SM.outputs,y)
print("Loss:",loss)