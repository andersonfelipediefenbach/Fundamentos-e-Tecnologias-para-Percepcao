

#importa as bibliotecas necessárias
import PIL, os, cv2
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report


# In[6]:


def get_data(dir_path):
    x, y = [], []
    category = {"Non Mask":0,"Mask":1}
    folders = os.listdir(dir_path)
    for folder in folders:
        folder_path = os.path.join(dir_path,folder)
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path,file)
            x.append(cv2.resize(cv2.imread(file_path),(224,224)))
            y.append(category[folder])
        print(folder,"Folder Done")
    x = np.array(x)
    y = np.array(y)
    x,y = shuffle(x,y)
    x = x / 255
    print("Shuffle and feature scaling Done")
    print("X Shape :",x.shape)
    print('Y Shape :',y.shape)
    print("Unique Categories :",np.unique(y,return_counts=True)[0])
    print("Unique Categories counts :",np.unique(y,return_counts=True)[1])
    return x, y


# In[7]:


xtrain, ytrain = get_data('D:/Mestrado/fundamentos/covid-face-mask-detection-dataset/New Masks Dataset/Train')


# In[8]:


xvalid, yvalid = get_data('D:/Mestrado/fundamentos/covid-face-mask-detection-dataset/New Masks Dataset/Validation')


# In[9]:


xtest, ytest = get_data('D:/Mestrado/fundamentos/covid-face-mask-detection-dataset/New Masks Dataset/Test')


# In[10]:


#cria a cnn
cnn = keras.Sequential(
    
                            [
                                
                                # Input
                                keras.layers.Input(shape=(224,224,3)),
                                
                                # CNN
                                keras.layers.Conv2D(200,(3,3),padding='same',activation='relu'),
                                keras.layers.Dropout(0.75),
                                keras.layers.MaxPool2D((2,2),padding='valid'),
                                keras.layers.Conv2D(200,(3,3),padding='same',activation='relu'),
                                keras.layers.Dropout(0.50),
                                keras.layers.MaxPool2D((2,2),padding='valid'),
                                keras.layers.Conv2D(200,(3,3),padding='same',activation='relu'),
                                keras.layers.Dropout(0.25),
                                keras.layers.MaxPool2D((2,2),padding='valid'),
                                keras.layers.Conv2D(200,(3,3),padding='same',activation='relu'),
                                keras.layers.MaxPool2D((2,2),padding='valid'),
                                
                                # Flatten
                                keras.layers.Flatten(),
                                
                                # Dense
                                keras.layers.Dense(100,activation='relu'),
                                keras.layers.Dense(50,activation='relu'),
                                keras.layers.Dense(10,activation='relu'),
                                
                                # Output
                                keras.layers.Dense(1,activation='sigmoid'),
                                
                            ]
    
                        )

cnn.summary()



# In[11]:


#plata o gráfico da cnn e suas camadas
keras.utils.plot_model(cnn)


# In[2]:


#treina o modelo
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
training = cnn.fit(xtrain,ytrain,batch_size=10,validation_data=(xvalid,yvalid),epochs=10)


# In[13]:


#exibe os parâmetros por epoch
training_history = pd.DataFrame(training.history)
training_history


# In[14]:


fig = px.line(training_history[['loss','val_loss']],labels={'value':'<-- Loss','index':'Epochs -->'})
fig.update_layout(title={'text':'Loss Per Epochs','font_size':23,'font_color':'orange','font_family':'Georgia','x':0.5})
fig.show()
fig = px.line(training_history[['accuracy','val_accuracy']],labels={'value':'Accuracy -->','index':'Epochs -->'})
fig.update_layout(title={'text':'Accuracy Per Epochs','font_size':23,'font_color':'orange','font_family':'Georgia','x':0.5})
fig.show()


# In[15]:


#Printa a classificação do modelo só com a CNN
ypred = []
for pred in cnn.predict(xtest):
    if pred > 0.5 :
        ypred.append(1)
    else : 
        ypred.append(0)
ypred = np.array(ypred)
print('\n\nConfusion Matrix : \n\n',confusion_matrix(ytest,ypred))
print('\n\nClassification Report : \n\n',classification_report(ytest,ypred))


# In[16]:


transfer = keras.applications.ResNet152V2()
for layer in transfer.layers:
    layer.trainable = False
inp = transfer.layers[0].input
out = transfer.layers[-2].output
out = keras.layers.Dense(50,activation='relu')(out)
out = keras.layers.Dense(25,activation='relu')(out)
out = keras.layers.Dense(10,activation='relu')(out)
out = keras.layers.Dense(5,activation='relu')(out)
out = keras.layers.Dense(1,activation='sigmoid')(out)
transfer = keras.Model(inputs=inp,outputs=out)
transfer.summary()


# In[17]:


keras.utils.plot_model(transfer)


# In[18]:


#treina agora com a Resnet
transfer.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
training = transfer.fit(xtrain,ytrain,batch_size=10,validation_data=(xvalid,yvalid),epochs=10)


# In[19]:


#Printa o valor de epoch com a Resnet
training_history = pd.DataFrame(training.history)
training_history


# In[20]:


fig = px.line(training_history[['loss','val_loss']],labels={'value':'<-- Loss','index':'Epochs -->'})
fig.update_layout(title={'text':'Loss Per Epochs','font_size':23,'font_color':'orange','font_family':'Georgia','x':0.5})
fig.show()
fig = px.line(training_history[['accuracy','val_accuracy']],labels={'value':'Accuracy -->','index':'Epochs -->'})
fig.update_layout(title={'text':'Accuracy Per Epochs','font_size':23,'font_color':'orange','font_family':'Georgia','x':0.5})
fig.show()


# In[21]:


#Mostra a precisão do modelo
ypred = []
for pred in transfer.predict(xtest):
    if pred > 0.5 :
        ypred.append(1)
    else : 
        ypred.append(0)
ypred = np.array(ypred)
print('\n\nConfusion Matrix : \n\n',confusion_matrix(ytest,ypred))
print('\n\nClassification Report : \n\n',classification_report(ytest,ypred))


# In[22]:


#Salva o modelo treinado para reconhecimento
transfer.save('covid_mask_detection_model.h5')


# In[23]:


model = keras.models.load_model('./covid_mask_detection_model.h5')


# In[24]:


def predict(img):
    if type(img) == str:
        img = cv2.imread(img)
    img = cv2.resize(img,(224,224))
    img = img / 255
    if model.predict(np.array([img]))[0] > 0.5:
        predict = 1 # Mask Recognized
    else:
        predict = 0 # Mask not Recognized
    return predict


# In[25]:


def view_prediction(img):
    if type(img) == str:
        img = cv2.imread(img)
    img = cv2.resize(img,(480,640))
    if predict(img) == 1:
        img = cv2.rectangle(img,(0,0),(640,50),(242,221,203),-1)
        img = cv2.putText(img,"Mask Detected",(30,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,61,142),2)
    else:
        img = cv2.rectangle(img,(0,0),(640,50),(186,186,245),-1)
        img = cv2.putText(img,"Mask not Detected",(30,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(56,56,255),2)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)


# In[26]:


#teste de pessoa sem máscara
view_prediction('D:/Mestrado/fundamentos/covid-face-mask-detection-dataset/New Masks Dataset/250.jpg')


# In[27]:


#teste de pessoa com máscara
view_prediction('D:/Mestrado/fundamentos/covid-face-mask-detection-dataset/New Masks Dataset/251.jpg')



