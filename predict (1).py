import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_data_from_directory(dirs):
    """
    Charge et normalise des images à partir d'un ensemble de répertoires donnés.

    :param dirs: Liste des chemins des répertoires à charger
    :return: tuple de données et étiquettes
    """
    data = []
    labels = []

    label_mapping = {
        'BBRP_TST': 0,  # barbe à papa (test)
        'PDA_TST': 1,  # pomme d’amour (test)
        'CHU_TST': 2,  # churros (test)
        'BBRP_F': 0,  # barbe à papa (train)
        'PDA_F': 1,  # pomme d’amour (train)
        'CHU_F': 2  # churros (train)
    }

    for dir_path in dirs:
        dir_name = os.path.basename(dir_path)
        label = label_mapping.get(dir_name, -1)

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                img = Image.open(file_path)
                img_gray = img.convert('L')
                img_resized = img_gray.resize((32, 32))
                img_array = np.array(img_resized)
                img_normalized = img_array / 255.0
                data.append(img_normalized)
                labels.append(label)

    return data, labels

# Chemins des répertoires
dirs1 = [
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Test\\BBRP_TST',
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Test\\CHU_TST',
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Test\\PDA_TST']
dirs2 = [
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Train\\BBRP_F',
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Train\\CHU_F',
    'C:\\Users\\DELL\\Downloads\\dataset\\DataSet_Train\\PDA_F'
]

# Charger les données de test
test_data, test_labels = load_data_from_directory(dirs1)
train_data, train_labels = load_data_from_directory(dirs2)
# Afficher les dimensions des données chargées
print(f"Nombre d'images chargées sur test: {len(test_data)}")
print(f"Nombre d'étiquettes sur test: {len(test_labels)}")
print(f"Nombre d'images chargées sur train: {len(train_data)}")
print(f"Nombre d'étiquettes sur train: {len(train_labels)}")
# Afficher quelques images avec leurs étiquettes
def display_sample_images(images, labels, n=5):
    plt.figure(figsize=(10, 2 * n))
    for i in range(min(n, len(images))):
        plt.subplot(n, 1, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_sample_images(test_data, test_labels, n=5)

# Transformer les datas[Train, Test] à une liste
train_data = np.array(train_data)
test_data = np.array(test_data)

#Transformer les labels à  une liste
train_labels = np.array(train_labels)

train_data = train_data.reshape(-1, 32*32)
test_data = test_data.reshape(-1, 32*32)

# Afficher la forme de cette image
print(train_data.shape)
print(test_data.shape)
#------------------------------------------------------------------------------------


input_size = 32*32
hidden_size = 350 # Vous pouvez choisir une autre taille pour la couche cachée.
output_size = 3

def init_params():
    W1 = np.random.randn(hidden_size, 1024) * np.sqrt(1. / 1024)
    B1 = np.random.randn(hidden_size,1)
    W2 = np.random.randn(3,hidden_size) * np.sqrt(1. / 1024)
    B2 = np.random.randn(3,1)
    return W1,B1,W2,B2

#la fonction d'activation
def tanh(x):
    return np.tanh(x)

#Fonction de propagaration sru laquelle on va utilisr la focntion tanh
# d'activation
train_data = train_data.T
test_data = test_data.T
def forward_prop(W1,B1,W2,B2,X):
    Z1 = W1.dot(X) + B1
    A1 = tanh(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = tanh(Z2)
    return Z1, A1, Z2, A2

# On switch les etiquette de [0,1,2] à
#       churros -> [1, 0, 0]
#       barbapapa -> [0, 1, 0]
#       pomme miel -> [0, 0, 1]
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y =one_hot_Y.T
    return one_hot_Y

def derivative_Tanh(x):
    return 1.0 - np.tan(x)**2

def back_prop(Z1, A1, Z2, A2, W2, X,Y):

   m = Y.size
   one_hot_Y = one_hot(Y)
   dZ2 = A2 - one_hot_Y
   dW2 = (1 / m )* dZ2.dot(A1.T)
   dB2 = (1 / m)* np.sum(dZ2, axis=1, keepdims=True)
   dZ1 = W2.T.dot(dZ2) * derivative_Tanh(Z1)
   dW1 = (1 / m )* dZ1.dot(X.T)
   dB1 = (1 / m)* np.sum(dZ1, axis=1, keepdims=True)
###   print("Dimensions des matrices/vecteurs :")
   #print("dZ2:", dZ2.shape)
   #print("dW2:", dW2.shape)
   #print("dB2:", dB2.shape)
   #print("dZ1:", dZ1.shape)
   #print("dW1:", dW1.shape)
   #print("dB1:", dB1.shape)
   return dW1, dB1, dW2, dB2

def update_weight(W1,B1,W2,B2,dW1,dB1,dW2,dB2,learn_rat):
    W1 = W1 - learn_rat * dW1
    B1 = B1 - learn_rat * dB1
    W2 = W2 - learn_rat * dW2
    B2 = B2 - learn_rat * dB2
    return W1,B1,W2,B2
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y,iterations, learn_rat):
    W1,B1,W2,B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW1, dB1,dW2,dB2 = back_prop(Z1, A1, Z2, A2, W2, X,Y)
        W1,B1,W2,B2 = update_weight(W1,B1,W2,B2,dW1,dB1,dW2,dB2,learn_rat)
        #Afficher l'accuracy tous les 10 epochs pour le suivi
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ",get_accuracy(get_predictions(A2),Y))
    return W1,B1,W2,B2

W1, B1, W2, B2 = gradient_descent(train_data,train_labels,600, 0.007)

