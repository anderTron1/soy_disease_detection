#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:27:47 2020

@author: andre
"""
from architectures import Architectures
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import PIL
import tensorflow as tf
import pandas as pd

import pathlib

#from tensorlayer.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from os import listdir
from os.path import isfile, join, isdir

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from zipfile import ZipFile

import keras

from time import time
from datetime import datetime, timedelta

from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.callbacks import TensorBoard
from keras.models import load_model

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc


#from ann_visualizer.visualize import ann_viz


###############################################################################
#           DATABASE CONTENDO 4 TIPOS DE DOENÇAS EM PLANTAÇÕES DE SOJA        #
###############################################################################

#=========================baixar databases====================================

def linksDoencas():
    disease = {}

    #crestamento bacteriano
    disease['crestamentoBacteriano'] = 'https://www.digipathos-rep.cnptia.embrapa.br/jspui/bitstream/123456789/1076/3792/Soja%20%28Soybean%29%20-%20Crestamento%20Bacteriano%20%28Bacterial%20blight%29%20-%20Cropped.zip'
    #ferrugem asiatica
    disease['ferrugemAsiatica'] = 'https://www.digipathos-rep.cnptia.embrapa.br/jspui/bitstream/123456789/1149/2266/Soja%20%28Soybean%29%20-%20Ferrugem%20%28Rust%29%20-%20Cropped.zip'
    #Mancha Alvo
    disease['manchaAlvo'] = 'https://www.digipathos-rep.cnptia.embrapa.br/jspui/bitstream/123456789/1081/967/Soja%20%28Soybean%29%20-%20Mancha%20Alvo%20%28Southern%20blight%29%20-%20Cropped.zip'
    #Mildio
    disease['mildio'] = 'https://www.digipathos-rep.cnptia.embrapa.br/jspui/bitstream/123456789/1126/2307/Soja%20%28Soybean%29%20-%20Mildio%20%28Downy%20mildew%29%20-%20Cropped.zip'
    
    disease['saudavel'] = 'https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/349ac012-2948-4172-bbba-3bf8f76596fd'
    return disease

#função responsavel por baixar e extrair arquivos da base de dados de doenças
#retorna o caminho da pasta 'doencas'
def disease():
    disease = linksDoencas()
    class_names = np.array([val for val in  disease.keys()])

    dataDir = '' #caminho path do diretorio
    folder = 'doencas' #nome da pasta que conterá as doencas
    for key, path in disease.items():
        print()
        #baixar databases
        dataDir = tf.keras.utils.get_file(key, origin=path, untar=True)
        dataDir = pathlib.Path(dataDir)
        dataDir = str(dataDir).replace(key, '')
        print('database: {} baixada'.format(key))

        #criar pasta para doencas
        if isdir(dataDir+'/'+folder) == False:
            try:
                os.makedirs(dataDir +'/'+ folder)
                print('pasta: {} criada'.format(folder))
            except OSError:
                pass
        #extrair arquivos para pasta doencas
        if isfile(dataDir+key+'.tar.gz') and isdir(dataDir+folder+'/'+key) != True:
            z = ZipFile(dataDir+key+'.tar.gz', 'r')
            z.extractall(dataDir+folder+'/'+key)
            z.close()
            print('arquivo: {}.tar.gz descompactado e movido para pasta: {}'.format(key, folder))
        print()
    return str(dataDir+folder), class_names

        
#criar um discionario contendo o caminho das imagens do dataset
#para coloca-los em um arquivo .csv
def creatFileCSV(path, nameFileCSV):
    images = {}
    print(path)
    for i in listdir(path):
        if isdir(str(path) + '/' + i):#verificar os diretorios das imagens
            imagePath = listdir(str(path) + '/' + i) #colocar todos os arquivos em uma lista
            for j in range(len(imagePath)):
                if imagePath[j].split('*.jpg'):
                    imagePath[j] = str(path)+'/'+i+'/'+imagePath[j] #pegar caminho completo do arquivo
            images[i] = imagePath #adicionar lista de arquivo no discionario conforme a classe

    filePath = []
    namesClass = []

    #extrair itens do dic images e criar uma lista chamada filePath contendo todas as imagens de todas
    #as classes.
    #extrair o nome de cada classe e colocar em uma lista (nameClass) conforme suas respectivas quantidades
    #de itens contida em cada classe
    for key, path in images.items():
        cont = len(path)
        for i in range(cont):
            namesClass.append(key)
        for j in images[key]:
            filePath.append(j)

    data = {'class': namesClass, 'file_path': filePath}
    df = pd.DataFrame.from_dict(data)
    #print(df.shape)

    nameFile = '{}.csv'.format(nameFileCSV)
    df.to_csv(nameFile, index=False)


def getFileCSV(nameFileCSV):
    nameFile = '{}.csv'.format(nameFileCSV)
    imgTraining = pd.read_csv(nameFile)
    return imgTraining


def create_model(dm_resize, class_names, opt, drop, l2, resolution):

    #if dm_resize is 64:
    #    model = Architectures.architecture_VGG16_64(class_names, drop, l2)
    #elif dm_resize is 128:
    model = Architectures.architecture_VGG16_128(class_names, drop, l2, resolution)
    #elif dm_resize is 224:
    #    model = Architectures.architecture_VGG16_224(class_names, drop, l2)
    # elif dm_resize is 256:
    #    model = Architectures.architecture_VGG16_256(class_names, drop, l2)

    #model = Architectures.architecture_train_model(class_names, drop, l2)

    #model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def trainingDt(train_data, imgHeight, imgWidht, batchSize):
    dataGen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=45, zoom_range=0.25,
                        width_shift_range=0.25, height_shift_range=0.25,
                        shear_range=0.25, horizontal_flip=True, validation_split=0.2)
    
    train_ds = dataGen.flow_from_dataframe(train_data, directory='doencas', x_col='file_path', y_col='class',
                                           target_size=(imgHeight, imgWidht), class_mode='categorical', batch_size=batchSize, subset='training')#, color_mode="rgb")
    val_ds = dataGen.flow_from_dataframe(train_data, directory='doencas', x_col='file_path', y_col='class', shuffle=False,
                                         target_size=(imgHeight, imgWidht), class_mode='categorical', batch_size=batchSize, subset='validation')#, color_mode="rgb")
    return train_ds, val_ds

def testBD(model,test_data, imgHeight, imgWidht, batchSize):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(test_data, directory='doencas', x_col='file_path', y_col='class',shuffle=False,
                                           target_size=(imgHeight, imgWidht), class_mode='categorical', batch_size=batchSize)

    return test_generator

def conpileModel(model, otimizador, training, validation, epochs, lr_scheduler, early_stop, salve, inicialil_epoch):
    history = model.fit_generator(training, validation_data=validation, epochs=epochs, callbacks=[lr_scheduler, early_stop, salve], initial_epoch=inicialil_epoch)
    
    print(model.summary())
    return history

def run_evaluate_model(model, val_images, BS):#, class_names):
    # make predictions on the validation set
    print("[INFO] evaluating network into validation set...")
    
    predIdxs = model.predict(val_images, batch_size=BS)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    val_preds = np.argmax(predIdxs, axis=-1)
    
    labels = list(val_images.class_indices.keys())
    
    # show a nicely formatted classification report
    print ("Classification report : \n",classification_report(val_images.classes, val_preds, target_names=labels, digits=4))#, target_names=val_images.class_indices.keys()))
    print("O modelo obteve %0.4f de acuracia" % accuracy_score(val_images.classes, val_preds))
    #print("Matrix de confusão: \n", confusion_matrix(val_images.classes))
    
    
 
'''
#def testImg()
class_names = []

for key, classe in doencas.items():
    class_names.append(key)
print(class_names)

sunflower_url = "https://portalsyngenta.com.br/sites/default/files/Syngenta_thumb_JonathaBolzanManchaAlvo_DiamantinoMT_842x384.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(imgHeight, imgWidht)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
'''

def learning_rate_scheduling():
    # Learning rate scheduling
    #lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                          patience=20, verbose=1, factor=0.5, min_lr=1E-09)
    #initial_learning_rate = 0.1
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
    #                    decay_steps=100000, decay_rate=0.96, staircase=True)

    # Interrupts training when it measures no progress on the validation set
    # early_stop = keras.callbacks.EarlyStopping(patience=10,  restore_best_weights=True)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=10)
    # early_stop = keras.callbacks.EarlyStopping(patience=10)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                        min_delta=0.01, patience=20, mode='min', verbose=1)
    return lr_scheduler, early_stop

def get_optimizers(opt_name, lr):
    if opt_name is 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    elif opt_name is 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name is 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name is 'Adadelta':
        optimizer = keras.optimizer_v2.adadelta.Adadelta(learning_rate=lr)
    elif opt_name is 'Adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=lr)
    elif opt_name is 'Adamax':
        optimizer = keras.optimizers.Adamax(learning_rate=lr)
    elif opt_name is 'Nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=lr)

    return optimizer


def salve_model(file):
    filePath = file
    folder = 'result_logs'
    filePath += folder
    
    #criar pasta para doencas
    if isdir(filePath) == False:
        try:
            os.makedirs(filePath)
            print('pasta: {} criada'.format(folder))
        except OSError:
            pass
    
    checkpoint = ModelCheckpoint(filePath+'/CNN-modelo.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
    tensorboard = TensorBoard(log_dir=filePath+"/logs/{}".format(time())) 
    csv_logger = CSVLogger(filePath+'/training-model.log')
    
    callbacks_list = [checkpoint, tensorboard, csv_logger]
    
    return callbacks_list

def metricas(prediction, y_dados):
    
    print('Accuracy:', accuracy_score(y_dados, prediction))
    print('F1 score:', f1_score(y_dados, prediction))
    print('Recall:', recall_score(y_dados, prediction))
    print('Precision:', precision_score(y_dados, prediction))
    print('\n clasification report:\n', classification_report(y_dados,prediction))
    print('\n confussion matrix:\n',confusion_matrix(y_dados, prediction))

def main():

    #--------------------------------configurações basicas-------------------
    dm_resize = 32
    batchSize = 32
    imgHeight = dm_resize
    imgWidht = dm_resize
    epochs = 10

    drop = 0.2
    l2 = 0.0001
    l2 = 0.0
    
    inicialil_epoch =0

    #opt = 'Adamax'

     # create learning curves for different learning rates
     #learning_rates = 1E-3 com o adamx chegou a 0.92 de acuracia com 0.19 de loss 
    learning_rates = [1E-3]#, 1E-3]
    #learning_rates = [1E-3]
    print ("[INFO] search best learning rates between ", learning_rates)

    #optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizers = ['Adadelta']#, 'Adagrad', 'Adadelta']

    print ("[INFO] search best optimizers between ", optimizers)

    #dataDir, class_names = disease()
    fold_abs = os.path.abspath('database/doencas')
    print('Caminho absoluto: ', fold_abs)
    dataDir, class_names= fold_abs, ['crestamentoBacteriano', 'ferrugemAsiatica', 'manchaAlvo', 'mildio','saudavel']
    #salve_modelHDSF = salve_model(dataDir.replace('doencas', ''))
    salve_modelHDSF = salve_model('database')
    
    #creatFileCSV(dataDir, 'imageDisease')
    creatFileCSV(dataDir,'imageDisease')
    imgTraining = getFileCSV('imageDisease')
    
    print(imgTraining['class'].value_counts())
    #print(imgTraining)
    x_train, x_test  = train_test_split(imgTraining,test_size=0.2, random_state=42)
    
    #x_test['class'] = x_test['class'].map({'crestamentoBacteriano': '0', 'ferrugemAsiatica': '1', 'manchaAlvo':'2', 'mildio':'3', 'saudavel':'4'})
    #print(x_test)
    #x_tes, y_tes = train_test_split(x_test,test_size=0.1, random_state=42)
    
    #checkResults(history, opt, model, epochs)

    lr_scheduler, early_stop = learning_rate_scheduling()

    
    fig_acc, axes_acc = plt.subplots()
    #fig_acc.subplot()
    #plt.rcParams.update({'font.size': 15})

    fig_loss, axes_loss = plt.subplots()
    #fig_loss.subplot()

    tempoInicioOperacao = int(time())
    for i in range(len(learning_rates)):
        for j in range(len(optimizers)):

            opt_name = optimizers[j]
            optimizer = get_optimizers(opt_name, learning_rates[i])
            print ("[INFO] Learning rate: ", learning_rates[i])
            print ("[INFO] Optimizer: ", opt_name) # optimizer.get_config()

            #TIRAR COMENTARIO QUANDO TREINAR A REDE DO ZERO
            model = create_model(dm_resize, class_names, optimizer, drop, l2,dm_resize)
            
            #REINICIAR REDE APARTIR DE DADOS SALVOS
            #model = load_model(dataDir.replace('doencas', 'result_logs/1CNN-modelo.hdf5'))    
            #ann_viz(model)
            
            train_ds, val_ds = trainingDt(x_train, imgHeight, imgWidht, batchSize)
            
            history = conpileModel(model, optimizer, train_ds, val_ds, epochs, lr_scheduler, early_stop, salve_modelHDSF, inicialil_epoch)

            axes_acc.plot(history.history['accuracy'], label='train_acc')
            axes_acc.plot(history.history['val_accuracy'], label='val_acc')
            axes_acc.set_title("Opt={}. Lr={}".format(opt_name, str(learning_rates[i])))
            axes_acc.set_xlabel('Epochs')
            axes_acc.set_ylabel('Accuracy')
            axes_acc.legend(loc=0)

            axes_loss.plot(history.history['loss'], label='loss')
            axes_loss.plot(history.history['val_loss'], label='val_loss')
            axes_loss.set_title("Opt={}. Lr={}".format(opt_name, str(learning_rates[i])))
            axes_loss.set_xlabel('Epochs')
            axes_loss.set_ylabel('Loss')
            axes_loss.legend(loc=0)

            print ("[INFO] OK terminated...")
        #if j == 2: break
    tempoFinalOperacao = int(time())
    print('Tempo de operação:', str(timedelta(seconds=(tempoFinalOperacao - tempoInicioOperacao))))
    
    #EXECUTAR PARA TESTE
    train_ds, val_ds = trainingDt(x_train, imgHeight, imgWidht, batchSize)
    model = load_model(dataDir.replace('doencas', 'result_logs//CNN-modelo.hdf5'))
    test = testBD(model,x_test, imgHeight, imgWidht, batchSize)
    #print(model.summary())
    run_evaluate_model(model,test, batchSize)

    
    fig_acc.savefig(dataDir + "accuracy.pdf", bbox_inches='tight') # + str(learning_rates[i]) + ".pdf")
    fig_loss.savefig(dataDir + "loss.pdf", bbox_inches='tight') # + str(learning_rates[i]) + ".pdf")
    # plt.savefig(pathOutput + "loss_" + str(learning_rates[i]) + ".pdf")
if __name__ == '__main__':
    main()