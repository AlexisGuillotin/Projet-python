import re
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
import fasttext
import os
import numpy as np
from cv2 import cv2
from PIL import Image
from pytesseract.pytesseract import Output
from skimage import color
from skimage import io
import enchant
from enchant.checker import SpellChecker
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import Orange
import pickle
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


list_traite_compte = []
list_traite_salaire = []

#ATTENTION ! chemin à mofidier suivant l'arborescence.
compte = "releve compte 1"
salaire = "bulletin salaire 3 - fraude"

##ouvrir un pdf pour extraire le texte du compte
#print("Lecture du fichier "+compte+"...")
#with open("dossier_compte/"+compte+".pdf",'rb') as pdf:
#       filename =compte
#       #Question
#       images = convert_from_bytes(pdf.read(),500,use_pdftocairo=True, strict=False)
#       tmp = 0
#       for image in images:
#                if not os.path.exists('out/'+filename+'/'):
#                    os.makedirs('out/'+filename+'/')
#                image.save('out/'+filename+'/'+str(filename)+str(tmp)+'.png', 'PNG')
#                src = io.imread("out/"+filename+"/"+str(filename)+str(tmp)+".png")
#                #Question 3
#                image_gray = get_grayscale(src)
#                #Question 4
#                custom_config = r'--oem 3 --psm 6'
#                result_txt = pytesseract.image_to_string(image_gray, config=custom_config)
#                list_traite_compte.append(result_txt)
#                pdf.close()
#                tmp+=1

print("Lecture du fichier "+salaire+"...")
with open("dossier_salaire/"+salaire+".pdf",'rb') as pdf:
       filename =salaire
       #Question
       images = convert_from_bytes(pdf.read(),500,use_pdftocairo=True, strict=False)
       tmp = 0
       for image in images:
                if not os.path.exists('out/'+filename+'/'):
                    os.makedirs('out/'+filename+'/')
                image.save('out/'+filename+'/'+str(filename)+str(tmp)+'.png', 'PNG')
                src = io.imread("out/"+filename+"/"+str(filename)+str(tmp)+".png")
                #Question 3
                image_gray = get_grayscale(src)
                #Question 4
                custom_config = r'--oem 3 --psm 6'
                result_txt = pytesseract.image_to_string(image_gray, config=custom_config)
                list_traite_salaire.append(result_txt)
                with open("out/output.txt",'w') as output:
                    output.write(str(result_txt))
                pdf.close()
                tmp+=1
output.close()


#TODO UTILISER FASTTEXT C'EST UNE BETTE D'IDEE MON POTE

#Récupérer le salaire net de la fiche de paie
words = str(list_traite_salaire[len(list_traite_salaire)-1])
lines = words.split("\n")
for line in lines:
    print(line)
    if "NET PAYÉ" in line:
        print ("TROUVE ! " +line)
salaire_net = words[len(words)-14:len(words)-6]
#print("Salaire net : "+salaire_net)

#Relevé de compte
fraude = True
for page in list_traite_compte:
    if salaire_net in page:
        print("Il y a bien un virement qui correspond au montant affiché sur le buletin de salaire.")
        fraude = False
    elif salaire_net.replace('.',',') in page:
        print("Il y a bien un virement qui correspond au montant affiché sur le buletin de salaire.")
        fraude = False
if fraude == True:
    print("ATTENTION ! Il n'y a pas de virement qui correspond au montant inscrit sur la fiche de paie.")

#PART IA

