print("Hello world")

# Modelo
# Crear el algoritmo de machine learning
#Importar las librerías relevantes

import numpy as np
import tensorflow as tf
import keras.models as km
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
from starlette.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

class Paciente(BaseModel):
    edad_actual: float 
    escolaridad: float 
    cecfamiliardm: float 
    ant_has: float 
    cantecedentesdm: float 
    imc_previo: float 
    sdg_ingreso: float 
    gestas: float
    peso_ultimohijo: float
    glucosa_ingreso: float
    class Config:
        schema_extra = {
            "example": {
                "edad_actual": 26.70, 
                "escolaridad": 2.63, 
                "cecfamiliardm": 1.0,
                "ant_has": 1.0,
                "cantecedentesdm": 1.0,
                "imc_previo": 25.42,
                "sdg_ingreso": 18.34,
                "gestas": 1,
                "peso_ultimohijo": 2.90,
                "glucosa_ingreso": 87.90
            }
        }

from fastapi import FastAPI, Form

app = FastAPI()

@app.on_event("startup")
def load_model():
    #global model
    global modelDG

    ##Cargar modelo entrenado
    modelDG = km.load_model('./modelo_v1')

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/predict')
def get_predict_DG(data: Paciente):
    received = data.dict()
    edad_actual = received['edad_actual']
    escolaridad = received['escolaridad']
    cecfamiliardm = received['cecfamiliardm']
    ant_has = received['ant_has']
    cantecedentesdm = received['cantecedentesdm']
    imc_previo = received['imc_previo']
    sdg_ingreso = received['sdg_ingreso']
    gestas = received['gestas']
    peso_ultimohijo = received['peso_ultimohijo']
    glucosa_ingreso = received['glucosa_ingreso']
    
    #estandarización de parámetros
    paramEstandar = pd.read_excel('./paramEstandar.xlsx')
    
    Eedad_actual = (edad_actual - paramEstandar[0][0]) / paramEstandar[1][0]
    Eescolaridad = (escolaridad - paramEstandar[0][1]) / paramEstandar[1][1]
    Ececfamiliardm = (cecfamiliardm - paramEstandar[0][2]) / paramEstandar[1][2]
    Eant_has = (ant_has - paramEstandar[0][3]) / paramEstandar[1][3]
    Ecantecedentesdm = (cantecedentesdm - paramEstandar[0][4]) / paramEstandar[1][4]
    Eimc_previo = (imc_previo - paramEstandar[0][5]) / paramEstandar[1][5]
    Esdg_ingreso = (sdg_ingreso - paramEstandar[0][6]) / paramEstandar[1][6]
    Egestas = (gestas - paramEstandar[0][7]) / paramEstandar[1][7]
    Epeso_ultimohijo = (peso_ultimohijo - paramEstandar[0][8]) / paramEstandar[1][8]
    Eglucosa_ingreso = (glucosa_ingreso - paramEstandar[0][9]) / paramEstandar[1][9]

    #pred_name = modelDG.predict([[edad_actual, escolaridad, cecfamiliardm, ant_has, cantecedentesdm, 
    #                            imc_previo, sdg_ingreso, gestas, peso_ultimohijo, glucosa_ingreso]]).tolist()[0]
    pred_name = modelDG.predict([[Eedad_actual, Eescolaridad, Ececfamiliardm, Eant_has, Ecantecedentesdm, 
                                Eimc_previo, Esdg_ingreso, Egestas, Epeso_ultimohijo, Eglucosa_ingreso]]).tolist()[0]
    return {'prediction': pred_name}

