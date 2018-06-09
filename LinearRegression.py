import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

np.set_printoptions(threshold=np.nan)

def start():
    trainingList = pd.read_csv('housing_training.csv', names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
    testingList = pd.read_csv('housing_test.csv', names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
    #each column value of training dataset
    CRIM    = trainingList['CRIM'].values.tolist()
    ZN      = trainingList['ZN'].values.tolist()
    INDUS   = trainingList['INDUS'].values.tolist()
    CHAS    = trainingList['CHAS'].values.tolist()
    NOX     = trainingList['NOX'].values.tolist()
    RM      = trainingList['RM'].values.tolist()
    AGE     = trainingList['AGE'].values.tolist()
    DIS     = trainingList['DIS'].values.tolist()
    RAD     = trainingList['RAD'].values.tolist()
    TAX     = trainingList['TAX'].values.tolist()
    PTRATIO = trainingList['PTRATIO'].values.tolist()
    B       = trainingList['B'].values.tolist()
    LSTAT   = trainingList['LSTAT'].values.tolist()
    MEDV    = trainingList['MEDV'].values.tolist()

    # each column value of testing dataset
    CRIM_T    = testingList['CRIM'].values.tolist()
    ZN_T      = testingList['ZN'].values.tolist()
    INDUS_T   = testingList['INDUS'].values.tolist()
    CHAS_T    = testingList['CHAS'].values.tolist()
    NOX_T     = testingList['NOX'].values.tolist()
    RM_T      = testingList['RM'].values.tolist()
    AGE_T     = testingList['AGE'].values.tolist()
    DIS_T     = testingList['DIS'].values.tolist()
    RAD_T     = testingList['RAD'].values.tolist()
    TAX_T     = testingList['TAX'].values.tolist()
    PTRATIO_T = testingList['PTRATIO'].values.tolist()
    B_T       = testingList['B'].values.tolist()
    LSTAT_T   = testingList['LSTAT'].values.tolist()
    MEDV_T    = testingList['MEDV'].values.tolist()

    predictResult = [] # prediction list

    X = np.mat([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).T # independent variables matrix
    X = np.insert(X, 0, np.ones(300), 1)
    Y = np.mat([MEDV]).T #dependent variables matrix

    B = (X.T * X).I * X.T * Y # optimization

    outputCSVFile(B) # export coifficient CSV file
    for x in range(0, len(testingList)):
        predictResult.append(calculateLR(B, CRIM_T[x], ZN_T[x], INDUS_T[x], CHAS_T[x], NOX_T[x], RM_T[x], AGE_T[x], DIS_T[x], RAD_T[x], TAX_T[x], PTRATIO_T[x], B_T[x], LSTAT_T[x]))

    plotResult(MEDV_T, predictResult)
    tempSum = 0
    for i in range(0, len(testingList)):
        temp = predictResult[i] - MEDV_T[i]
        temp = np.power(temp, 2)
        tempSum = tempSum + temp
    tempSum = tempSum / len(testingList)
    RMSE = math.sqrt(tempSum) # compute RMSE
    print('RMSE:',RMSE)

# This function calcuate LR model
def calculateLR(B,crim ,zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat):
    intercept = B[0, 0]
    LR = intercept + B[1, 0] * crim + B[2, 0] * zn + B[3, 0] * indus + B[4,0] * chas + B[5, 0] * nox + B[6, 0] * rm + B[7,0] * age + B[8, 0] * dis + B[9, 0] * rad + B[10, 0] * tax + B[11, 0] * ptratio + B[12, 0] * b + B[13, 0] * lstat
    return LR

#This function export coifficient CSV file
def outputCSVFile(B):
    coKey = ['Intercept', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']
    coValue = [B[0, 0], B[1, 0], B[2, 0], B[3, 0], B[4, 0], B[5, 0], B[6, 0], B[7, 0], B[8, 0], B[9, 0], B[10, 0], B[11, 0], B[12, 0], B[13, 0]]
    with open('coificientTable.csv', 'w+', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(coKey)
        wr.writerow(coValue)

# This function plot result
def plotResult(groundTruth,predictResult):
    fig = plt.figure()
    outputImg = fig.add_subplot(111)
    outputImg.scatter(predictResult, groundTruth)
    outputImg.set_xlabel('Prediction')
    outputImg.set_ylabel('Ground Truth')
    outputImg.set_xlim([0, 60])
    outputImg.set_ylim([0, 60])
    outputImg.plot((0,60),(0,60), 'r-')
    plt.savefig('Pred_GroundTruth.png')
    plt.show()

if __name__ == "__main__":
    start()