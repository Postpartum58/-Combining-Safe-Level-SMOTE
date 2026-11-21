from numpy import matlib
from sklearn.utils import shuffle
import pandas as pd
from AOA import AOA
from ARA import ARA
from ARO import ARO
from BCO import BCO
from Model_Bert import Model_Bio_ClinicalBERT
from plot_results import *
from Global_Vars import Global_Vars
from Model_CNN import Model_CNN
from Model_GAFWPDARCAN import Model_GAFWPDARCAN
from Model_ResNet import Model_ResNet
from Model_Vgg16 import Model_vgg16
from Proposed import Proposed
from Objective_Function import objfun
from model_safe_emote import balance_with_smote



No_of_dataset = 3


# Read dataset 1
an = 0
if an == 1:
    DataFrame = pd.read_excel('./Dataset/1/1. Data_sheet_of_Survey_colected from hospital (1).xlsx')
    Data = DataFrame.values[1:, :]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = np.random.choice(["Yes", "No"], size=(1000, 1))
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    np.save('Data_1.npy', Data)
    np.save('Target_1.npy', target)

# Read Dataset 2
an = 0
if an == 1:
    DataFrame = pd.read_csv('./Dataset/2/2. post_natal_data git hub .csv')
    DropData = DataFrame.drop(['Feeling anxious'], axis=1)
    DropData = DropData.astype('string').fillna('')
    Data = DropData.values[:, 1:]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = np.asarray(DataFrame.get(['Feeling anxious']))
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    np.save('Data_2.npy', Data)
    np.save('Target_2.npy', target)

# Read Dataset 3
an = 0
if an == 1:
    DataFrame = pd.read_csv('./Dataset/3/3. Synthetic_PostPartum_Depression_Dataset_2000.csv')
    DropData = DataFrame.drop(['Feeling anxious'], axis=1)
    DropData = DropData.astype('string').fillna('')
    Data = DropData.values[:, 1:]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = np.asarray(DataFrame.get(['Feeling anxious']))
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1
    np.save('Data_3.npy', Data)
    np.save('Target_3.npy', target)

# Preprocessing
an = 0
if an == 1:
    for i in range(No_of_dataset):
        raw_data = np.load('Data_'+str(i+1)+'.npy',allow_pickle=True)
        df = pd.DataFrame(raw_data)
        if isinstance(df.columns, pd.Index) and df.columns.dtype == 'object':
            df.columns = (
                df.columns.str.strip()
                          .str.replace(' ', '_')
                          .str.replace('-', '_')
            )
        df = df.replace(['?', '-', 'NA', 'nan', 'NaN', 'None', ''], np.nan)
        df = df.fillna(0)
        np.save('preprocessed_data_'+str(i+1)+'.npy', np.asarray(df))

# Word to Vector conversion
an = 0
if an == 1:
    for n in range(No_of_dataset):
        Data =np.load('preprocessed_data_'+str(n+1)+'.npy',allow_pickle=True)
        features = Model_Bio_ClinicalBERT(Data)
        np.save('features_'+str(n+1)+'.npy', features)

# Data balancing
an = 0
if an == 1:
    for n in range(No_of_dataset):
        X = np.load('features_'+str(n+1)+'.npy',allow_pickle=True)
        y = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True).ravel()
        X_bal= balance_with_smote(X, y)
        np.save('balanced_data_'+str(n+1)+'.npy', X_bal)


# optimization
an = 0
if an == 1:
    for n in range(No_of_dataset):
        Data = np.load('balanaced_data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count,  Learning Rate, No of epoches
        xmin = matlib.repmat([5, 0.01, 5], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 50], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("AOA...")
        [bestfit1, fitness1, bestsol1, time] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("ARO...")
        [bestfit2, fitness2, bestsol2, time1] = ARO(initsol, fname, xmin, xmax, Max_iter)  # ARO

        print(" BCO...")
        [bestfit3, fitness3, bestsol3, time2] = BCO(initsol, fname, xmin, xmax, Max_iter)  # BCO

        print("ARA...")
        [bestfit4, fitness4, bestsol4, time3] =ARA(initsol, fname, xmin, xmax, Max_iter)  #ARA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  #Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
        Time = [time, time1, time2, time3, time4]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)
        np.save('Fitness_' + str(n + 1) + '.npy', Fitness)
        np.save('Time_' + str(n + 1) + '.npy', Time)

# classification
an = 0
if an == 1:
    Evaluate = []
    for n in range(No_of_dataset):
        Feat = np.load('balanaced_data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Steps_per_epoch = [100, 200, 300, 400, 500]
        for act in range(len(Steps_per_epoch)):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = BestSol[j, :]
                Eval[j, :], pred = Model_GAFWPDARCAN(Train_Data, Train_Target, Test_Data, Test_Target,
                                               Steps_per_epoch[act], sol)
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                         Steps_per_epoch[act])
            Eval[6, :], pred2 = Model_vgg16(Train_Data, Train_Target, Test_Data,
                                            Test_Target, Steps_per_epoch[act])
            Eval[7, :], pred3 = Model_ResNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                             Steps_per_epoch[act])
            Eval[8, :], pred4 =Model_GAFWPDARCAN(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Steps_per_epoch[act])
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Evaluate_prop.npy', Evaluate)  # Save Eval all


Plot_ROC_Curve()
plotConvResults()
Plots_results()
Table()
plot_proposed()

