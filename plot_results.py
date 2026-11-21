import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings("ignore")

No_of_Dataset = 3

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def Plot_ROC_Curve():
    cls = [ 'MDKR', 'RCO-DRNN', 'Transfer learning', 'GW-ARCANFD ', 'SF-AOA-GW-ARCANFD']
    for a in range(No_of_Dataset):  # For 5 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        # Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight':'bold', 'size':12})
        path = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()



def plotConvResults():
    Algorithm = ['TERMS','AOA-GW-ARCANFD', 'BCO-GW-ARCANFD', 'ARO-GW-ARCANFD', 'AOA-GW-ARCANFD', 'SF-AOA-GW-ARCANFD']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Fitness = np.load('Fitness_' + str(i + 1) + '.npy', allow_pickle=True)
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[1])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        plt.plot(length, Fitness[0, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Fitness[1, :], color='#7b5de1', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Fitness[2, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Fitness[3, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Fitness[4, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight':'bold', 'size':12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Table():
    eval = np.load('Evaluate_tab.npy', allow_pickle=True)
    Algorithm = ['Activation function','AOA-GW-ARCANFD', 'BCO-GW-ARCANFD', 'ARO-GW-ARCANFD', 'AOA-GW-ARCANFD', 'SF-AOA-GW-ARCANFD']
    Classifier = ['Activation function','MDKR', 'RCO-DRNN', 'Transfer learning', 'GW-ARCANFD ', 'SF-AOA-GW-ARCANFD']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']

    Graph_Terms = np.array([0, 2, 4, 5,7,9,10]).astype(int)
    Table_Terms = [0, 2, 4, 5,7,9,10]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['Sigmoid', 'Relu', 'Softmax', 'Leaky ReLU', 'Tanh']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batchsize)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_results():
    eval = np.load('Evaluate_plot.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4,5,6]
    Classifier = ['MDKR', 'RCO-DRNN', 'Transfer learning', 'GW-ARCANFD ', 'SF-AOA-GW-ARCANFD']
    Algorithm = ['AOA-GW-ARCANFD', 'BCO-GW-ARCANFD', 'ARO-GW-ARCANFD', 'AOA-GW-ARCANFD', 'SF-AOA-GW-ARCANFD']
    Batchsize = ['8', '16', '32', '48', '64']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j]+4]

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
            ax.set_facecolor("#f0f0f0")

            fig.canvas.manager.set_window_title('Epoch vs ' + Terms[Graph_Terms[j]])
            plt.plot(Batchsize, Graph[:, 0], color='k', linewidth=5, marker='o', markersize=15,
                     label=Classifier[0])
            plt.plot(Batchsize, Graph[:, 1], color='#9400d3', linewidth=5, marker='*', markersize=15,
                     label=Classifier[1])
            plt.plot(Batchsize, Graph[:, 2], color='darkblue', linewidth=5, marker='p', markersize=15,
                     label=Classifier[2])
            plt.plot(Batchsize, Graph[:, 3], color='#00bbf9', linewidth=5, marker='v', markersize=15,
                     label=Classifier[3])
            plt.plot(Batchsize, Graph[:, 4], color='#a30046', linewidth=5, marker='D', markersize=15,
                     label=Classifier[4])

            plt.xticks(Batchsize, ('8', '16', '32', '48', '64'), fontname="Arial", fontsize=12,
                       fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.grid(color='k', linestyle='-', linewidth=1, axis='y')
            plt.xlabel('Batchsize', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), fontsize=9, ncol=2, fancybox=True,
                       shadow=True, prop={'weight':'bold', 'size':10})
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Epoch vs ' + Terms[Graph_Terms[j]])
            plt.plot(Batchsize, Graph[:, 5], color='k', linewidth=5, marker='o', markersize=15,
                     label=Algorithm[0])
            plt.plot(Batchsize, Graph[:, 6], color='#9400d3', linewidth=5, marker='*', markersize=15,
                     label=Algorithm[1])
            plt.plot(Batchsize, Graph[:, 7], color='darkblue', linewidth=5, marker='p', markersize=15,
                     label=Algorithm[2])
            plt.plot(Batchsize, Graph[:, 8], color='#00bbf9', linewidth=5, marker='v', markersize=15,
                     label=Algorithm[3])
            plt.plot(Batchsize, Graph[:, 9], color='#a30046', linewidth=5, marker='D', markersize=15,
                     label=Algorithm[4])

            plt.xticks(Batchsize, ('8', '16', '32', '48', '64'), fontname="Arial", fontsize=12,
                       fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.grid(color='k', linestyle='-', linewidth=1, axis='y')
            plt.xlabel('Batchsize', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), fontsize=9, ncol=2, fancybox=True,
                       shadow=True, prop={'weight': 'bold', 'size': 10})
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')

            path = "./Results/Dataset_%s_%s_mThd_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()




def plot_proposed():
    eval = np.load('Evaluate_prop.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Classifier = ['MDKR', 'RCO-DRNN', 'Transfer learning', 'GW-ARCANFD ', 'SF-AOA-GW-ARCANFD']
    Algorithms = ['AOA-GW-ARCANFD', 'BCO-GW-ARCANFD', 'ARO-GW-ARCANFD', 'AOA-GW-ARCANFD', 'SF-AOA-GW-ARCANFD']
    Colors = ['#f4a261', '#2a9d8f', '#e76f51', '#264653', '#457b9d']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
            Graph_mean = np.mean(Graph, axis=0)
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_axes([0.17, 0.12, 0.75, 0.80])
            x = np.arange(len(Algorithms))
            bars = ax.bar(x,Graph_mean[0:5], width=1.0, color=Colors, edgecolor='white', alpha=0.95)
            for k, bar in enumerate(bars):
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                        f'{yval:.2f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=11, color=Colors[k])
            ax.set_xlim(-0.5, len(Algorithms) - 0.5)
            ax.set_xticks(x)
            ax.set_xlabel("Steps per Epochs", fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels(['100', '200', '300', '400', '500'],
                               fontsize=11, fontweight='bold')
            dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10)
                           for c in Colors]

            ax.legend(dot_markers, Algorithms, title='Algorithm', fontsize=10, title_fontsize=12,
                      loc='upper left', bbox_to_anchor=(-0.23, 0.98),columnspacing=0.8, handletextpad=0.3, frameon=False)
            ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
            ax.set_axisbelow(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            plt.tight_layout()
            plt.savefig(f'./Results/Dataset_{i + 1}_{Terms[Graph_Terms[j]]}_Proposed.png', dpi=300)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_axes([0.22, 0.12, 0.73, 0.80])
            x = np.arange(len(Classifier))
            bars = ax.bar(x, Graph_mean[0:5], width=1.0, color=Colors, edgecolor='white', alpha=0.95)
            for k, bar in enumerate(bars):
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                        f'{yval:.2f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=11, color=Colors[k])

            ax.set_xlim(-0.5, len(Classifier) - 0.5)
            ax.set_xticks(x)
            ax.set_xlabel("Steps per Epochs", fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels(['100', '200', '300', '400', '500'],
                               fontsize=11, fontweight='bold')
            dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10)
                           for c in Colors]

            ax.legend(dot_markers, Classifier, title='Classifier', fontsize=10, title_fontsize=12,
                      loc='upper left', bbox_to_anchor=(-0.30, 0.98), columnspacing=0.8, handletextpad=0.3,
                      frameon=False)
            ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
            ax.set_axisbelow(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            plt.tight_layout()
            plt.savefig(f'./Results/Dataset_{i + 1}_{Terms[Graph_Terms[j]]}_Proposed.png', dpi=300)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

def TableEx():
    eval = np.load('EvaluateEx.npy', allow_pickle=True)
    Algorithm = ['Activation function','AOA-GW-ARCANFD', 'BCO-GW-ARCANFD', 'ARO-GW-ARCANFD', 'AOA-GW-ARCANFD', 'SF-AOA-GW-ARCANFD']
    Classifier = ['Activation function','Mod 1', 'Mod 2', 'Mod 3', 'Mod 4', 'SF-AOA-GW-ARCANFD']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']

    Graph_Terms = np.array([0, 2, 4, 5,7,9,10]).astype(int)
    Table_Terms = [0, 2, 4, 5,7,9,10]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['Sigmoid', 'Relu', 'Softmax', 'Leaky ReLU', 'Tanh']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]
            #
            # Table = PrettyTable()
            # Table.add_column(Algorithm[0], Batchsize)
            # for j in range(len(Algorithm) - 1):
            #     Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            # print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
            #       '---------------------------------------')
            # print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 2):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k], '  Existing Comparison',
                  '---------------------------------------')
            print(Table)

if __name__ == '__main__':
    # Plot_ROC_Curve()
    # plotConvResults()
    # # Plots_results()
    # Table()
    TableEx()
    # plot_proposed()