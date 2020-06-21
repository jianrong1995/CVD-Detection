from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from scipy.stats import shapiro

def plotROC(groundTruth, prediction):
    fpr, tpr, threshold = metrics.roc_curve(groundTruth, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def parameterSweeping(classifier, param):
    gridSearcher = GridSearchCV(estimator = classifier,
                            param_grid = param, 
                            cv = 3, n_jobs = -1, iid=False, verbose=True, refit=True)
    return gridSearcher

def featureImportancePlot(fittedModel, headers):
    feature_imp = pd.Series(fittedModel.feature_importances_,index = headers).sort_values(ascending = False)
    sns.barplot(x = feature_imp, y = feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()
    print(feature_imp)

def histogramPlot(data):
    sns.distplot(data)

def scatterPlot(x, y):
    sns.jointplot(x = "x", y = "y")

def normalisedDataPCA(x_train):
    #normalisation
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    pca = PCA() 
    x_train = pca.fit_transform(x_train)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    plt.plot(np.cumsum(explained_variance))
    plt.grid(True)
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.show()

def executePCA(components, train):
    #normalisation
    sc = StandardScaler()
    train_scaled = sc.fit_transform(train)
    try:
        pca = PCA(n_components = components)
        pca_train = pca.fit_transform(train_scaled)
        return pca_train
    except Exception:
        print("Number of components is not correct!")

def normalityShapiroWilk(data, alpha = 0.05):
    stat, pvalue = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    if pvalue > alpha:
        print('Data looks Gaussian (fail to reject H0)')
    else:
        print('Data does not look Gaussian (reject H0)')    
