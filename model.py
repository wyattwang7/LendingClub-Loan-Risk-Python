import numpy as np
import pandas as pd
import dill
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 99
import plotly.graph_objects as go

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve,RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score, roc_curve,precision_score, recall_score, precision_recall_curve


from custom_transformer import Feature_Selector, Convert_LoanAmnt, Convert_Term, Convert_IntR, \
Convert_Home, Credit_Length, Convert_FICO, Convert_DTI, Smote

from bayesian_optimization import BayesSearchCV

def train_model(pipe, param_grid, CV, title, X_train, X_test, y_train, y_test):
    '''
    Train a machine learning model to predict survival on the Titanic.

    Parameters:
        pipe: Pipeline for model training.
        param_grid: Hyperparameters.
        data: training data

    Returns:
        best_model : The best model after hyperparameter tuning.
    '''
    rs = 0 #n_jobs = -1
    cv_search = CV(pipe, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 1, random_state = rs)
    cv_search.fit(X_train, y_train)
    
    best_model = cv_search.best_estimator_
    y_score = best_model.predict_proba(X_test)[:, 1]
    #y_pred = best_model.predict(X_test)
    #fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    #f1_score = f1_score(y_test, y_pred)
    
    result = "{} is {}"
    BOLD = '\033[1m'
    END = '\033[0m'
    print('AUC is {}{}{}'.format(BOLD, auc, END))
    print(result.format("Training score", cv_search.score(X_train, y_train)))
    print(result.format("CV score", cv_search.best_score_))
    print(result.format("Best parameter", cv_search.best_params_))
          
    #Plot the AUC curve
    #plot_roc_curve(fpr, tpr, auc)
    
    # plot the PR curve
    #thresholds = np.arange(0, 1.1, 0.1)
    #plot_precision_recall_curve(y_score, y_test, thresholds)
    
    # Plot the learning curve 
    #plot_learning_curve(best_model, X_train, y_train, 'Learning Curve of ' + title)
    
    # Persist the model
    with open(title + '.dill', 'wb') as f:
        dill.dump(best_model, f)
    
    return best_model

def plot_roc_curve(fpr, tpr, auc):
	'''
    Plot the roc curve.
    '''
    
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(fpr, tpr, label = 'AUC: {:.2f}'.format(auc))
    ax.plot([0, 1], [0, 1], '--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_score, y_test, thresholds):
	'''
    Plot the precision-recall-threshold curve.
    '''

    preds = [np.where(y_score > threshold, 1, 0) for threshold in thresholds]
    precisions = [precision_score(y_test, pred) for pred in preds]
    recalls = [recall_score(y_test, pred) for pred in preds]
    precision_recall = pd.DataFrame({'Precision': precisions, 'Recall': recalls, "Threshold": thresholds})
    

    precision_recall.plot(x = 'Threshold')
    plt.title('Precision, Recall vs. Threshold')
    plt.show()
    
    
def plot_cv_result(model, col, target1 = 'mean_test_score', target2 ='std_test_score'):
    '''
    Plot the hyperparameter vs. validation score from CV result.
    '''
    col = 'param_'+col
    result = pd.DataFrame(model.cv_results_)
    result = result.loc[:,[col, target1, target2]].sort_values(col).astype('float')

    fig = plt.figure(figsize = (10, 6))
    axes = fig.add_subplot(1, 2, 1)
    axes = sns.lineplot(x = col, y = target1, data = result)

    axes = axes = fig.add_subplot(1, 2, 2)
    axes = sns.lineplot(x = col, y = target2, data = result)
    plt.tight_layout()
    plt.show()
    
def feature_importance(estimator, cat_cols, threshold):
	 '''
    Output features meeting the threshold of feature imporance by tree model.
    '''
    
    fi = estimator.named_steps['classifier'].feature_importances_
    
    transformed_num_cols = estimator.named_steps['union'].transformer_list[0][1].steps[0][1].columns
    transformed_cat_cols = estimator.named_steps['union'].transformer_list[1][1].steps[2][1].get_feature_names(cat_cols)
    transformed_cols = np.append(np.array(transformed_num_cols).reshape(-1, 1), 
                             transformed_cat_cols)
   
    feature_ranking = pd.DataFrame(data = fi, index = transformed_cols, columns = ['Feature Importance']).sort_values('Feature Importance', ascending = False)
    cols = feature_ranking.loc[feature_ranking['Feature Importance'] > threshold]
    
    return cols
    
    
def plot_learning_curve(clf, X, y, title):
    '''
    Plot the test and training learning curve.
    
    Parameters:
        clf: estimator
        title: graph title
    Returns:
        Learning curve
    '''
    rs = 0
    
    train_sizes,train_scores,test_scores = learning_curve(clf,X,y,random_state = rs,cv = 5, scoring = 'roc_auc',shuffle = True)

    plt.figure()
    plt.title(title)
    
    ylim = (0.5, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha = 0.1,
                color = "r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",
        label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = "g",
        label = "Cross-validation score")

    plt.legend(loc = "best")
    plt.show()