# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:15:17 2024

@author: jspark
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(targets, outputs, preds):
    # metric 계산 
    cm = confusion_matrix(targets, outputs)

    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(targets, outputs)
    pre = precision_score(targets, outputs)
    rec = recall_score(targets, outputs)
    auc = roc_auc_score(targets,preds)
    spe = tn/(fp+tn)
    F1 = f1_score(targets, outputs)
    summary = {
        'Accuracy': acc,
        'Precision': pre,
        'Recall': rec,
        'Specificity': spe,
        'F1 score': F1,
        'ROC-AUC score': auc        
    }
    for key, value in summary.items():
        print(f'{key} : {value}')
    
    # CM 그리기
    df = pd.DataFrame(cm)
    perc = df.copy()
    perc = perc / perc.sum().sum() * 100
    annot = df.astype(str) + "\n(" + perc.round(1).astype(str) + "%)"
    
    plt.figure(figsize=(7,5))
    #heatmap = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'fontsize':20}, fmt='.2%', cmap='Blues')
    heatmap = sns.heatmap(cm, annot=annot, annot_kws={'fontsize':16}, fmt='', cmap='Blues')
    #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), ha='right')
    #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), ha='right')
    heatmap.yaxis.set_ticklabels(['non-toxic', 'toxic'], fontsize=14)
    heatmap.xaxis.set_ticklabels(['non-toxic', 'toxic'], fontsize=14)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize = 16)
    plt.show()
    plt.close()
    
    # ROC Curve 그리기
    fpr, tpr, thresholds = roc_curve(targets, preds)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle = '--', color = 'k')
    plt.xlabel('False positive rate', fontsize = 16)
    plt.ylabel('True positive rate', fontsize = 16)
    AUROC = np.round(roc_auc_score(targets, preds), 2)
    plt.title(f'Binary Classification Model ROC curve; AUROC: {AUROC}', fontsize = 16);
    plt.show()
    plt.close()
    
    return summary

def output(output_path, smiles_list, labels, probs, ad_mf, ad_lf, KE, save_proba = True, AD = 1):
    if save_proba:
        df = pd.DataFrame({'SMILES':smiles_list, f'KE{KE} Predicted Labels':labels, f'KE{KE} Predicted Probabilities':probs})
    else:
        df = pd.DataFrame({'SMILES':smiles_list, f'KE{KE} Predicted Labels':labels})
    if AD == 'all':
        df[f'KE{KE} Applicability Domain 1'] = ad_mf
        df[f'KE{KE} Applicability Domain 2'] = ad_lf
    elif AD == 1:
        df[f'KE{KE} Applicability Domain'] = ad_mf
    elif AD == 2:
        df[f'KE{KE} Applicability Domain'] = ad_lf
    else:
        pass
    df.to_excel(output_path, index=False)
    return df

def merge(output_path, df_list):
    if len(df_list) == 1:
        return df_list[0]
    else:
        merged_df = df_list[0]
        for i in range(1, len(df_list)):
            merged_df = pd.merge(merged_df, df_list[i], on = 'SMILES')
        merged_df.to_excel(output_path, index=False)
        return merged_df
    
def applicability_domain_test(KE, script_path, test_vectors_mf, test_vectors_lf):   
    train_mf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_mf.xlsx'))
    train_mf_dist_matrix = euclidean_distances(train_mf)
    train_mf_mean_dist = np.mean(train_mf_dist_matrix[train_mf_dist_matrix != 0])
    ad_mf = []
    for test_vector in test_vectors_mf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_mf)
        check_ad = (test_train_distance < train_mf_mean_dist).sum()
        if check_ad == 0:
            ad_mf.append('out')
        else:
            ad_mf.append('in')
    
    train_lf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_lf.xlsx'))
    train_lf_dist_matrix = euclidean_distances(train_lf)
    train_lf_mean_dist = np.mean(train_lf_dist_matrix[train_lf_dist_matrix != 0])
    ad_lf = []
    for test_vector in test_vectors_lf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_lf)
        check_ad = (test_train_distance < train_lf_mean_dist).sum()
        if check_ad == 0:
            ad_lf.append('out')
        else:
            ad_lf.append('in')
            
    return ad_mf, ad_lf
    
def applicability_domain_average(KE, script_path, test_vectors_mf, test_vectors_lf):   
    train_mf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_mf.xlsx'))
    train_mf_dist_matrix = euclidean_distances(train_mf)
    train_mf_mean_dist = np.mean(train_mf_dist_matrix[train_mf_dist_matrix != 0])
    ad_mf = []
    for test_vector in test_vectors_mf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_mf)
        test_mean_dist = np.mean(test_train_distance[test_train_distance != 0])
        if test_mean_dist > train_mf_mean_dist:
            ad_mf.append('out')
        else:
            ad_mf.append('in')
    
    train_lf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_lf.xlsx'))
    train_lf_dist_matrix = euclidean_distances(train_lf)
    train_lf_mean_dist = np.mean(train_lf_dist_matrix[train_lf_dist_matrix != 0])
    ad_lf = []
    for test_vector in test_vectors_lf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_lf)
        test_mean_dist = np.mean(test_train_distance[test_train_distance != 0])
        if test_mean_dist > train_lf_mean_dist:
            ad_lf.append('out')
        else:
            ad_lf.append('in')
            
    return ad_mf, ad_lf
    
    
    
    
    
    
    
        
        