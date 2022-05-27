import pandas as pd
import datetime as datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import traceback


def group_top_x_amount_of_categories(df, col, amount_of_categories):
    new_df = df.copy()
    top_x_freq_cat = new_df[col].value_counts()[:amount_of_categories].index.tolist()
    #top_categories_filter = new_df[col].isin(top_x_freq_cat)
    #new_df[col + 'top' + str(amount_of_categories) + '_categories'] = new_df[col]
    new_df.loc[~new_df[col].isin(top_x_freq_cat),col] = 'out of top ' + str(amount_of_categories) + ' categories'
    return new_df



def mono_bin(Y, X, n = 20):
   
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1
    if len(d2) == 1:
        n = 3        
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, 
                           "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)
   
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
   
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
   
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 
             'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
   
    return(d3)


def char_bin(Y, X):
       
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
   
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
   
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
   
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*d3["WOE"]
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 
             'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
   
    return(d3)


def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
   
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(pd.Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
               
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
   
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


def plot_bivariant(final_iv, feature, feature_name_in_plot, specific_x_label_rotation = -1, categories_order_by_STR = True, string_int_ordering = False):
    #specific_x_label_rotation if needed , for some charts you can pass this value so that x labels rotate an specific_x_label_rotation amount
    plot_info = final_iv[final_iv['VAR_NAME'] == feature][['MIN_VALUE', 'MAX_VALUE', 'EVENT_RATE', 'COUNT']]
    feature_iv = final_iv[final_iv['VAR_NAME'] == feature][['IV']].iloc[0,0]
    continuous_feature = not plot_info['MIN_VALUE'].equals(plot_info['MAX_VALUE']) # there is no range, so max and min values are the same
    amount_of_bins = len(plot_info)
    max_amount_of_bins_for_visible_labels = 10
    if continuous_feature:
        plot_info['RANGE'] = plot_info['MIN_VALUE'].astype('Int64').astype('str') + '-' + plot_info['MAX_VALUE'].astype('Int64').astype('str')
        plot_info['RANGE'].replace('<NA>-<NA>', 'no info', inplace = True)
        plot_info.sort_values(by = 'RANGE', ascending = True, inplace = True)
    else:
        plot_info['RANGE'] = plot_info['MIN_VALUE'] # just sets each category as a 'range'
        plot_info['RANGE'].fillna('no data',inplace = True)
        if string_int_ordering:
            plot_info['RANGE'] = plot_info['RANGE'].astype(float).astype(int)
            plot_info.sort_values(by = 'RANGE', ascending = True, inplace = True)
        if categories_order_by_STR:
            plot_info.sort_values(by = 'EVENT_RATE', ascending = False, inplace = True)
    plot_info.reset_index(drop = True, inplace = True)
    #plotting data
    ax = plot_info[['RANGE', 'COUNT']].plot(kind = 'bar', x='RANGE', y = 'COUNT', figsize=(20,10), label='volumen (left)')
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        height_normalized = height / sum(plot_info['COUNT'])
        padding = max(plot_info['COUNT']) * 0.01
        if amount_of_bins > max_amount_of_bins_for_visible_labels:
            fontsize = 10
        else:
            fontsize = 15
        ax.annotate(f'{height_normalized:.2%}', (x + width/2, y + height + padding), ha='center', fontsize = fontsize)
    leg = plt.legend()
    ax2 = plot_info['EVENT_RATE'].plot( secondary_y=True, style='o-', color = 'orange', grid = True,  label='tasa de atraso')
    leg2 = plt.legend()
    ax.set_xlabel(feature_name_in_plot, fontsize=15)
    ax.set_ylabel('volumen', fontsize=15, labelpad=20)
    ax.right_ax.set_ylabel('tasa de atraso', rotation = 270, fontsize=15, labelpad=30)
    ax.set_title("tasa de atraso para " + feature_name_in_plot + " feature - "  + get_feature_predictor_importance(feature_iv) + " indicator", fontsize=15)
    plt.legend(leg.get_patches()+leg2.get_lines(),
               [text.get_text() for text in leg.get_texts()+leg2.get_texts()],
                fancybox=True, framealpha=1, shadow=True, borderpad=1)
    if amount_of_bins > max_amount_of_bins_for_visible_labels:
        ax.set_xticklabels(plot_info['RANGE'] ,fontsize = 15, rotation=80)
    if specific_x_label_rotation > -1:
        ax.set_xticklabels(plot_info['RANGE'] ,fontsize = 15, rotation=specific_x_label_rotation)
    leg.remove()
    plt.show()


def get_feature_predictor_importance(feature_iv):
    limits = [0.02, 0.1, 0.3, 0.5]
    if feature_iv < limits[0]:
        return('useless')
    elif feature_iv < limits[1]:
        return('weak')
    elif feature_iv < limits[2]:
        return('medium')
    elif feature_iv < limits[3]:
        return('strong')
    else:
        return('too good')