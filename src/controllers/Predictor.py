import pandas as pd
import seaborn as sns

import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_model(X, X_test, y, params, folds, plot_feature_importance=False, averaging='usual', model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(params,
                train_data,
                num_boost_round=20000,
                valid_sets = [train_data, valid_data],
                verbose_eval=1000,
                early_stopping_rounds = 200)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X.columns
        fold_importance["importance"] = model.feature_importance()
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.8f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    feature_importance["importance"] /= n_fold
    if plot_feature_importance:
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        plt.title('LGB Features (avg over folds)');

        return oof, prediction, feature_importance, y_pred, model
    return oof, prediction, scores, y_pred, model

def predict(games):
    df = beforeFit()
    oof_lgb, prediction_lgb, scores, y_pred, model = fit(df)
    makePredict(model, games)
    
def makePredict(model, data):
    df2 = pd.DataFrame(data)

    df2.head()

    df2['Team1'] = 0
    df2['Team2'] = 0
    for i in range(len(df2)):
      df2['Team1'][i] = teams_dict[df2['Team 1'][i]]
      df2['Team2'][i] = teams_dict[df2['Team 2'][i]]
    df2 = df2.drop(columns=['Team 1','Team 2'])

    y_pr = model.predict(df2.drop(columns=['Team 1 Win']))
    return y_pr

def fit(df):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    params = {'boost': 'gbdt',
              'feature_fraction': 0.05,
              'learning_rate': 0.01,
              'max_depth': -1,  
              'metric':'auc',
              'min_data_in_leaf': 50,
              'num_leaves': 32,
              'num_threads': -1,
              'verbosity': 1,
              'objective': 'binary'
             }

    df1 = df.drop(columns=['Team 1 Win'])

    y = df['Team 1 Win']

    X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size=0.33, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)

    return train_model(df1, X_test, y, params=params, folds=folds, plot_feature_importance=True)

def beforeFit():
    df2020 = pd.read_csv('eng_4.csv')
    df2019 = pd.read_csv('eng_2 (1).csv')
    df2018 = pd.read_csv('eng_2 (2).csv')
    cup2019 = pd.read_csv('eng_cup.csv')
    cup2018 = pd.read_csv('eng_cup (2).csv')

    df2020 = df2020.dropna()
    df2019 = df2019.dropna()
    df2018 = df2018.dropna()
    cup2019 = cup2019.dropna()
    cup2018 = cup2018.dropna()

    df =  pd.concat([df2019, df2020,df2018,cup2019,cup2018]).reset_index()

    df['FT1'] = 0
    df['FT2'] = 0
    df['Year'] = 0
    df['Month'] = ' '
    df['Day'] = 0
    df['WDay'] = ' '

    for i in range(len(df)):
      df['FT1'][i] = df['FT'][i].split('-')[0]
      df['FT2'][i] = df['FT'][i].split('-')[1]
      df['Year'][i] = df['Date'][i].split(' ')[3]
      df['Month'][i] = df['Date'][i].split(' ')[1]
      df['Day'][i] = df['Date'][i].split(' ')[2]
      df['WDay'][i] = df['Date'][i].split(' ')[0]

    df = df.drop(columns=['FT','Date','index','Round'])

    df = df.replace({'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6, 'Sun':7})
    df = df.replace({'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
                     'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12})

    df['Team 1 Win'] = 0
    for i in range(len(df)):
      if ((df['FT1'][i] - df['FT2'][i]) > 0):
        df['Team 1 Win'][i] = 1
      elif ((df['FT1'][i] - df['FT2'][i]) == 0):
        df['Team 1 Win'][i] = 2

    teams = pd.concat([df['Team 1'],df['Team 2']])
    team_list = []

    for item in teams: 
        if item not in team_list: 
            team_list.append(item)

    teams_dict = {}
    for i in team_list:
      teams_dict[i] = team_list.index(i)

    df['Team1'] = 0
    df['Team2'] = 0

    for i in range(len(df)):
      df['Team1'][i] = teams_dict[df['Team 1'][i]]
      df['Team2'][i] = teams_dict[df['Team 2'][i]]

    df = df.drop(columns=['Team 1','Team 2'])
    
    return df
