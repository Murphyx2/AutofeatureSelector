import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def cor_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()
    for i in feature_name:
        cor_list.append(np.corrcoef(X[i], y)[0, 1])

    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(),
                       n_features_to_select=num_feats,
                       step=10,
                       verbose=0
                       )

    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)

    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)

    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbc = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3,
                          reg_lambda=1, min_split_gain=0.01, min_child_weight=40, verbose=-1)

    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)

    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    player_df = pd.read_csv("fifa19.csv")
    num_feats = 30

    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols]

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns

    traindf = traindf.dropna()

    traindf = pd.DataFrame(traindf, columns=features)

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']
    # Nationality removed from the dataset
    for feature in X.columns:
        if 'Nationality' in feature:
            del X[feature]

    # Your code ends here
    return X, y, num_feats


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    selection_support = {'Feature': list(X.columns)}
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        selection_support['pearson'] = cor_support
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        selection_support['chi-square'] = chi_support
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        selection_support['rfe'] = rfe_support
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        selection_support['log-reg'] = embedded_lr_support
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        selection_support['rf'] = embedded_rf_support
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        selection_support['lgbm'] = embedded_lgbm_support

    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    feature_selection_df = pd.DataFrame(selection_support)

    df_numeric = feature_selection_df.map(lambda x: 1 if x is True else (0 if x is False else None))
    # count the selected times for each feature
    feature_selection_df['Total'] = df_numeric.sum(axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    best_features = feature_selection_df.head(10).Feature
    #### Your Code ends here
    return best_features


if __name__ == '__main__':
    best_features = autoFeatureSelector(dataset_path="fifa19.csv",
                                        methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
    print(best_features)
