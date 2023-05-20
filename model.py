# Purpose: Contains the model for the project.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.pipeline import make_pipeline
import pickle

def __normal_imputer(X : pd.DataFrame , random_state=None, inplace=False):
    '''Imputes the empty part of the columns using a list of random numbers
    with the same mean & std deviation as the available data
    
    Parameters
    ----------
    X : pd.DataFrame
        Data with empty cells & some data available
    
    random_state : int
        To reproduce the list of random numbers using np.normal.random
        (default: None)
    
    inplace : bool
        To avoid mutations in the org data
        (default: False)
    
    Returns
    -------
    X_imputed : pd.DataFrame
        if inplace is False:
            Data with cells imputed using random normal values
        None if inplace is True'''

    # to avoid mutations
    if inplace:
        X_imputed = X
    else:
        X_imputed = X.copy()
    
    # num of total rows in the data
    num_rows = X.shape[0]

    # for each column
    for col in range(X.shape[1]):
        # calculating mean & std
        mean = X.iloc[:, col].mean()
        std = X.iloc[:, col].std()
        
        # num of empty rows in the column
        num_nan_rows = np.isnan(X.iloc[:, col]).sum()

        if random_state is not None:
            np.random.seed(random_state)

        # imputed data assigned to empty rows
        X_imputed.iloc[(num_rows-num_nan_rows):, col] = np.random.normal(loc=mean, scale=std, size=num_nan_rows)

    if inplace:
        return
    else:
        return X_imputed

def preprocess(filepath):
    df = pd.read_csv(filepath)
    
    df['Financial Distress'] = df['Financial Distress'].apply(lambda x: 0 if x > -0.5 else 1)
    df.drop(columns=['Company', 'Time'], inplace=True)

    # Splitting target & features
    target = 'Financial Distress'
    X = df.drop(columns=target)
    y = df[target]

    # required bankrupt companies
    bankrupt = y[y==1].shape[0]
    total = y.shape[0]
    healthy = total - bankrupt
    required = healthy - bankrupt

    # creating empty records to be imputed
    null_samples = np.full((required, X.shape[1]), np.nan)
    null_samples = pd.DataFrame(null_samples)
    null_samples.columns = X.columns
    null_samples.reset_index(drop=True, inplace=True)

    # concatenating y_axis again
    X_y = pd.concat([X, y], axis=1)

    # Seperating bankrupt comapanies
    mask = X_y['Financial Distress'] == 1

    # concatenating bankrupt companies with null records
    X_bankrupt = pd.concat([X_y[mask].drop(columns='Financial Distress'), null_samples])

    # using normal/custom imputation technique
    X_bankrupt_over = __normal_imputer(X_bankrupt, random_state=42)

    # Concatenating healthy companies with imputed bankrupt data
    mask = X_y['Financial Distress'] == 0
    X_normal_over = pd.concat([X_y[mask].drop(columns='Financial Distress'), X_bankrupt_over])

    # creating extra target records
    over_target = np.ones(3400, dtype=int)
    over_target = pd.DataFrame(over_target)

    # concatenating extra target records
    y_normal_over = pd.concat([y, over_target]).squeeze()

    # top 43 features
    feat_imp = ['x35', 'x26', 'x83', 'x41', 'x12', 'x50', 'x75', 'x25', 'x34', 'x29', 'x65', 'x61', 'x79', 'x53', 'x23', 'x43', 'x36', 'x81', 'x14', 'x37', 'x5', 'x9', 'x60', 'x13', 'x15', 'x3', 'x22', 'x44', 'x42', 'x57', 'x10', 'x49', 'x31', 'x2', 'x16', 'x46', 'x52', 'x8', 'x38', 'x19', 'x59', 'x32', 'x47']
    # X_normal_over = SelectKBest(k=43).fit_transform(X_normal_over, y_normal_over)

    X_normal_over = X_normal_over[feat_imp]

    return X_normal_over, y_normal_over


def fit_model(X, y):
    bbc = make_pipeline(StandardScaler(), BalancedBaggingClassifier(
        RandomForestClassifier(
            max_depth=10, n_estimators=50, random_state=42, ccp_alpha=0.00054
        ), n_estimators=10, random_state=42
    ))

    bbc.fit(X, y)

    return bbc

def save_model(model, filepath):
    file = open(filepath, 'wb')
    pickle.dump(model, file)


if __name__ == "__main__":
    X, y = preprocess('data/bankruptcy-data-1.csv')
    bbc = fit_model(X, y)
    save_model(bbc, 'model.pkl')