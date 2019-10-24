import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

class Feature_Selector(BaseEstimator, TransformerMixin):
    '''
    Return a dataframe with selected columns
    '''

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.columns]
    
class Convert_LoanAmnt(BaseEstimator, TransformerMixin):
    '''
    Bin the loan amount by LendingClub's practice
    '''
    
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
  
    def transform(self, X):
        def div_Amnt(amnt):
            if amnt < 5000:
                x = 1
            elif amnt < 25000:
                x = 2
            elif amnt < 30000:
                x = 3
            elif amnt < 35000:
                x = 4
            else:
                x = 5
            return x
        
        df = X.copy()
        df[self.column] = df[self.column].apply(lambda x: div_Amnt(x))
        
        return df

class Convert_Term(BaseEstimator, TransformerMixin):
    '''
    Convert the term into groups 'first' and 'second'
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].apply(lambda x: 'short' if x == ' 36 months' else 'long')
        
        return df



class Convert_IntR(BaseEstimator, TransformerMixin):
    '''
    Bin the interest rate by LendingClub's practice
    Source: https://www.lendingclub.com/foliofn/rateDetail.action
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = pd.cut(df[self.column], bins = 25, labels = False) + 1
        
        return df

class Convert_Home(BaseEstimator, TransformerMixin):
    '''
    Convert the home ownership
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].apply(lambda x: 'OTHER' if x == 'ANY' or x == 'NONE' else x)
        
        return df

class Installment_To_Income(BaseEstimator, TransformerMixin):
    '''
    Calculate  (yearly installments) / (annual income)
    '''
    def __init__(self, ins, inc):
        self.ins = ins
        self.inc = inc
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        def f(*x):
            if x[1] != 0:
                return np.round(100 * x[0] * 12 / x[1])
            else:
                pass

        df = X.copy()
        df['inst_to_inc'] = df[[self.ins, self.inc]].apply(lambda x: f(*x), axis = 1)
        df.drop(columns = [self.ins, self.inc], inplace = True)
        
        return df

class Credit_Length(BaseEstimator, TransformerMixin):
    '''
    Calculate the credit length in year
    '''
    
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        now = datetime.datetime.now()
        df[self.column] = pd.to_datetime(df[self.column].apply(lambda x: x[-4:]), format = '%Y')
        df['credit_length'] = now.year - df[self.column].dt.year
        df.drop(columns = [self.column], inplace = True)
        
        return df

class Convert_FICO(BaseEstimator, TransformerMixin):
    '''
    Bin the FICO
    Source: https://www.lendingclub.com/loans/resource-center/understanding-credit-scores
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        def div_FICO(fico):
            if fico <= 579:
                x = 'poor'
            elif fico <= 669:
                x = 'fair'
            elif fico <= 739:
                x = 'good'
            elif fico <=799:
                x = 'very good'
            else:
                x = 'exceptional'
            return x
        
        df = X.copy()
        df['FICO_group'] = df[self.column].apply(lambda x: div_FICO(x))
        df.drop(columns = [self.column], inplace = True)
        
        return df

class Convert_DTI(BaseEstimator, TransformerMixin):
    '''
    Convert DTI
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].apply(lambda x: 'good' if x <= 40 else 'not good')
        
        return df                                  
    
class Total_Accts(BaseEstimator, TransformerMixin):
    '''
    Calculate the total accounts
    '''
    
    def __init__(self, column_1, column_2):
        self.column_1 = column_1
        self.column_2 = column_2
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['total_accts'] = df[self.column_1] + df[self.column_2]
        df.drop(columns = [self.column_1, self.column_2], inplace = True)
        
        return df
    
class Inquiry(BaseEstimator, TransformerMixin):
    '''
    Bin the inquiries in past 6 months
    '''
    
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].apply(lambda x: 'good' if x < 6 else 'not good')
        
        return df    
    
class Total_Trades(BaseEstimator, TransformerMixin):
    '''
    Calculate the total number of trades
    '''
    
    def __init__(self, column_1, column_2):
        self.column_1 = column_1
        self.column_2 = column_2
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['total_trades'] = df[self.column_1] + df[self.column_2]
        df.drop(columns = [self.column_1, self.column_2], inplace = True)
        
        return df
    
class Smote(BaseEstimator, TransformerMixin):
    '''
    Oversample the minor class. Use this if it is integrated into a pipeline.
    '''
    
    def fit(self, X, y = None):
        smote = SMOTE()
        X, y = smote.fit_sample(X, y)
        
        return self
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.smote.sample(X, y)
    
    def transform(self, X):
        return X