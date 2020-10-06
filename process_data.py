#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

np.random.seed(2)


# About the data
# 
# The German Credit Data contains data on 20 pieces of information (or variables) and the classification ('Creditability') whether an applicant is considered a Good or a Bad credit risk for 1000 loan applicants. The 20 variables include for each individual information such as gender, savings, credit history, years employed, credit amount, etc.
# 
# For the purpose of this tutorial, we filter the data and rearrange some some comlumns as described bellow.
# 
#     Creditability: 0 - credit-worthy, 1 - not credit-worthy
#     Account Balance: No account (1), None (No balance) (2), Some Balance (3)
#     Payment Status: Some Problems (1), Paid Up (2), No Problems (in this bank) (3)
#     Savings/Stock Value: (1) NoSaving, (2) BelowHundred 100 DM, Other (3, 4) [100, 1000] DM, 5 AboveThousand 1000 DM
#     Employment Length: (1, 2) Below 1 year (including unemployed), (3) [1, 4), (4) [4, 7), (5) Above 7
#     Sex/Marital Status: (1, 2) Male Divorced/Single, (3) Male Married/Widowed, (4) Female
#     No of Credits at this bank: (1) 1, (2, 3, 4) More than 1
#     Guarantor: (1) None, (2, 3) Yes
#     Concurrent Credits: (1, 2) Other Banks or Dept Stores, (3) None
#     Purpose of Credit: (1) New car, (2) Used car, (3, 4, 5, 6) Home Related, (0, 7, 8, 9, 10) Other
#     age groups: Young (1) [0, 25], (2) middle-age-adults [26, 39], (3) old adults [40, 59], (4, 5) seniors [60, 80)
# 

# In[3]:


COLUMNS_OF_INTEREST = ['Creditability', 'Account Balance', 'Payment Status of Previous Credit', 
                      'Value Savings/Stocks', 'Length of current employment', 'Sex & Marital Status', 
                      'No of Credits at this Bank', 'Guarantors', 'Concurrent Credits', 
                       'Purpose', 'Age (years)']


def plot_marginal_distribution(data, var1='AgeGroups', var2='Creditability', title=''):
    """
    Plot the marginal conditional distribution of var1,
    with respect to var2
    """
    Y = pd.crosstab(data[var1], data[var2], margins=True, margins_name='Total')
    for value in data[var2].unique():
        Y[value] = Y[value]*100/Y.loc['Total', value]

    P = Y.drop(columns=['Total']).transpose().copy()
    P.drop(columns=['Total'], inplace=True)
    P.plot.bar(stacked=True, figsize=(8, 6), title=title)



def pre_process_data(data):
    """
    load the data and process it as indicated above
    """

    df = data[[x for x in COLUMNS_OF_INTEREST if x not in ['Creditability']]].copy()
    df.rename(columns={'Payment Status of Previous Credit': 'Payment Status',
                       'No of Credits at this Bank': 'NumberCredits',
                       'Length of current employment': 'Employment Length',
                       'Value Savings/Stocks': 'Savings/Stock Value'}, inplace=True)

    df['Account Balance'] = df['Account Balance'].apply(lambda x: 'NoAccount'
                                                        if x == 1 else ('NoBalance' if x == 2 else
                                                                        'SomeBalance'))
    df['Payment Status'] = df['Payment Status'].apply(lambda x: 'SomeProblems'
                                                      if x == 1 else('PaidUp' if x == '2' else 'NoProblem'))
    df['Savings/Stock Value'] = df['Savings/Stock Value'].apply(lambda x: 'NoSavings'
                                                                if x == 1 else ('BellowHundred' if x == 2
                                                                               else ('AboveThousand'
                                                                                     if x == 5 else 'Other')))
    df['Employment Length'] = df['Employment Length'].apply(lambda x: 'BellowOneYear'
                                                            if x in [1, 2] else('OneToFour'
                                                                                if x == 3 else
                                                                                ('FourToSevent'
                                                                                 if x == 4 else 'AboveSevent')))
    df['Sex & Marital Status'] = df['Sex & Marital Status'].apply(lambda x: 'MaleSingle'
                                                                  if x in [1, 2] else ('MaleMarried'
                                                                                       if x == 3 else 'Female'))
    df['NumberCredits'] = df['NumberCredits'].apply(lambda x: 'One' if x == 1 else 'OnePlus')
    df['Guarantors'] = df['Guarantors'].apply(lambda x: 'No' if x == 1 else 'Yes')
    df['Concurrent Credits'] = df['Concurrent Credits'].apply(lambda x: 'NoCredit' if x == 3 else 'OtherBanks')
    df['Purpose'] = df['Purpose'].apply(lambda x: 'NewCar'
                        if x == 1 else ('UsedCar' if x == 2 else ('HouseRelated' if x in [3, 4, 5, 6] else
                                                                  'Other')))
    df['AgeGroups'] = df['Age (years)'].apply(lambda x:
                                             'Young' if x in range(0, 26) else (
                                                 'MidAgeAdult' if x in range(26, 40) else (
                                                     'OldAdult' if x in range(40, 60) else
                                                     'Senior')))
    return df


def plot_confusion_matrix_subgroups(estimator,
                                    X_test,
                                    y_test,
                                    fig_nrows,
                                    fig_ncols,
                                    figsize=(12, 6),
                                    key_column='Age (years)',
                                    class_names=['Good', 'Bad'],
                                    labels=[1, 0],
                                    groups=['Young', 'Old'],
                                    group_function=None,
                                    title='Confusion matrix within subgroups'):
    """
    Plot a confusion matrix within subgroups of a variable

    Args:
        estimator: a fitted classifier or a fitted Pipeline in which the last estimator is a classifier
        X_test: the test data
        y_test: the truth outcome
        fig_nrows: number of rows argument for subplot
        fig_ncols: number of columns argument for subplot
        figsize: figure size
        Key_column: Collumn for which win are interested in subgroups
        class_names: names for the predicted classes
        labels: labels associated to class names
        groups: categories within the data key column, those categories should be created
                by the group_function if they do not exist in the data yet
        group_function: a function to apply to the key column to create subgroups
        title: the title of the plot
    """
    fig, ax = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, figsize=figsize, sharex=False)
    data = X_test.copy()
    if group_function is not None:
        data['Groups'] = data[key_column].apply(group_function)

    for group, axis in zip(groups, [x for x in ax.flatten()]):
        tmp_data = data[data['Groups'] == group]
        disp = plot_confusion_matrix(estimator, tmp_data,
                                     y_test[tmp_data.index], display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize='all', values_format='.2f', labels=labels, ax=axis)
        axis.set_title(group)

    plt.suptitle(title)

