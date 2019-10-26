# LendingClub: Predict Loan Charge-Offs and Expected Loan Loss

LendingClub Corp. is an online lender who makes loans online to consumers and sells the loans to investors.

In this project, we analyze the loan data of LendingClub from the year 2007 to the second quarter in 2019. The data explores the demographic dimensions of personal loans and loan status (fully paid/ charged off), which is available on their website once you create an account. **Our goal** is to identify the risk of unsecured personal loans. Specifically, a machine learning model was developed to predict the probability of full payment and charge off. On top of that, we utilize the model to predict the expected loan loss from current borrowers, providing insights about risk control and loss reduction.

[Here](https://nbviewer.jupyter.org/github/wyattwang7/LendingClub/blob/master/lending_club.ipynb) is the static html file of lending_club.ipynb.

1. Keywords  
downcasting dtypes, correlation matrix, custom transformers, baseline model, feature selection w/ regularization, feature complexity, oversampling, precision-recall-threshold, loan loss

2. Summary
* Data: 2.2+ GB
* Data Preprocessing: Memory footprint down 73%
* Feature Engineering: Create custom transformers
* Model: Logistic regression With L1 regularization (AUROC 0.72)
* Business Insights:
  - Top risk factors  
    Fico, credit length, dti, annual income, interest rate, subgrade, inquiries in last 6 months and their interactions.    
  - Business objectives  
    Explained by the trade-off between preicision and recall.  
  - Loan Loss  
    `Expected Credit Loss = LGD x PD x EAD`  
    LGD(%): Mmount unrecovered by the lender after selling the underlying asset if a borrower defaults on a loan.  
    **PD**: Probability of default, which is calculated by the **Logistic regression model**.  
    EAD: Remaining outstanding principal. 
