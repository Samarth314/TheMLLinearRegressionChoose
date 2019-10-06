txt = input("Type the name of the dataset you would like to use to test this out: ")



def runMachine(txt):

    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import seaborn as sns
    import matplotlib.pyplot as plt

    exo_data = pd.read_csv(txt)
    # print(exo_data.head())
    # print(exo_data.describe())

    exo_X = exo_data.iloc[:, :-1].values
    exo_Y = exo_data.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(exo_X, exo_Y, test_size = 1/3, random_state = 0)




    from sklearn.linear_model import LinearRegression
    regressor = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    regressor.fit(X_Train, Y_Train)

    Y_Pred = regressor.predict(X_Test)

    num_correct = 0
    for i in range(len(Y_Pred)):
        if Y_Pred[i] == Y_Test[i]:
            print(Y_Pred[i], Y_Test[i])
            num_correct = num_correct+1
    
    print("The fraction of correctly classified examples in the test set is: " + str(num_correct / len(Y_Pred)))










runMachine(txt)


