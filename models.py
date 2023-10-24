import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, train_test_split)

####################################################################################################

def plsemodel (perc_split, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=42)

    mses_test = []
    mses_train = []
    r2s_test = []
    r2s_train = []
    ncomps = []

    for ncomp in range(1,15):
        pls = PLSRegression(ncomp)
        pls.fit(X_train, y_train)

        y_pred = pls.predict(X_train)
        y_pred_test = pls.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)

        r2s_train.append(r2_train)
        mses_train.append(mse_train)
        r2s_test.append(r2_test)
        mses_test.append(mse_test)
        ncomps.append(ncomp)

    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(ncomps, mses_test, '-o', color='black')
    plt.plot(ncomps, mses_train, '-o', color='red')
    plt.xlabel('Number of Components')
    plt.ylabel('MSE')
    plt.xticks(ncomps)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(ncomps, r2s_test, '-o', color='black')
    plt.plot(ncomps, r2s_train, '-o', color='red')
    plt.xlabel('Number of Components')
    plt.ylabel('R2')
    plt.xticks(ncomps)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

####################################################################################################
