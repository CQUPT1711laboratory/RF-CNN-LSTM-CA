import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


def mix():
    # The driver made in data_processing.py.
    driver = np.loadtxt('....txt')
    print(driver.shape)

    # The expansion of land use from start year to end year.
    data = np.loadtxt('...', skiprows=6)
    data = data.reshape(data.shape[0] * data.shape[1], 1)
    dataAll = np.hstack((driver, data))
    print(dataAll.shape)
    uniques = np.unique(dataAll, axis=0)
    x0 = uniques[uniques[:, -1] == 0]
    x1 = uniques[uniques[:, -1] == 1]
    x2 = uniques[uniques[:, -1] == 2]
    x3 = uniques[uniques[:, -1] == 3]
    x4 = uniques[uniques[:, -1] == 4]
    x5 = uniques[uniques[:, -1] == 5]
    x6 = uniques[uniques[:, -1] == 6]
    # x99 = uniques[uniques[:, -1] == -9999]
    np.random.shuffle(x0)
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    np.random.shuffle(x3)
    np.random.shuffle(x4)
    np.random.shuffle(x5)
    np.random.shuffle(x6)
    # np.random.shuffle(x99)
    # print("data_all.shape:", dataAll.shape)
    data_sample = np.concatenate((
        x0[0:int((x0.shape[0] + 1) / 10)],
        x1[0:int((x1.shape[0] + 1) / 10)], x2[0:int((x2.shape[0] + 1) / 10)],
        x3[0:int((x3.shape[0] + 1) / 10)], x4[0:int((x4.shape[0] + 1) / 10)],
        x5[0:int((x5.shape[0] + 1) / 10)], x6[0:int((x6.shape[0] + 1) / 10)],
        # x99[0:int((x99.shape[0] + 1) / 10)]
    ), axis=0)
    X_data = data_sample[:, 0:-1]
    Y_data = data_sample[:, -1]
    rfc = RandomForestClassifier(random_state=10, n_estimators=100, verbose=2, n_jobs=10)
    rfc = rfc.fit(X_data, Y_data)
    # predict_results = rfc.predict(X_test)
    #
    # print(accuracy_score(predict_results, Y_test))
    # print('-------------------')
    # conf_mat = confusion_matrix(Y_test, predict_results)
    # print(conf_mat)
    # print('-------------------')
    # print(classification_report(Y_test, predict_results))
    joblib.dump(rfc, f'rfc_lc2015one.model')


mix()
