import numpy as np
import joblib


def profit():
    rfc = joblib.load(f'rfc_lc2015one.model')
    driver = np.loadtxt('../cq21/1520/data/new2015_driverAll_NoNei.txt')
    print(driver.shape)

    # predict_results = rfc.predict(data_set)  # 预测结果
    # print(accuracy_score(predict_results, data))  # 算出精确的分数
    # print('-------------------')
    # conf_mat = confusion_matrix(data, predict_results)  # 算出混淆矩阵
    # print(conf_mat)
    # print('-------------------')
    # print(classification_report(data, predict_results))

    data_pro = rfc.predict_proba(driver)
    np.savetxt(f"nn.txt", data_pro, fmt='%f', delimiter=' ')

    # f, ax = plt.subplots(figsize=(7, 5))
    # ax.bar(range(len(rfc.feature_importances_)), rfc.feature_importances_)
    # ax.set_title("Feature importance")
    # f.show()
    print("rf:", rfc.feature_importances_)

profit()

