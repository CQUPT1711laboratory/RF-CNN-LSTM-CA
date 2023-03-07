import numpy as np
import paddle
import paddle.nn as nn

window_size = 3
windowShape = 7
fea_num1 = windowShape
fea_num2 = windowShape
batch_size = 128
base_lr = 0.0001
EPOCH = 200


class CNN_LSTM(nn.Layer):
    def __init__(self, window_size, fea_num1, fea_num2):
        super().__init__()
        self.window_size = window_size
        self.fea_num1 = fea_num1
        self.fea_num2 = fea_num2
        # self.conv1 = nn.Conv3D(in_channels=1, out_channels=16, stride=1, kernel_size=3, padding='same')
        # self.relu1 = nn.ReLU()
        # self.pool = nn.MaxPool3D(kernel_size=2, stride=1, padding='same')
        # self.dropout = nn.Dropout3D(0.5)

        self.lstm1 = nn.LSTM(input_size=1 * fea_num1 * fea_num2, hidden_size=16, num_layers=1, time_major=False)
        self.fc = nn.Linear(in_features=16, out_features=7)
        # self.relu2 = nn.ReLU()
        # self.head = nn.Linear(in_features=8, out_features=7)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, self.window_size, self.fea_num1, self.fea_num2])
        x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.pool(x)
        # x = self.dropout(x)

        x = x.reshape([x.shape[0], self.window_size, -1])
        x, (h, c) = self.lstm1(x)
        x = x[:, -1, :]  # 最后一个LSTM只要窗口中最后一个特征的输出
        x = self.fc(x)
        # x = self.relu2(x)
        # x = self.head(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = CNN_LSTM(window_size, fea_num1, fea_num2)
    print("----------------main------------")
    # 定义超参数
    lr_schedual = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=EPOCH, verbose=True)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss(use_softmax=True)
    metric = paddle.metric.Accuracy()
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_schedual, beta1=0.9, beta2=0.999)

    factorsM1 = np.loadtxt('/home/aistudio/data/data172791/cq2000_res6.txt', skiprows=6)
    factorsM2 = np.loadtxt('/home/aistudio/data/data172791/cq2005_res6.txt', skiprows=6)
    factorsM3 = np.loadtxt('/home/aistudio/data/data172791/cq2010_res6.txt', skiprows=6)
    labelTrain = np.loadtxt('/home/aistudio/data/data172791/expan1015.txt', skiprows=6)
    factorsMAll = np.array([factorsM1, factorsM2, factorsM3])


    def trainDataSet(factorsM, expan):
        print(factorsM.shape, expan.shape)
        print("-------trainDataSet-------------")
        xboder = expan.shape[0]
        yboder = expan.shape[1]
        temp = []
        label = []
        for x in range(0, xboder, 1):
            for y in range(0, yboder, 1):
                if expan[x, y] != -9999:
                    label.append(expan[x, y])
                    window = np.ones((window_size, windowShape, windowShape), dtype=float) * (-9999)
                    row = 0
                    col = 0
                    for i in range(x - windowShape // 2, x + windowShape // 2 + 1):
                        for j in range(y - windowShape // 2, y + windowShape // 2 + 1):
                            if (0 <= i < xboder) and (0 <= j < yboder):
                                window[:, row, col] = factorsM[:, i, j]
                            col = col + 1
                        row = row + 1
                        col = 0
                    temp.append(window)
        return np.array(temp), np.array(label, dtype=int)


    def process(data, bs):
        l = len(data)
        tmp = []
        for i in range(0, l, bs):
            if i + bs > l:
                tmp.append(data[i:].tolist())
            else:
                tmp.append(data[i:i + bs].tolist())
        tmp = np.array(tmp)
        return tmp


    # ## ---------------------train---------------------------------
    pre_len = int(factorsMAll[0].shape[0] / 2)
    All_X, All_Y = trainDataSet(factorsMAll, labelTrain)
    print(All_X.shape, All_Y.shape)
    train_X = process(All_X, batch_size)
    train_Y = process(All_Y, batch_size)
    print(train_X.shape, train_Y.shape)  # (16,)(16, )
    for epoch in range(EPOCH):
        model.train()
        loss_train = 0
        for batch_id, data in enumerate(train_X):
            label = train_Y[batch_id]
            data = paddle.to_tensor(data, dtype='float32')
            label = paddle.to_tensor(label, dtype='int64')
            label = label.reshape([label.shape[0], 1])
            y = model(data)

            loss = loss_fn(y, label)
            opt.clear_grad()
            loss.backward()
            opt.step()
            loss_train += loss.item()
        print("[TRAIN] ========epoch : {},  loss: {:.4f}==========".format(epoch + 1, loss_train / batch_size))
        lr_schedual.step()

    # save model
    paddle.save(model.state_dict(), '/home/aistudio/work/cnn_lstm_w9all.params')
    paddle.save(lr_schedual.state_dict(), '/home/aistudio/work/cnn_lstm_w9all.pdopts')

    # ## ------------------------------------------预测------------------------------------
    model = CNN_LSTM(window_size, fea_num1, fea_num2)
    model_dict = paddle.load('/home/aistudio/work/cnn_lstm_w9all.params')
    model.load_dict(model_dict)
    factorsM4 = np.loadtxt('/home/aistudio/data/data172791/cq2015_res6.txt', skiprows=6)
    labelPre = np.loadtxt('/home/aistudio/data/data172791/expan1520.txt', skiprows=6)
    factorsMAll = np.array([factorsM2, factorsM3, factorsM4])

    def preDataSet(factorsM):
        print("-------preDataSet-------------")
        xboder = factorsM[0].shape[0]
        yboder = factorsM[0].shape[1]
        temp = []
        for x in range(0, xboder):
            for y in range(0, yboder):
                window = np.ones((window_size, windowShape, windowShape), dtype=float) * (-9999)
                row = 0
                col = 0
                for i in range(x - windowShape // 2, x + windowShape // 2 + 1):
                    for j in range(y - windowShape // 2, y + windowShape // 2 + 1):
                        if (0 <= i < xboder) and (0 <= j < yboder):
                            window[:, row, col] = factorsM[:, i, j]
                        col = col + 1
                    row = row + 1
                    col = 0
                temp.append(window)
        return np.array(temp)


    f = open('/home/aistudio/work/neiProb_w9.txt', 'wb')
    step = 10
    pre_len = int(factorsMAll[0].shape[0] / step)
    print("pre_len:", pre_len)
    scaled_true = labelPre.reshape(labelPre.shape[0] * labelPre.shape[1], 1)
    i = 0
    while (i != step):
        Pre_X = preDataSet(factorsMAll[:, pre_len * i:pre_len * (i + 1), :])
        print(Pre_X.shape)
        Pre_X = process(Pre_X, batch_size)
        print(Pre_X.shape)

        for _, data in enumerate(Pre_X):
            data = paddle.to_tensor(data, dtype='float32')
            prediction = model(data)
            prediction = prediction.cpu().numpy()
            np.savetxt(f, prediction)
        i += 1
        # np.savetxt(f'/home/aistudio/data/data172791/neiProb', prediction, fmt='%f', delimiter=' ')
        # scaled_prediction = np.argmax(prediction, axis=1)
        # labelPre=labelPre[pre_len*i:pre_len*i+pre_len:,:]
        # scaled_true = labelPre.reshape(labelPre.shape[0] * labelPre.shape[1], 1)
        # print('RMSE', np.sqrt(mean_squared_error(scaled_prediction, scaled_true)))
    print("--------------结束-------------")

    f.close()