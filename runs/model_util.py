import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from paths import result_data_path, test_ids_npy


def train_model(model, train, valid, batch_size, epochs, callback_lists):
    model.fit(x=[train[0], train[1]],
              y=train[2],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([valid[0], valid[1]], valid[2]),
              callbacks=callback_lists
              )


def eval(model, valid):
    # 预测
    y_pred = model.predict([valid[0], valid[1]])

    # one-hot ==> 标签
    y_test = np.argmax(valid[2], axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # 计算准确率、精确率、召回率、F1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("accuracy_score = %.2f" % accuracy)
    print("precision_score = %.2f" % precision)
    print("recall_score = %.2f" % recall)
    print("f1_score = %.2f" % f1)


def predict(model, test):
    label_map = {}
    for i in range(9):
        label_map[i] = str(i + 1).zfill(3)
    labels_pred = []

    preds = model.predict([test[0], test[1]])
    y_preds = np.argmax(preds, axis=1)

    for y_pred in y_preds:
        label_pred = label_map[y_pred]
        labels_pred.append(label_pred)

    with open(result_data_path, 'w') as f:
        test_ids = np.load(test_ids_npy)
        for i in range(10000):
            f.write(test_ids[i] + '\t' + labels_pred[i] + '\n')
