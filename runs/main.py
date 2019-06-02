from model import bulid_model
from preprocess.load_data import load_train_valid_data, load_test_data
from runs.model_util import train_model, eval, predict
from set_gpu import set_gpu
from callback import callback_lists

# 超参数
# reg = regularizers.l1(0)
num_classes = 9
epochs = 20
batch_size = 32


def main():
    #指定GPU
    set_gpu()

    # 加载数据
    train, valid = load_train_valid_data()
    test = load_test_data()

    # 建立模型
    model = bulid_model(num_classes)

    # 训练模型
    train_model(model, train, valid, batch_size, epochs, callback_lists)
    # 评估
    eval(model, valid)
    # 预测
    predict(model, test)


if __name__ == '__main__':
    main()