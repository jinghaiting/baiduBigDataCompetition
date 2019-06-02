from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from paths import model_path

tensorboard = TensorBoard(log_dir='./log',
                          write_graph=False,
                          write_grads=True,
                          write_images=True)

# 当标准评估停止提升时，降低学习速率
change_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.25,
                              patience=2,
                              verbose=1,
                              mode='auto',
                              min_lr=1e-7)
# 在每个训练期之后保存模型，最后保存的是最佳模型
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='val_acc',
                             mode='auto',
                             verbose=True)

callback_lists = [tensorboard, change_lr, checkpoint]
