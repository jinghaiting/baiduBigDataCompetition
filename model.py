from keras import Model, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, concatenate


def bulid_model(num_classes):
    # 分支一 ：图像输入
    images_input_tensor = Input(shape=(100, 100, 3), name='input_model_images')
    base_model_images = DenseNet121(input_tensor=images_input_tensor, include_top=False, weights=None)
    x = base_model_images.output
    images_output = GlobalAveragePooling2D(name='images_output')(x)

    # 分支二 ：用户到访输入
    visits_input_tensor = Input(shape=(52, 28, 3), name='visit_input_tensor')
    base_model_visits = ResNet50(input_tensor=visits_input_tensor, include_top=False, weights=None)
    x = base_model_visits.output
    visits_output = GlobalAveragePooling2D(name='visits_output')(x)

    # 两分支连接
    output = concatenate([images_output, visits_output])
    output = Dense(128, activation='relu', name='fc', kernel_initializer='he_normal')(output)
    output = Dense(num_classes, activation='relu', name='Pre', kernel_initializer='he_normal')(output)

    model = Model(inputs=[images_input_tensor, visits_input_tensor], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', ])
    # model.summary()
    return model
