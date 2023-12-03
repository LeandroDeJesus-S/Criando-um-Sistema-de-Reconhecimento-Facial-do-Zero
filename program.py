import numpy as np
import keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
import cv2
from mtcnn.mtcnn import MTCNN
import os
import requests
from typing import Any

BATCH_SIZE = 32
IMAGE_SIZE = 299, 299  # no mínimo (32, 32) como requerido pela MobileNetV2
CHANNELS = 3,  # 3 -> rgb, 1 -> gray, 4 -> rgba
IMAGE_SHAPE = IMAGE_SIZE + CHANNELS
CLASS_NAMES = []  # Ex: ['0', '1'] ou ['classe_x', 'classe_y']
EPOCHS = 500
CAM_IMG_SIZE = (965, 540)  # tamanho da imagem capturada pela webcam
VERBOSE_FIT = 0

def detect_face(img: np.ndarray, detector: Any, return_bounding_box=False, multi=False) -> tuple|list|np.ndarray:
    """usa o MTCNN para fazer a detecção dos rostos e os retorna.

    Args:
        img (np.ndarray or MatLike): imagem para detecção
        detector (Any): MTCNN ou detector com método `detect_faces`
        return_bounding_box (bool, optional): se verdadeiro retorna as coordenadas das boxes. Defaults to False.
        multi (bool, optional): se verdadeiro detecta mais de um rosto. Defaults to False.

    Returns:
        tuple or list or ndarray: rostos e/ou bounding boxes
    """
    faces = detector.detect_faces(img)
    if not faces and not return_bounding_box:
        if multi:
            return [np.array(img)]
        return np.array(img)
    
    elif not faces and return_bounding_box:
        if multi:
            return [np.array(img)], [None]
        return np.array(img), None
    
    if multi:
        detected_faces = []
        boxes = []
        for face in faces:
            box = face['box']
            x, y, w, h = box
            img_rect = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
            final_img_crop = cv2.resize(img_rect[y:y+h, x:x+w], IMAGE_SIZE)
            detected_faces.append(np.array(final_img_crop, np.uint8))
            boxes.append(box)
        
        if return_bounding_box:
            return detected_faces, boxes
        return np.array(detected_faces)

    box = faces[0]['box']
    x, y, w, h = box
    img_rect = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
    final_img_crop = cv2.resize(img_rect[y:y+h, x:x+w], IMAGE_SIZE)
    if return_bounding_box:
        return np.array(final_img_crop, np.uint8), box
    
    return np.array(final_img_crop, np.uint8)


def get_data(path: str, target_size: tuple, mtcnn: Any) -> tuple[np.ndarray]:
    """pega as imagens do diretório especificado detectando os rostos e
    os label retornando no formato X, y. O diretório deve ter o formato:
    
    train/
        class_1/
            img_1.jpg
            ...
        class_2/
            img_2.jpg
            ...

    Args:
        path (str): caminho do diretório
        target_size (tuple): tamanho desejado para converter as imagens
        mtcnn (Any): detector MTCNN ou equivalente

    Returns:
        tuple[ndarray]: tupla de arrays X e y
    """
    class_names = []
    X, y = [], []
    mtcnn
    for n, (root, dirs, files) in enumerate(os.walk(path)):
        if n == 0:
            class_names = dirs.copy()
            
        else:
            for file in files:
                path = os.path.join(root, file)
                image = keras.utils.load_img(path, target_size=target_size)
                img_arr = keras.utils.img_to_array(image, dtype=np.uint8)
                img_arr = detect_face(img_arr, mtcnn)
                X.append(img_arr)
                y.append(class_names[n-1])
    
    return np.array(X), np.array(y, dtype=np.int32)


def draw_box(img: np.ndarray, box: list, label: str) -> None:
    """desenha as bounding boxes na imagem.

    Args:
        img (np.ndarray ou MatLike): imagem na qual sera aplicada as boxes
        box (list): coordenadas para as bounding boxes (x, y, w, h)
        label (str): label resultante da classificação do resto da respectiva
        box
    """
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
    cv2.putText(
        img,
        label,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )
    return img


mtcnn = MTCNN()
X_train, y_train = get_data('train', target_size=IMAGE_SIZE, mtcnn=mtcnn)
X_val, y_val = get_data('val', target_size=IMAGE_SIZE, mtcnn=mtcnn)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inp = tf.keras.layers.Input(shape=IMAGE_SHAPE)
data_argumentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomRotation(.2)
])(inp)
ppi = tf.keras.applications.mobilenet_v2.preprocess_input(data_argumentation)
base = base_model(ppi, training=False)
avg_pool = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(base)
drop = tf.keras.layers.Dropout(.2)(avg_pool)
flat = tf.keras.layers.Flatten()(drop)
out = tf.keras.layers.Dense(1, activation='sigmoid')(flat)

model = tf.keras.Model(inputs=inp, outputs=out)

optimizer = keras.optimizers.Adam(learning_rate=.01)
ES = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    min_delta=.01,
    mode='max',
    restore_best_weights=True,
    start_from_epoch=50,
)
LR_decay = keras.callbacks.ReduceLROnPlateau(
    min_lr=.1,
    min_delta=.01,
    monitor='val_precision'
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(name='precision')
    ],
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=VERBOSE_FIT,
    batch_size=BATCH_SIZE
)

for k, v in history.history.items():
    print(k, v[-1])


def capture_and_detect(url: str, class_names: list[str], cam_window_size: tuple):
    """Captura a imagem da camera via app IP Webcam, detecta os rostos e faz
    a classificação.

    Args:
        url (str): url para acesso a camera do celular via url gerada pelo app
        class_names (list[str]): nome das classes para exibição nas bounding 
        boxes
    """
    names = class_names
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        imgd = cv2.imdecode(img_arr, -1)
        img = cv2.cvtColor(imgd, cv2.COLOR_BGR2RGB)

        results, boxes = detect_face(imgd, mtcnn, return_bounding_box=True, multi=True)
        for result, box in zip(results, boxes):
            result = cv2.resize(result, IMAGE_SIZE)
            result = np.array([result])
            pred = model.predict(result)
            pred = (pred > .5).astype(int)
            if box is not None:
                draw_box(img, box, names[pred.item()])


        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, cam_window_size)
        cv2.imshow("Android_cam", img)

        if cv2.waitKey(1) == 27:  # Pressionar 'Esc' para sair do loop
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    url = "sua/url/gerada/shot.jpg"  # mantenha /shot.jpg ao final da url
    capture_and_detect(
        url, 
        class_names=CLASS_NAMES, 
        cam_window_size=CAM_IMG_SIZE
    )
