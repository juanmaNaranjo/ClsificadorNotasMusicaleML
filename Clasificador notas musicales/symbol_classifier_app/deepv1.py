import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # Usar Agg para evitar problemas con la interfaz gráfica
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import invert
from skimage.transform import resize

def preprocess_image(img, target_size=(64, 64)):
    h, w = img.shape
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(img, (new_h, new_w), anti_aliasing=True)

    canvas = np.zeros(target_size, dtype='float32')
    y_start = (target_size[0] - new_h) // 2
    x_start = (target_size[1] - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized

    return canvas

def predecir_y_comparar(ruta_imagen, model_cnn, model_rf, encoder, size=(64, 64)):
    img = imread(ruta_imagen)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)
    if np.mean(img) < 0.5:
        img = invert(img)

    img_processed = preprocess_image(img, size)
    img_flat = img_processed.reshape(1, -1)
    img_cnn = img_processed[np.newaxis, ..., np.newaxis]

    pred_cnn = model_cnn.predict(img_cnn, verbose=0)
    pred_class_idx_cnn = np.argmax(pred_cnn, axis=1)[0]
    pred_class_cnn = encoder.inverse_transform([pred_class_idx_cnn])[0]

    pred_rf = model_rf.predict(img_flat)
    pred_class_rf = encoder.inverse_transform(pred_rf)[0]

    # Guardar visualización en /static/
    output_path = os.path.join("static", "comparacion_prediccion.png")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_processed, cmap='gray')
    plt.title(f'CNN: {pred_class_cnn}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_processed, cmap='gray')
    plt.title(f'RF: {pred_class_rf}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return pred_class_cnn, pred_class_rf
