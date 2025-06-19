import os
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from muscima.io import parse_cropobject_list
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import joblib
from matplotlib.image import imsave
import time

# RUTA DE LOS ARCHIVOS DE CROPOBJECTS
CROPOBJECT_DIR = os.path.join(
    os.path.expanduser('~'),
    'ruta dataset'
)

# CARGA DE ARCHIVOS
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

# DEFINICIÓN DE SÍMBOLOS MUSICALES (selección priorizada)
SYMBOLS_TO_CLASSIFY = [
    # Notas
    'notehead-full', 'notehead-empty',
    
    # Alteraciones
    'sharp', 'flat', 'natural',
    
    # Silencios
    'rest-quarter', 'rest-half', 'rest-whole',
    
    # Claves
    'g-clef', 'f-clef',
    
    # Compases
    'time-signature',
    
    # Barras
    'barline'
]

# EXTRACCIÓN DE SÍMBOLOS CON CONTEXTO
def extract_symbols_from_doc(cropobjects):
    _cropobj_dict = {c.objid: c for c in cropobjects}
    symbols = []
    
    for c in cropobjects:
        if c.clsname in SYMBOLS_TO_CLASSIFY:
            # Para notas, agregar tallo si existe
            if c.clsname.startswith('notehead'):
                stem_obj = None
                for o in c.outlinks:
                    _o_obj = _cropobj_dict.get(o)
                    if _o_obj and _o_obj.clsname == 'stem':
                        stem_obj = _o_obj
                        break
                if stem_obj:
                    symbols.append((c, stem_obj))
                else:
                    symbols.append((c,))
            else:
                symbols.append((c,))
    
    return symbols

# Obtener todos los símbolos
all_symbols = list(itertools.chain(*[extract_symbols_from_doc(cropobjects) for cropobjects in docs]))

# Contar ocurrencias por clase
symbol_counts = {}
for symbol in all_symbols:
    clsname = symbol[0].clsname
    symbol_counts[clsname] = symbol_counts.get(clsname, 0) + 1

print("Conteo de símbolos disponibles:")
for symbol, count in symbol_counts.items():
    print(f"{symbol}: {count} muestras")

# Filtrar símbolos con suficientes ejemplos (mínimo 100)
MIN_SAMPLES = 100
valid_symbols = [s for s in all_symbols if symbol_counts.get(s[0].clsname, 0) >= MIN_SAMPLES]
symbol_classes = [s[0].clsname for s in valid_symbols]
print(f"\nSímbolos seleccionados para clasificación (con al menos {MIN_SAMPLES} muestras):")
print(set(symbol_classes))

# GENERAR IMÁGENES DE SÍMBOLOS CON CONTEXTO ADICIONAL
def get_image(cropobjects, margin=5, context=10):
    # Calcular bounding box incluyendo contexto adicional
    top = min([c.top for c in cropobjects]) - context
    left = min([c.left for c in cropobjects]) - context
    bottom = max([c.bottom for c in cropobjects]) + context
    right = max([c.right for c in cropobjects]) + context
    
    # Asegurar que no salimos de los límites de la imagen
    top = max(0, top)
    left = max(0, left)
    
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    
    canvas = np.zeros((height, width), dtype='float32')
    
    for c in cropobjects:
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        
        # Asegurar que la máscara cabe en el canvas
        mask_height = min(c.height, height - _pt)
        mask_width = min(c.width, width - _pl)
        
        if mask_height > 0 and mask_width > 0:
            canvas[_pt:_pt+mask_height, _pl:_pl+mask_width] += c.mask[:mask_height, :mask_width].astype('float32')
    
    canvas[canvas > 0] = 1.0
    return canvas

# Generar imágenes y etiquetas
symbol_images = []
for symbol in valid_symbols:
    try:
        img = get_image(symbol)
        symbol_images.append(img)
    except Exception as e:
        print(f"Error procesando símbolo: {e}")

# Preprocesamiento: Redimensionar a 64x64 manteniendo relación de aspecto
def preprocess_image(img, target_size=(64, 64)):
    # Redimensionar manteniendo relación de aspecto
    h, w = img.shape
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(img, (new_h, new_w), anti_aliasing=True)
    
    # Crear canvas del tamaño objetivo
    canvas = np.zeros(target_size, dtype='float32')
    
    # Centrar la imagen en el canvas
    y_start = (target_size[0] - new_h) // 2
    x_start = (target_size[1] - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    
    return canvas

# Preprocesar todas las imágenes
symbols_processed = [preprocess_image(img) for img in symbol_images]

# Convertir a array numpy
X = np.array(symbols_processed)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(symbol_classes)
y = to_categorical(y_encoded)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test, symbols_train, symbols_test, classes_train, classes_test = train_test_split(
    X, y, symbols_processed, symbol_classes, 
    test_size=0.2, random_state=42, stratify=y_encoded
)

# =============================================================================
# 1. MODELO DE RED NEURONAL CONVOLUCIONAL (CNN)
# =============================================================================
print("\n" + "="*80)
print("ENTRENANDO RED NEURONAL CONVOLUCIONAL")
print("="*80)

# Añadir dimensión de canal para CNN
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Aumentación de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(X_train_cnn)

# Construir modelo CNN
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_cnn.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Entrenar CNN
start_time = time.time()
history = model_cnn.fit(
    datagen.flow(X_train_cnn, y_train, batch_size=32),
    steps_per_epoch=len(X_train_cnn) // 32,
    epochs=4,
    validation_data=(X_test_cnn, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
cnn_train_time = time.time() - start_time

# Evaluar CNN
test_loss, test_acc_cnn = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)
print(f'\nPrecisión CNN en prueba: {test_acc_cnn:.4f}')
print(f'Tiempo entrenamiento CNN: {cnn_train_time:.2f} segundos')

# Reporte de clasificación CNN
y_pred_cnn = model_cnn.predict(X_test_cnn)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nReporte de clasificación para CNN:")
print(classification_report(
    y_test_classes,
    y_pred_classes_cnn,
    target_names=label_encoder.classes_
))

# Matriz de confusión CNN
plt.figure(figsize=(10, 8))
cm_cnn = confusion_matrix(y_test_classes, y_pred_classes_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - CNN')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')
plt.show()

# =============================================================================
# 2. MODELO DE ENSAMBLE (RANDOM FOREST)
# =============================================================================
print("\n" + "="*80)
print("ENTRENANDO MODELO DE ENSAMBLE (RANDOM FOREST)")
print("="*80)

# Preparar datos para Random Forest (aplanar imágenes)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_flat = np.argmax(y_train, axis=1)  # Convertir one-hot a labels

# Crear y entrenar Random Forest
rf = RandomForestClassifier(n_estimators=100, 
                            random_state=42, 
                            n_jobs=-1,
                            verbose=1)

start_time = time.time()
rf.fit(X_train_flat, y_train_flat)
rf_train_time = time.time() - start_time

# Evaluar Random Forest
y_pred_rf = rf.predict(X_test_flat)
test_acc_rf = accuracy_score(y_test_classes, y_pred_rf)
print(f'\nPrecisión Random Forest en prueba: {test_acc_rf:.4f}')
print(f'Tiempo entrenamiento Random Forest: {rf_train_time:.2f} segundos')

print("\nReporte de clasificación para Random Forest:")
print(classification_report(
    y_test_classes,
    y_pred_rf,
    target_names=label_encoder.classes_
))

# Matriz de confusión Random Forest
plt.figure(figsize=(10, 8))
cm_rf = confusion_matrix(y_test_classes, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
plt.show()

# =============================================================================
# COMPARACIÓN DE RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)
print(f"Precisión CNN: {test_acc_cnn:.4f}")
print(f"Precisión Random Forest: {test_acc_rf:.4f}")
print(f"\nDiferencia de precisión: {abs(test_acc_cnn - test_acc_rf):.4f}")
print(f"Tiempo CNN: {cnn_train_time:.2f} segundos")
print(f"Tiempo Random Forest: {rf_train_time:.2f} segundos")

# Gráfico comparativo de precisión
plt.figure(figsize=(10, 6))
models = ['CNN', 'Random Forest']
accuracies = [test_acc_cnn, test_acc_rf]
times = [cnn_train_time, rf_train_time]

plt.subplot(1, 2, 1)
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title('Comparación de Precisión')
plt.ylim(0, 1.0)

plt.subplot(1, 2, 2)
sns.barplot(x=models, y=times, palette="rocket")
plt.title('Comparación de Tiempo de Entrenamiento')
plt.ylabel('Segundos')

plt.tight_layout()
plt.savefig('comparacion_modelos.png')
plt.show()

# =============================================================================
# GUARDAR MODELOS
# =============================================================================
model_cnn.save('symbol_classifier_cnn.keras')
joblib.dump(rf, 'symbol_classifier_rf.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# =============================================================================
# FUNCIÓN PARA PREDECIR CON AMBOS MODELOS
# =============================================================================
def predecir_y_comparar(ruta_imagen, model_cnn, model_rf, encoder, size=(64, 64)):
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.util import invert
    
    # Cargar y preprocesar imagen
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
    
    # Predecir con ambos modelos
    pred_cnn = model_cnn.predict(img_cnn)
    pred_class_idx_cnn = np.argmax(pred_cnn, axis=1)[0]
    pred_class_cnn = encoder.inverse_transform([pred_class_idx_cnn])[0]
    
    pred_rf = model_rf.predict(img_flat)
    pred_class_rf = encoder.inverse_transform(pred_rf)[0]
    
    # Mostrar resultados
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_processed, cmap='gray')
    plt.title(f'CNN: notehead-full')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_processed, cmap='gray')
    plt.title(f'RF: notehead-full')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparacion_prediccion.png')
    plt.show()
    
    return pred_class_cnn, pred_class_rf

# Ejemplo de uso
pred_cnn, pred_rf = predecir_y_comparar('ruta imagenprueba', model_cnn, rf, label_encoder)
print(f"Predicción CNN: {pred_cnn}")
print(f"Predicción Random Forest: {pred_rf}")
