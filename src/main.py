import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, accuracy_score
from tabulate import tabulate


def setup_configuration(dataset_type, model_output_dir, least_learning_rate, reduction_by, patience):
    least_learning_rate = least_learning_rate
    reduction_by = reduction_by
    patience = patience
    verbose = 1
    image_size = (100, 100)
    input_shape = (100, 100, 3)
    base_dir = f'C:/CourseWork/ArtificialIntelligence/Python Projects/Natural Calamity Detection - Group Project/{dataset_type} Dataset'
    split_dir = os.path.join(base_dir, 'train_val_test_split')
    train_dir = os.path.join(split_dir, 'train')
    validation_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')
    output_dir = os.path.join(split_dir, 'output_files')
    model_out_dir = os.path.join(output_dir, model_output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    print(f"Model will be saved to: {model_out_dir}")
    labels = os.listdir(train_dir)
    num_classes = len(labels)
    return least_learning_rate, reduction_by, patience, verbose, image_size, input_shape, train_dir, validation_dir, test_dir, output_dir, model_out_dir, labels, num_classes


def build_data_generators(train_folder, validation_folder, test_folder, labels=None, image_size=(100, 100), batch_size=50):
    train_datagen = ImageDataGenerator(width_shift_range=0.0, height_shift_range=0.0, zoom_range=0.0, horizontal_flip=True, vertical_flip=True)
    validation_datagen = ImageDataGenerator(width_shift_range=0.0, height_shift_range=0.0, zoom_range=0.0, horizontal_flip=True, vertical_flip=True)
    test_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(train_folder, target_size=image_size, class_mode='sparse', batch_size=batch_size, shuffle=True, classes=labels)
    validation_gen = validation_datagen.flow_from_directory(validation_folder, target_size=image_size, class_mode='sparse', batch_size=batch_size, shuffle=False, classes=labels)
    test_gen = test_datagen.flow_from_directory(test_folder, target_size=image_size, class_mode='sparse', batch_size=batch_size, shuffle=False, classes=labels)
    return train_gen, validation_gen, test_gen


def faster_rcnn(input_shape, num_classes):
    img_input = Input(shape=input_shape, name='data')
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='backbone_conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='backbone_pool1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='backbone_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='backbone_pool2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='backbone_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='backbone_pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='rpn_conv1')(x)
    x = Conv2D(4, (1, 1), activation='linear', padding='same', name='rpn_bbox_pred')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=[x, predictions])
    return model


def train_and_evaluate_model(model, epochs, optimizer, verbose=1):
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=patience, verbose=verbose, factor=reduction_by, min_lr=least_learning_rate)
    save_model = ModelCheckpoint(filepath=model_out_dir + "/model.keras", monitor='val_accuracy', verbose=verbose, save_best_only=True, save_weights_only=False, mode='max')
    history = model.fit(x=trainGen, epochs=epochs, validation_data=validationGen, verbose=verbose, callbacks=[learning_rate_reduction, save_model])
    model.save(model_out_dir + "/model.keras")
    history_file_path = os.path.join(model_out_dir, 'training_history.pkl')
    with open(history_file_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    return history


def evaluate_model(model_out_dir, testGen, labels):
    trained_model = load_model(model_out_dir + "/model.keras")
    trained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    y_true = testGen.classes
    y_pred_probs = trained_model.predict(testGen)[1]
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    precision = round(precision_score(y_true, y_pred, average='weighted') * 100, 4)
    recall = round(recall_score(y_true, y_pred, average='weighted') * 100, 4)
    f1 = round(f1_score(y_true, y_pred, average='weighted') * 100, 4)
    mcc = round(matthews_corrcoef(y_true, y_pred), 4)
    accuracy = str(round(accuracy_score(y_true, y_pred) * 100, 4)) + '%'
    metrics_file_path = os.path.join(model_out_dir, 'evaluation_metrics.txt')
    with open(metrics_file_path, 'w') as metrics_file:
        metrics_file.write("Evaluation Metrics:\n")
        metrics_file.write(f"Accuracy: {accuracy}\n")
        metrics_file.write(f"Precision: {precision}\n")
        metrics_file.write(f"Recall: {recall}\n")
        metrics_file.write(f"F1-score: {f1}\n")
        metrics_file.write(f"MCC: {mcc}\n")
    # Load history
    history_file_path = os.path.join(model_out_dir, 'training_history.pkl')
    with open(history_file_path, 'rb') as history_file:
        history = pickle.load(history_file)
    # Plot training history
    output_layers = ['predictions']
    for layer in output_layers:
        train_key_acc = f'{layer}_accuracy' if f'{layer}_accuracy' in history else f'acc_{layer}'
        val_key_acc = f'val_{layer}_accuracy' if f'val_{layer}_accuracy' in history else f'val_acc_{layer}'
        train_key_loss = f'{layer}_loss' if f'{layer}_loss' in history else f'loss_{layer}'
        val_key_loss = f'val_{layer}_loss' if f'val_{layer}_loss' in history else f'val_loss_{layer}'
        plt.plot(history[train_key_acc], label=f'Training {layer.title()} Accuracy')
        plt.plot(history[val_key_acc], label=f'Validation {layer.title()} Accuracy')
        plt.title(f'Training and Validation {layer.title()} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.plot(history[train_key_loss], label=f'Training {layer.title()} Loss')
        plt.plot(history[val_key_loss], label=f'Validation {layer.title()} Loss')
        plt.title(f'Training and Validation {layer.title()} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_df, annot=True, fmt=".0f", cmap="Blues", linewidths=.2, cbar_kws={"shrink": 0.8}, annot_kws={"size": 12})
    plt.title('Confusion Matrix', fontweight='bold', fontsize=25, pad=20).set_bbox(dict(boxstyle='round,pad=0.25', edgecolor='black', facecolor='#bfa'))
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.subplot(1, 2, 2)
    metrics_data = [["Metric", "Value"], ["Accuracy", accuracy], ["Precision", precision], ["Recall", recall], ["F1-score", f1], ["MCC", mcc]]
    plt.title('Metrics', fontweight='bold', fontsize=25, pad=20).set_bbox(dict(boxstyle='round,pad=0.25', edgecolor='black', facecolor='#bfa'))
    plt.table(cellText=metrics_data[1:], colLabels=None, cellLoc='center', loc='center', bbox=[0.25, 0.5, 0.5, 0.5])
    plt.axis('off')
    plt.suptitle('Confusion Matrix with Metrics', y=1.05)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
    print(tabulate(metrics_data[1:], headers=metrics_data[0], tablefmt="fancy_grid", showindex=False))
    plt.show()


least_learning_rate, reduction_by, patience, verbose, image_size, input_shape, train_dir, validation_dir, test_dir, output_dir, model_out_dir, labels, num_classes \
    = setup_configuration(dataset_type='Finalized', model_output_dir='faster_rcnn_model', least_learning_rate=0.0001, reduction_by=0.05, patience=3)
trainGen, validationGen, testGen = build_data_generators(train_dir, validation_dir, test_dir, labels=labels, image_size=image_size, batch_size=50)
history_file_path = os.path.join(model_out_dir, 'training_history.pkl')
if os.path.exists(history_file_path):
    with open(history_file_path, 'rb') as history_file:
        saved_history = pickle.load(history_file)
else:
    model = faster_rcnn(input_shape, num_classes)
    saved_history = train_and_evaluate_model(model, epochs=25, optimizer='adadelta', verbose=1)
evaluate_model(model_out_dir, testGen, labels)
