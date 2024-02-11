import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
# Define dataset paths
train_data_dir = r'C:\Users\HP\PycharmProjects\Newgen\CNN\Dataset\Classification'

# Define image size and batch size
img_size = (150, 150)
batch_size = 32

def save_classification_metrics_image(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = sorted(set(y_true))
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > (cm.max() / 2) else 'black')

    plt.tight_layout()

    plt.savefig(save_path + '_confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.axis('off')
    plt.table(cellText=[[f"{k}: {v}" for k, v in report[label].items()] for label in sorted(set(y_true))],
              colLabels=['Precision', 'Recall', 'F1-Score', 'Support'],
              rowLabels=sorted(set(y_true)),
              cellLoc='center',
              loc='center',
              bbox=[0, 0, 1, 1])
    plt.savefig(save_path + '_classification_report.png')
    plt.close()

# Create data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the dataset into 80% training and 20% validation
)
print("suyash -----------", os.listdir(train_data_dir))
# Generate training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
)

# Generate validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Adjusted for 10 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=15
)
# ...

# Save the model
dir_path = r"C:\Users\HP\PycharmProjects\Newgen\CNN\Programs"
# Generate a unique model name based on the current timestamp
model_save_path = os.path.join(dir_path, f'cnn_model.keras')
# Save the model
model.save(model_save_path)

# Evaluate the model on validation data with corrected steps
validation_eval = model.evaluate(validation_generator, steps=(validation_generator.samples // batch_size) + 1)

# Get predictions on validation data with corrected steps
validation_predictions = model.predict(validation_generator, steps=(validation_generator.samples // batch_size) + 1)
validation_labels = validation_generator.classes

# Convert predictions to class labels
predicted_classes = tf.argmax(validation_predictions, axis=1)

# Convert one-hot encoded labels to integers
true_classes = tf.argmax(tf.one_hot(validation_labels, depth=10), axis=1)

# Print accuracy
print(f"Validation Accuracy: {validation_eval[1]*100:.2f}%")
accuracy = r'C:\Users\HP\PycharmProjects\Newgen\CNN\Programs\accuracy.txt'
with open(accuracy, 'w') as file:
    file.write(f"accuracy : {validation_eval[1]*100:.2f}")

# Correct the path and use true_classes and predicted_classes
save_path = r'C:\Users\HP\PycharmProjects\Newgen\CNN\Programs\cnn'
save_classification_metrics_image(true_classes, predicted_classes, save_path)