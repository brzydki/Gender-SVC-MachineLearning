import cv2
import os
import numpy as np
from sklearn import svm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


male_images_dir = 'male_images'
female_images_dir = 'female_images'
output_dir = 'gender_classifier'
os.makedirs(output_dir, exist_ok=True)

def load_images_and_labels(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

male_images, male_labels = load_images_and_labels(male_images_dir, label=0) # man
female_images, female_labels = load_images_and_labels(female_images_dir, label=1) #women
images = np.array(male_images + female_images)
labels = np.array(male_labels + female_labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
joblib.dump(classifier, os.path.join(output_dir, 'gender_classifier.pkl'))

print('Complete.')