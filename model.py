import cv2
import joblib

model_path = 'gender_classifier/gender_classifier.pkl'
classifier = joblib.load(model_path)
def predict_gender(input_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if input_image is not None:
        input_image = input_image.reshape(1, -1)

        # Предсказание пола
        prediction = classifier.predict(input_image)[0]
        gender_mapping = {0: 'Man', 1: 'Women'}
        return gender_mapping[prediction]
    else:
        return 'Error.'

input_image_path = 'image3.jpg'  # Replace with the path to your image
predicted_gender = predict_gender(input_image_path)
print(f'Predicted gender: {predicted_gender}')


