import tensorflow as tf
from PIL import Image

# Load your pre-trained model
model = tf.keras.models.load_model('your_model_path.h5')  # Replace 'your_model_path.h5' with the actual path to your model file

# Load and resize the image
image_path = '/content/random.png'
image = Image.open(image_path)
resized_image = image.resize((512, 512))

# Convert the image to a numpy array and normalize
image_array = tf.keras.preprocessing.image.img_to_array(resized_image)
image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

# Add batch dimension and make prediction
prediction = model.predict(tf.expand_dims(image_array, axis=0))

# Display the prediction results
predicted_class = tf.argmax(prediction, axis=1)
print("Predicted class:", predicted_class.numpy()[0])

# If you have class labels, you can map the predicted class index to the corresponding label
# For example, if your class labels are stored in a list called class_names:
# predicted_label = class_names[predicted_class]
# print("Predicted label:", predicted_label)

# Optionally, show the resized image
resized_image.show()
