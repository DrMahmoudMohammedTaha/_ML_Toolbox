import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from keras.optimizers import Adam

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize data
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 784)

# Define generator
generator = Sequential()
generator.add(Dense(256, input_dim=100))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh'))

# Define discriminator
discriminator = Sequential()
discriminator.add(Dense(512, input_dim=784))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile discriminator
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined network
discriminator.trainable = False
gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Training GAN
epochs = 100
batch_size = 64
for epoch in range(epochs):
    for _ in range(X_train.shape[0] // batch_size):
        # Sample noise and generate images
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_images = generator.predict(noise)
        
        # Train discriminator
        real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(gen_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
    # Print progress
    print(f'Epoch {epoch}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}')

# Generate and plot images
noise = np.random.normal(0, 1, (10, 100))
generated_images = generator.predict(noise)
for i in range(10):
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.show()
