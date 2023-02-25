import zipfile
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_prep_image(filename, img_shape=224, scale=True):
  """
    Reads in an image from filename, turns it into a tensor and reshapes into (224,224,3)

    Parameters:
      filename (str): filename of target image
      img_shape (int): size to resize target image to, default 224
      scale (bool): wheter to scale pixel values to range(0,1), default True
  """

  img = tf.io.read_file(filename)
  img = tf.image_decode_jpeg(img)
  img = tf.image.resize(img, [img_shape, img_shape])

  if scale:
    return img/255.
  else:
    return im


def unzip_data(filename):
  """
    Unzips filename into the current working directory.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def create_model(model_url, num_classes=1):
  """
    Takes a tensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in the output layer,
        should be equal to number of target classes, default 1.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature extractor
      layer and Dense outpu layer with num_classes output neurons.
  """

  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           name="feature_extractor_layer")

  model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
        ])

  return model

def plot_loss_curves(history):
  """
    Return separate loss curves for training and validation metrics.

    Args:
      History: TensorFlow History object.

    Returns:
      Plots of training/validation loss and accuracy metrics
  """  

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()
