import numpy as np
import keras
import tensorflow as tf


xbbbNN_model_counter = 0
def construct_parallel_MLP(input, layers, name = None, regularizers = None, regularize_input = False):
    """ 
    Constructs the architecture for multiple, parallel running MultiLayer Perceptrons.

    Args:
        input (pandas.DataFrame or list(pandas.DataFrame)): Input from which the input shape is inferred. Should be an example of training data.
        layers (array-like): Array of integers specifying width of hidden layers. If a single array is given, a single MLP will be constructed. Otherwise a MLP is constructed for each list passed.
        name (str or array-like): Name of the MLPs created.

    """
    layers = np.array(layers)
    created_inputs = []
    created_layers = []
    if len(layers.shape) == 1:
        layers = np.array([layers])
        name = np.array([name])
        input = [input]
        regularizers = [regularizers]

    if regularizers is None:
        regularizers = [None] * len(layers)

    assert len(input) == len(layers)
    assert len(name) == len(layers)
    if name is None: name = [_get_default_name() for i in range(len(layers))]
    # for each set of names, inputs and layers:
    for n, i, parallel_layers, reg in zip(name, input, layers, regularizers):
        input_layer = keras.layers.Input(shape = (len(i.keys())), name = f"{n}_input")
        created_inputs.append(input_layer)
        if regularize_input:
            model = IsolatedActivation(name = "input_gate")(input_layer)
            model = keras.layers.ActivityRegularization(l1 = 1e-4)(model)
            model = keras.layers.Dense(parallel_layers[0], activation=tf.nn.leaky_relu, name = f"{n}_dense_0")(model)
        else:
            model = keras.layers.Dense(parallel_layers[0], activation=tf.nn.leaky_relu, name = f"{n}_dense_0")(input_layer)
        for i, layer_size in enumerate(parallel_layers[1:], 1):
            if (not reg is None) and (i in reg):
                model = keras.layers.Dense(layer_size, kernel_regularizer = reg[i], activation=tf.nn.leaky_relu, name = f"{n}_dense_{i}")(model)
            else:
                model = keras.layers.Dense(layer_size, activation=tf.nn.leaky_relu, name = f"{n}_dense_{i}")(model)
        model = keras.layers.Dense(1, activation=tf.nn.leaky_relu, name = f"{n}_dense_{i + 1}")(model)
        created_layers.append(model)
    if len(created_layers) > 1:
        model = keras.layers.Concatenate(name = f"{'-'.join(name)}_result_concat")(created_layers)
    else:
        model = created_layers[0]    
    return created_inputs, model

def construct_residual_MLP(input, layers, name = None):
    """
    Constructs a MLP containing residual nodes denoted by "r" in the layer specification.

    Args:
        input (pd.DataFrame): Input from which the input shape is inferred. Should be an example for training data.
        layers (array-like): Array of integers specifying width of hidden layers. If instead a "r" is passed, a residual layer for the previous layer is inserted.
        name (str): Name of the network.
    """
    if name is None: name = _get_default_name()
    input_layer = keras.layers.Input(shape = (len(input.keys())), name = f"{name}_input")
    model = keras.layers.Dense(layers[0], activation=tf.nn.leaky_relu, name = f"{n}_dense_0")(input_layer)
    last_size = layers[0]
    for i, layer_size in enumerate(layers[1:], 1):
        if layer_size == "r":
            residual_model = keras.layers.Dense(last_size, activation = tf.nn.leaky_relu, name = f"{n}_residual_{i}")(model)
            model = model + residual_model
        else:
            model = keras.layers.Dense(layer_size, activation=tf.nn.leaky_relu, name = f"{n}_dense_{i}")(model)
            last_size = layer_size
    model = keras.layers.Dense(1, activation=tf.nn.leaky_relu, name = f"{n}_dense_{i + 1}")(model)
    return input, model

def compile_model(input_layers, model, loss, metrics):
  model = tf.keras.Model(inputs=input_layers, outputs=model)
  optimizer = tf.keras.optimizers.Adam(0.001)
  metrics =metrics
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
  return model

def _get_default_name():
    global xbbbNN_model_counter
    name = f"model_{xbbbNN_model_counter}"
    xbbbNN_model_counter += 1
    return name


from keras.layers import Layer
class IsolatedActivation(Layer):
  def build(self, input_shape):
    self.alphas = self.add_weight('alpha', shape=[int(input_shape[-1]),],
                        initializer=tf.constant_initializer(1.0),
                            dtype=tf.float32)
    self.size = int(input_shape[-1])

  def call(self, input):
    linear = tf.multiply(input, self.alphas)
    return linear

