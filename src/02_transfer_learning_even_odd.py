import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import io # To log the summary also
import time


STAGE = "Transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def update_even_odd_labels(list_of_labels):

    for idx,label in enumerate(list_of_labels):
        even_condition = label%2 == 0
        list_of_labels[idx] = np.where(even_condition,1,0) #if number is even then 1 else 0

    return list_of_labels

#def main(config_path, params_path):

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    

    #get the data

    (X_train_full, y_train_full),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full/255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000] , X_train_full[5000:] 
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:] #These are classes

    y_train_bin,y_test_bin,y_valid_bin = update_even_odd_labels([y_train,y_test,y_valid])

    ##set the seeds

    seed = 2022 #or get it from config 
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## Log our model summary information in logs

    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn =lambda x:stream.write(f"{x}\n"))
            summary_str =stream.getvalue()
        return summary_str 

    ## Load the base model
    base_model_path = os.path.join("artifacts","models","base_model.h5")
    base_model =tf.keras.models.load_model(base_model_path) #Load old model
    logging.info(f"Loaded base model summary: \n{_log_model_summary(base_model)}")


    ##  define layers

    # LAYERS = [
    #       tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),#Flatten 28X28 matrix to 784 neurons
    #       tf.keras.layers.Dense(300, name="hiddenLayer1"),
    #       tf.keras.layers.LeakyReLU(), #Another way to use activation function
    #       tf.keras.layers.Dense(100, name="hiddenLayer2"),
    #       tf.keras.layers.LeakyReLU(),
    #       tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
    #       ]

    ## Freeze the weights
    for layer in base_model.layers[:-1]: #We dont have to make all layers freeze. Here we exclude the last layer.
        print(f"Trainable status of before {layer.name}:{layer.trainable}")
    # The above code will give output as all layers are trainable eg:Trainable status of inputLayer:True
    # Trainable status of hiddenLayer1:True
    # Trainable status of leaky_re_lu:True

    # Therefore we have to freeze this weight or in other words make trainable status as false.
    # Otherwise during the back propagation the weights will get updated in which layers are trainable
        layer.trainable = False
        print(f"Trainable status of after {layer.name}:{layer.trainable}")

    base_layer = base_model.layers[:-1]
    # # Define the model and compile it

    new_model = tf.keras.models.Sequential(base_layer) 
    new_model.add(
        tf.keras.layers.Dense(2,activation="softmax",name="output_layer")
    )
    logging.info(f"Loaded new Transfer learning model summary: \n{_log_model_summary(new_model)}")


    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    #OPTIMIZER = "SGD"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    ## Train the model

    EPOCHS = 10
    VALIDATION_SET = (X_valid, y_valid_bin) # y_valid_bin and y_train_bin for our use case

    # record start time
    start = time.time()

    history = new_model.fit(X_train, y_train_bin, epochs=EPOCHS, validation_data=VALIDATION_SET,verbose=2)

    end = time.time()

    print("The time of execution of training the transfer learning model is :",
    (end-start) * 10**3, "ms")

    logging.info(f"The time of execution of training the transfer learning model is :{(end-start) * 10**3} ms")

    ## Save the base model 

    model_dir_path = os.path.join("artifacts","models") #INside artifacts models directory is made

    model_file_path = os.path.join(model_dir_path,"even_odd_model.h5")

    new_model.save(model_file_path)

    logging.info(f"Base model is saved at {model_file_path}")

    logging.info(f"Evaluation metrics {new_model.evaluate(X_test,y_test_bin)}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        #main(config_path=parsed_args.config, params_path=parsed_args.params)
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e