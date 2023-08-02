import flwr as fl
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf
from pathlib import Path

GLOBAL_ROUNDS = 100  # 100

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 1 if rnd < 4 else 2
    return {"val_steps": val_steps}


base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

initial_parameters = fl.common.weights_to_parameters(model.get_weights())

# class SaveModelStrategy(fl.server.strategy.1FedYogi): #DONE
# class SaveModelStrategy(fl.server.strategy.FastAndSlow):
# class SaveModelStrategy(fl.server.strategy.1FaultTolerantFedAvg): DONE
# class SaveModelStrategy(fl.server.strategy.1FedFSv1):
# class SaveModelStrategy(fl.server.strategy.QFedAvg): # ERRO
# class SaveModelStrategy(fl.server.strategy.FedAdam):


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
    fraction_fit=0.2,
    fraction_eval=0.1,
    min_fit_clients=3,
    min_eval_clients=3,
    min_available_clients=3,
    on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
    initial_parameters=initial_parameters,
)

fl.server.start_server(server_address="[::]:8080",
                       config={"num_rounds": GLOBAL_ROUNDS},
                       certificates=(
                            Path("ca.crt").read_bytes(),
                            Path("server.pem").read_bytes(),
                            Path("server.key").read_bytes()),
                       strategy=strategy)

# fl.server.start_server(config={"num_rounds": 100})
