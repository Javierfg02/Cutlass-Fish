import tensorflow as tf
# from tensorflow.keras.optimizers.schedules import PolynomialDecay, \
#     ExponentialDecay
# from tensorflow import Optimizer
from helpers import ConfigurationError

def build_gradient_clipper(config):
    """
    Define the function for gradient clipping as specified in configuration.
    If not specified, returns None.

    Current options:
        - "clip_grad_val": clip the gradients if they exceed this value,
            see `tf.clip_by_value`
        - "clip_grad_norm": clip the gradients if their norm exceeds this value,
            see `tf.clip_by_norm`

    :param config: dictionary with training configurations
    :return: clipping function (in-place) or None if no gradient clipping
    """
    clip_grad_fun = None
    if "clip_grad_val" in config:
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda grads: \
            [tf.clip_by_value(g, -clip_value, clip_value) for g in grads]
    elif "clip_grad_norm" in config:
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda grads: \
            [tf.clip_by_norm(g, max_norm) for g in grads]

    if "clip_grad_val" in config and "clip_grad_norm" in config:
        raise ConfigurationError(
            "You can only specify either clip_grad_val or clip_grad_norm.")

    return clip_grad_fun


# def build_optimizer(config, parameters):
def build_optimizer(config):
    """
    Create an optimizer for the given parameters as specified in config.
    Except for the weight decay and initial learning rate,
    default optimizer settings are used.

    Currently supported configuration settings for "optimizer":
        - "sgd" (default): see `tf.keras.optimizers.SGD`
        - "adam": see `tf.keras.optimizers.Adam`
        - "adagrad": see `tf.keras.optimizers.Adagrad`
        - "adadelta": see `tf.keras.optimizers.Adadelta`
        - "rmsprop": see `tf.keras.optimizers.RMSprop`

    The initial learning rate is set according to "learning_rate" in the config.
    The weight decay is set according to "weight_decay" in the config.
    If they are not specified, the initial learning rate is set to 3.0e-4, the
    weight decay to 0.

    :param config: configuration dictionary
    :return: optimizer
    """
    optimizer_name = config.get("optimizer", "sgd").lower()
    learning_rate = config.get("learning_rate", 3.0e-4)
    weight_decay = config.get("weight_decay", 0)

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-07)
    elif optimizer_name == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate,
                                                initial_accumulator_value=0.1,
                                                epsilon=1e-07)
    elif optimizer_name == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                  rho=0.95, epsilon=1e-07)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                rho=0.9, momentum=0.0,
                                                epsilon=1e-07)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=0.0, nesterov=False)
    else:
        raise ConfigurationError("Invalid optimizer. Valid options: 'adam', "
                                 "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")

    return optimizer

# def build_scheduler(config, optimizer, scheduler_mode, hidden_size=0):
#     """
#     Create a learning rate scheduler if specified in config and
#     determine when a scheduler step should be executed.

#     Current options:
#         - "plateau": not directly supported in TensorFlow, but can be
#             implemented using callbacks
#         - "decaying": see `tf.keras.optimizers.schedules.PolynomialDecay`
#         - "exponential": see `tf.keras.optimizers.schedules.ExponentialDecay`
#         - "noam": not directly supported in TensorFlow

#     If no scheduler is specified, returns (None, None) which will result in
#     a constant learning rate.

#     :param config: training configuration
#     :param optimizer: optimizer for the scheduler, determines the set of
#         parameters which the scheduler sets the learning rate for
#     :param scheduler_mode: "min" or "max", depending on whether the validation
#         score should be minimized or maximized.
#         Only relevant for "plateau".
#     :param hidden_size: encoder hidden size (required for NoamScheduler)
#     :return:
#         - scheduler: scheduler object,
#         - scheduler_step_at: either "validation" or "epoch"
#     """
#     scheduler, scheduler_step_at = None, None
#     if "scheduling" in config and \
#             config["scheduling"]:
#         if config["scheduling"].lower() == "plateau":
#             # Plateau not directly supported in TensorFlow, can be
#             # implemented using callbacks
#             # pass
#             # TODO: not in tensorflow
#             scheduler = ReduceLROnPlateau(
#                 optimizer=optimizer,
#                 mode=scheduler_mode,
#                 verbose=False,
#                 threshold_mode='abs',
#                 threshold=1e-8,
#                 factor=config.get("decrease_factor", 0.1),
#                 patience=config.get("patience", 10))
#             # scheduler step is executed after every validation
#             scheduler_step_at = "validation"
#         elif config["scheduling"].lower() == "decaying":
#             # decay_steps = config.get("decay_steps", 1000)
#             # end_learning_rate = config.get("end_learning_rate", 0.0001)
#             # decay_power = config.get("decay_power", 1.0)
#             # scheduler = PolynomialDecay(initial_learning_rate=optimizer.learning_rate,
#             #                             decay_steps=decay_steps,
#             #                             end_learning_rate=end_learning_rate,
#             #                             power=decay_power)
#             # scheduler_step_at = "step"
#             scheduler = StepLR(
#                 optimizer=optimizer,
#                 step_size=config.get("decaying_step_size", 1))
#             # scheduler step is executed after every epoch
#             scheduler_step_at = "epoch"
#         elif config["scheduling"].lower() == "exponential":
#             scheduler = ExponentialLR(
#                 optimizer=optimizer,
#                 gamma=config.get("decrease_factor", 0.99))
#             # scheduler step is executed after every epoch
#             scheduler_step_at = "epoch"
#         elif config["scheduling"].lower() == "noam":
#             factor = config.get("learning_rate_factor", 1)
#             warmup = config.get("learning_rate_warmup", 4000)
#             scheduler = NoamScheduler(hidden_size=hidden_size, factor=factor,
#                                       warmup=warmup, optimizer=optimizer)

#             scheduler_step_at = "step"
#     return scheduler, scheduler_step_at


# class NoamScheduler:
#     """
#     The Noam learning rate scheduler used in "Attention is all you need"
#     See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
#     """

#     def __init__(self, hidden_size: int, optimizer: Optimizer,
#                  factor: float = 1, warmup: int = 4000):
#         """
#         Warm-up, followed by learning rate decay.

#         :param hidden_size:
#         :param optimizer:
#         :param factor: decay factor
#         :param warmup: number of warmup steps
#         """
#         self.optimizer = optimizer
#         self.step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.hidden_size = hidden_size
#         self.rate = 0

#     def step(self):
#         """Update parameters and rate"""
#         self.step += 1
#         rate = self._compute_rate()
#         # self.optimizer.learning_rate = rate
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self.rate = rate

#     def _compute_rate(self):
#         """Implement `lrate` above"""
#         step = self.step
#         return self.factor * \
#             (self.hidden_size ** (-0.5) *
#                 min(step ** (-0.5), step * self.warmup ** (-1.5)))

#     def state_dict(self):
#         return None