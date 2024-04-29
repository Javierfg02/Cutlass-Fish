
import math
import tensorflow as tf 

def initialize_model(model, cfg: dict, src_padding_idx: int,
                     trg_padding_idx: int) -> None:
    """
    This initializes a model based on the provided config.

    All initializer configuration is part of the `model` section of the
    configuration file.
    For an example, see e.g. `https://github.com/joeynmt/joeynmt/
    blob/master/configs/iwslt_envi_xnmt.yaml#L47`

    The main initializer is set using the `initializer` key.
    Possible values are `xavier`, `uniform`, `normal` or `zeros`.
    (`xavier` is the default).

    When an initializer is set to `uniform`, then `init_weight` sets the
    range for the values (-init_weight, init_weight).

    When an initializer is set to `normal`, then `init_weight` sets the
    standard deviation for the weights (with mean 0).

    The word embedding initializer is set using `embed_initializer` and takes
    the same values. The default is `normal` with `embed_init_weight = 0.01`.

    Biases are initialized separately using `bias_initializer`.
    The default is `zeros`, but you can use the same initializers as
    the main initializer.

    :param model: model to initialize
    :param cfg: the model configuration
    :param src_padding_idx: index of source padding token
    :param trg_padding_idx: index of target padding token
    """

    # defaults: xavier, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = float(cfg.get("init_gain", 1.0))  # for xavier
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))

    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # for xavier

    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    # pylint: disable=unnecessary-lambda, no-else-return
    def _parse_init(s, scale, _gain):
        scale = float(scale)
        assert scale > 0., "incorrect init_weight"
        if s.lower() == "xavier":
            return tf.keras.initializers.GlorotUniform()
        elif s.lower() == "uniform":
            return tf.random_uniform_initializer(-scale,scale)
        elif s.lower() == "normal":
            return tf.random_normal_initializer(mean=0.0, stddev=scale)
        elif s.lower() == "zeros":
            return tf.zeros_initializer()
        else:
            raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)


    for name, p in model.named_parameters():

        if "embed" in name:
            if "bias" in name:
                bias_init_fn_(p)
            else:
                embed_init_fn_(p)

        elif "bias" in name:
            bias_init_fn_(p)


        # zero out paddings
        model.src_embed.lut.weight.data[src_padding_idx].zero_()