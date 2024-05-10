# coding: utf-8
"""
Module to implement training loss
"""

import tensorflow as tf

class RegLoss(tf.keras.Model):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = tf.keras.losses.MeanAbsoluteError()
        elif self.loss == "mse":
            self.criterion = tf.keras.losses.MeanSquaredError()

        else:
            #print("Loss not found - revert to default L1 loss")
            self.criterion = tf.keras.losses.MeanAbsoluteError()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def call(self, preds, targets):

       #loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds[:,:,:-1]
        #* loss_mask
        targets_masked = targets
        #* loss_mask

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

class XentLoss(tf.keras.Model):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # standard xent loss
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(ignore_index=self.pad_index,
                                    reduction='sum')

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        # targets: indices with batch*seq_len
        targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)

        return loss