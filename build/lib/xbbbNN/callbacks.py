import tensorflow as tf

class ClippedWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, min_label, max_label, *args, **kwargs):
        self.min_label = min_label
        self.max_label = max_label
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        prediction = tf.clip_by_value(y_pred[:,0], self.min_label, self.max_label)
        error = (prediction - y_true[:, 0]) 
        return tf.reduce_sum(tf.square(error) * y_true[:, 1])/tf.reduce_sum(y_true[:,1])


class FlaggedMAE(tf.keras.metrics.Metric):
    def __init__(self, min_label, max_label, name = None, flag_idx = 2, positive = True, **kwargs):
        if name is None: name = f"flagged_MAE_{'' if positive else 'not'}_{flag_idx}"
        super().__init__(name = name, **kwargs)
        self.min_label = min_label
        self.max_label = max_label
        self.flag_idx = flag_idx
        self.positive = positive
        self._result = self.add_weight(name=f"{self.name}_result", initializer='zeros')
        self._count = self.add_weight(name=f"{self.name}_count", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        if not (sample_weight is None): raise NotImplementedError
        if self.positive:
            flagged_int = y_true[:, self.flag_idx]
        else:
            flagged_int = 1.0 - y_true[:, self.flag_idx]
        
        flagged_sum = tf.reduce_sum(flagged_int)
        if flagged_sum == 0.0:
            result = 0.0
        else:
            prediction = tf.clip_by_value(y_pred[:,0], self.min_label, self.max_label)
            error = y_true[:, 0] - prediction
            masked_error = tf.boolean_mask(error, tf.cast(flagged_int, bool))
            result = tf.reduce_sum(tf.abs(masked_error))
        self._result.assign_add(result)
        self._count.assign_add(flagged_sum)
    
    def result(self):
        return self._result/self._count

class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def __init__(self, patience = 50, min_delta = 1e-4 ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_loss = 1e20
        self.patience = patience
        self.patience_cnt = 0
        self.min_delta = min_delta

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('val_loss')
        if self.last_loss - loss < self.min_delta:
            self.patience_cnt += 1
        else:
            self.patience_cnt = 0
            self.last_loss = loss
        if self.patience_cnt > self.patience:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr * 0.5
            #print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
            self.model.optimizer.lr.assign(new_lr)
            self.patience_cnt = 0
            self.last_loss = 1e20
        logs['lr'] = self.model.optimizer.lr.read_value()
        logs["lr_cnt"] = self.patience_cnt

class ReadInputGateCB(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs={}):
        for layer in self.model.layers:
            if layer.name == "input_gate":
                print(layer.weights)