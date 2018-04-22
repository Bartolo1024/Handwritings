import tensorflow as tf
from tensorflow.contrib.tensor_forest.client import eval_metrics


def _conv_layer(x, filters, kernel, padding='SAME'):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            kernel_size=kernel,
                            padding=padding)


def _max_pool_layer(x, pool_size, strides):
    return tf.layers.max_pooling2d(inputs=x,
                                   pool_size=pool_size,
                                   strides=strides)


def conv_model(input_layer_shape, labels, num_of_classes, mode):
    input_layer = tf.placeholder(input_layer_shape)

    conv = _conv_layer(input_layer,
                       32,
                       [7, 7],
                       'SAME')
    pool = _max_pool_layer(conv, [3, 3], 2)
    conv = _conv_layer(pool,
                       32,
                       [5, 5],
                       'SAME')
    pool = _max_pool_layer(conv, [3, 3], 2)
    conv = _conv_layer(pool,
                       32,
                       [7, 7],
                       'SAME')
    pool = _max_pool_layer(conv, [3, 3], 2)
    conv = _conv_layer(pool,
                       32,
                       [7, 7],
                       'SAME')
    pool = _max_pool_layer(conv, [3, 3], 2)
    conv = _conv_layer(pool,
                       32,
                       [7, 7],
                       'SAME')
    pool = _max_pool_layer(conv, [3, 3], 2)
    pool_flat = tf.layers.flatten(pool)
    dense = tf.layers.dense(inputs=pool_flat, units=20, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=num_of_classes)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(input=logits, name='softmax')
    }

    if mode is tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = loss_function(logits, labels)

    if mode is tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(loss)

    if mode is tf.estimator.ModeKeys.EVAL:
        eval_metrics = {"accurency": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)


def loss_function(logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)