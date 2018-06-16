import numpy as np
import tensorflow as tf
import datetime
import os
import time
import ipdb
from tensorflow.python import debug as tf_debug

# Training Data
x_train = np.load("./saved_data/mod/train/branch_arrays.npy")
mask_train = np.load("./saved_data/mod/train/mask.npy")
y_train = np.load("./saved_data/mod/train/padlabel.npy")
rmdoublemask_train = np.load("./saved_data/mod/train/rmdoublemask.npy")
# bool_train = np.load("./saved_data/mod/train/bool.npy")

# Dev Data
x_dev = np.load("./saved_data/mod/dev/branch_arrays.npy")
mask_dev = np.load("./saved_data/mod/dev/mask.npy")
y_dev = np.load("./saved_data/mod/dev/padlabel.npy")
rmdoublemask_dev = np.load("./saved_data/mod/dev/rmdoublemask.npy")
# bool_dev = np.load("./saved_data/mod/dev/bool.npy")

# Result file
path_to_results_file = "./results"
path_to_saved_models = "./saved-models"

# Parameters
lstm_units = 100
lstm_layers = 2
dense_units = 500
dense_layers = 2
num_epochs = 30
learn_rate = 0.001
mb_size = 100
l2reg = 0.0
rng_seed = 364

dropout_wrapper = True
prediction_layers = "tf.layers.dense AND all tf.layers.dropout - softmax"
loss_calculations = "Categorical cross entropy"
shuffle = False
keep_prob = 0.5
training = True

branch_length = 25
tweet_length = 314
num_classes = 4

iterations = 5
saving = 5


# Helper functions
def write_parameters(results):
    with open(results, 'a+') as out:
        print("Loss über jeden Batch gemittelt", file=out)
        print("LSTM Units: " + str(lstm_units), file=out)
        print("LSTM Layers: " + str(lstm_layers), file=out)
        print("Dense Units: " + str(dense_units), file=out)
        print("Dense Layers: " + str(dense_layers), file=out)
        print("Num epochs: " + str(num_epochs), file=out)
        print("Learn rate: " + str(learn_rate), file=out)
        print("Batch size: " + str(mb_size), file=out)
        print("L2 Reg: " + str(l2reg), file=out)
        print("RNG Seed: " + str(rng_seed), file=out)
        print("Dropout Wrapper: " + str(dropout_wrapper), file=out)
        print("Several dense layers: " + str(prediction_layers), file=out)
        print("Loss via sequence mask: " + str(loss_calculations), file=out)
        print("Shuffle: " + str(shuffle), file=out)
        print("Keep prob: " + str(keep_prob), file=out)
        print("Iterations: " + str(iterations), file=out)
        print("Savings iterations: " + str(saving), file=out)
        print("-------------------------------------------------------", file=out)


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def weight_and_bias(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)


def get_predictions(output):
    weight, bias = weight_and_bias(dense_units, num_classes)
    output = tf.reshape(output, [-1, dense_units])
    prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
    prediction = tf.reshape(prediction, [-1, branch_length])
    return prediction


def cost(predictions, labels):
    cross_entropy = labels * tf.log(predictions)
    cross_entropy = -tf.reduce_sum(cross_entropy, 1)
    # mask = tf.sign(tf.reduce_max(tf.abs(labels), 1))
    # cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, 0)
    # cross_entropy /= tf.reduce_sum(mask, 0)
    return tf.reduce_mean(cross_entropy)


# def get_batches(x, y, mask, rmd, bool, mb_size, shuffle):
def get_batches(x, y, mask, rmd, mb_size, shuffle):
    if shuffle:
        n_batches = len(x) // mb_size
        np.random.seed(rng_seed)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        for i in range(0, len(x) - mb_size + 1, mb_size):
            excerpt = indices[i:i+mb_size]
            # yield x[excerpt], y[excerpt], mask[excerpt], rmd[excerpt], bool[excerpt]
            yield x[excerpt], y[excerpt], mask[excerpt], rmd[excerpt]
    else:
        n_batches = len(x) // mb_size
        x, y = x[:n_batches * mb_size], y[:n_batches * mb_size]
        for i in range(0, len(x), mb_size):
            # yield x[i:i + mb_size], y[i:i + mb_size], mask[i:i+mb_size], rmd[i:i+mb_size], bool[i:i+mb_size]
            yield x[i:i + mb_size], y[i:i + mb_size], mask[i:i + mb_size], rmd[i:i + mb_size]
        # return x[0:mb_size], y[0:mb_size]


# Create default graph
tf.reset_default_graph()

# Declare variables
inputs = tf.placeholder(tf.float32, [mb_size, branch_length, tweet_length])
labels = tf.placeholder(tf.float32, [mb_size, branch_length, num_classes])
mask = tf.placeholder(tf.float32, [mb_size, branch_length])
rmdmask = tf.placeholder(tf.float32, [mb_size, branch_length])
# bool = tf.placeholder(tf.bool, [mb_size])

# LSTM Cell
def lstm_cell():
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=lstm_units, activation=tf.nn.sigmoid)
    if dropout_wrapper:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    else:
        return lstm_cell

# Several LSTM layers
cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
initial_state = cells.zero_state(mb_size, tf.float32)
# initial_state = tf.Print(input_=initial_state, data=[initial_state], message="Initial State", summarize=10)
print("Cells, InitialState")

# Unrolling the network
seq_length = length(inputs)
# outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state, sequence_length=seq_length) # sequence_length = branch_length -> branch_length funktioniert nicht
outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state)
# outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state)
# outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_train, dtype=tf.float32)
# outputs = tf.Print(input_=outputs, data=[outputs], message="Outputs", summarize=10)
print("Outputs, FinalState")

# Scores
if prediction_layers == "tf.layers.dense":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(inputs=hidden_1, units=num_classes, activation=tf.nn.relu)

elif prediction_layers == "test":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu, use_bias=True)
    scores = tf.layers.dense(inputs=hidden_1, units=num_classes, activation=tf.nn.relu, use_bias=True)

elif prediction_layers == "tf.contrib.layers.fully_connected":
    scores = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=tf.nn.relu)

elif prediction_layers == "tf.layers.dense AND all tf.layers.dropout":
    lstm_output_drop = tf.layers.dropout(inputs=outputs, rate=keep_prob, training=training)
    hidden_1 = tf.layers.dense(inputs=lstm_output_drop, units=dense_units, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_1_drop = tf.layers.dropout(inputs=hidden_1, rate=keep_prob, training=training)
    hidden_2 = tf.layers.dense(inputs=hidden_1_drop, units=num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, use_bias=True, trainable=training)
    scores = tf.layers.dropout(hidden_2, rate=keep_prob, training=training)

elif prediction_layers == "tf.layers.dense AND all tf.layers.dropout - softmax":
    # lstm_output_drop = tf.layers.dropout(inputs=outputs, rate=keep_prob, training=training)
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_1_drop = tf.layers.dropout(inputs=hidden_1, rate=keep_prob, training=training)
    hidden_2 = tf.layers.dense(inputs=hidden_1_drop, units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_2_drop = tf.layers.dropout(hidden_2, rate=keep_prob, training=training)
    scores = tf.layers.dense(inputs=hidden_2_drop, units=num_classes, activation=tf.nn.softmax, use_bias=True, trainable=training)
    # scores = tf.nn.softmax(hidden_2_drop)

elif prediction_layers == "tf.layers.dense AND single tf.layers.dropout":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
    hidden_1_drop = tf.layers.dropout(inputs=hidden_1, rate=keep_prob, training=training)
    scores = tf.layers.dense(inputs=hidden_1_drop, units=num_classes, activation=tf.nn.relu)

else:
    # scores = get_predictions(outputs)
    print(prediction_layers)
scores = tf.Print(input_=scores, data=[scores], message="Scores", summarize=100)
# weights = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden_2.name)[0] + '/kernel:0')
# print(weights)
print("Scores")

# Loss
if loss_calculations == "Own cost calculation":
    # mask = tf.sequence_mask(maxlen=mb_size, lengths=branch_length, dtype=tf.float32)
    loss = cost(predictions=scores, labels=labels)
    loss *= mask
    loss *= rmdmask
    # loss = tf.Print(input_=loss, data=[loss], message="Loss", first_n=50)
elif loss_calculations == "Softmax cross entropy with logits":
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores)
    loss *= mask
    loss *= rmdmask
elif loss_calculations == "Softmax cross entropy":
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=scores)
    loss *= mask
    loss *= rmdmask
elif loss_calculations == "Cross entropy":
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=scores, pos_weight=1)
    loss *= mask
    loss *= rmdmask
elif loss_calculations == "Categorical cross entropy":
    loss = tf.contrib.keras.backend.categorical_crossentropy(output=scores, target=labels)
    loss *= mask
    loss *= rmdmask # average loss pro branch (doppelte tweets und Elemente kürzerer Tweets ausschließen mit mask ausschließen); average pro branch
    # if tf.reduce_sum(rmdmask, 1) != 0:
    # loss = tf.Print(input_=loss, data=[loss], message="Loss Prev", summarize=100)
    # loss = tf.where(bool, (tf.reduce_sum(loss, 1) / tf.reduce_sum(rmdmask, 1)), (tf.reduce_sum(loss, 1)))
    # loss = tf.reduce_mean(loss) # Noch überlegen
else:
    print(loss_calculations)
# ipdb.set_trace()
loss = tf.Print(input_=loss, data=[loss], message="Loss Prev", summarize=100)
# loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(mask, 1)
loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(rmdmask, 1)
loss = tf.reduce_mean(loss)
loss = tf.Print(input_=loss, data=[loss], message="Loss Post", summarize=100)
# ipdb.set_trace()
# tf.summary.scalar('Loss', loss)
print("Loss")

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
print("Optimizer")

# Evaluation
predictions = tf.cast(tf.argmax(scores, 2), tf.float32)
predictions *= mask
predictions *= rmdmask
predictions = tf.Print(input_=predictions, data=[predictions], message="Predictions", summarize=100)
# correct_pred = tf.equal(tf.cast(predictions, tf.float32), labels)
# masked_labels = tf.cast(tf.argmax(labels, 2), tf.float32)
# masked_labels *= mask
# masked_labels *= rmdmask
# correct_pred = tf.cast(tf.equal(predictions, masked_labels), tf.float32)
correct_pred = tf.cast(tf.equal(predictions, tf.cast(tf.argmax(labels, 2), tf.float32)), tf.float32)
correct_pred *= mask
correct_pred *= rmdmask
correct_pred = tf.Print(input_=correct_pred, data=[correct_pred], message="Correct Pred", summarize=100)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.reduce_sum(correct_pred, 1) / tf.reduce_sum(rmdmask, 1)
accuracy = tf.reduce_mean(accuracy, 0)
# accuracy = tf.reduce_sum(correct_pred, 0) / (tf.reduce_sum(mask, 0) - (1 - tf.reduce_sum(rmdmask, 0)))
# tf.summary.scalar('Accuracy', accuracy)
# ipdb.set_trace()
accuracy = tf.Print(input_=accuracy, data=[accuracy], message="Accuracy", summarize=100)
print("Accuracy")

# Prepare files for results, weights+biases and saved models
result_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".txt")
results = os.path.join(path_to_results_file, result_file).replace(os.path.sep, '/')
saved_model_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".ckpt")
save_path = os.path.join(path_to_saved_models, saved_model_file).replace(os.path.sep, '/')
weights_bias_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-weights-bias.txt")
weights_bias = os.path.join(path_to_results_file, weights_bias_file).replace(os.path.sep, '/')

write_parameters(results)

# merged = tf.summary.merge_all()

# Initialise variables
init = tf.global_variables_initializer()
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

# Training session
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_infs_or_nans", tf_debug.has_inf_or_nan)
    init.run()
    saver = tf.train.Saver()

    # Visualization
    # logdir_train = "./tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-train")
    # writer_train = tf.summary.FileWriter(logdir_train, sess.graph)
    # logdir_dev = "./tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-dev")
    # writer_dev = tf.summary.FileWriter(logdir_dev, sess.graph)

    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<")
        with open(results, 'a') as out:
            print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<", file=out)
        with open(weights_bias, 'a') as out:
            print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<", file=out)
        # for batch in get_batches(x=x_train, y=y_train, mask=mask_train, rmd=rmdoublemask_train, bool=bool_train, mb_size=mb_size, shuffle=shuffle):
        for batch in get_batches(x=x_train, y=y_train, mask=mask_train, rmd=rmdoublemask_train, mb_size=mb_size, shuffle=shuffle):
            # x_batch, y_batch, mask_batch, rmd_batch, bool_batch = batch
            x_batch, y_batch, mask_batch, rmd_batch = batch
            # sess.run(fetches=optimizer, feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch, bool: bool_batch})
            sess.run(fetches=optimizer,
                     feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k, v in zip(variable_names, values):
                with open(weights_bias, 'a') as out:
                    print(k, v, file=out)
            # accuracy_run, loss_run = sess.run(fetches=[accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch, bool: bool_batch})
            accuracy_run, loss_run = sess.run(fetches=[accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # accuracy_run, loss_run, merged_run = sess.run(fetches=[accuracy, loss, merged], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch, bool: bool_batch})
            loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # writer_train.add_summary(merged_run, epoch)
            with open(results, 'a') as out:
                print("Loss (Run) for this batch: ", np.average(loss_run), file=out)
                print("Loss (Eval) for this batch: ", np.average(loss_eval), file=out)
                print("Accuracy for this batch: ", accuracy_run, file=out)
        print("Save model")
        with open(results, 'a') as out:
            print("Save model", file=out)
        saved = saver.save(sess=sess, save_path=save_path, global_step=epoch)
        # Evaluation on dev data
        training = False
        print("Start evaluation on dev data")
        with open(results, 'a') as out:
            print("-----------------------\nStart evaluation on dev data", file=out)
        with open(weights_bias, 'a') as out:
            print("-----------------------\nStart evaluation on dev data", file=out)
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(path_to_saved_models))
        # for batch in get_batches(x=x_dev, y=y_dev, mask=mask_dev, rmd=rmdoublemask_dev, bool=bool_dev, mb_size=mb_size, shuffle=shuffle):
        for batch in get_batches(x=x_dev, y=y_dev, mask=mask_dev, rmd=rmdoublemask_dev, mb_size=mb_size, shuffle=shuffle):
            # x_batch, y_batch, mask_batch, rmd_batch, bool_batch = batch
            x_batch, y_batch, mask_batch, rmd_batch = batch
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k, v in zip(variable_names, values):
                with open(weights_bias, 'a') as out:
                    print(k, v, file=out)
            # accuracy_run, loss_run = sess.run([accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch, bool: bool_batch})
            accuracy_run, loss_run = sess.run([accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # accuracy_run, optimizer_run, loss_run, merged_run = sess.run([accuracy, optimizer, loss, merged], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch, bool: bool_batch})
            loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            # writer_dev.add_summary(merged_run, epoch)
            with open(results, 'a') as out:
                print("Loss (Run) for this batch: ", np.average(loss_run), file=out)
                print("Loss (Eval) for this batch: ", np.average(loss_eval), file=out)
                print("Accuracy for this batch: ", accuracy_run, file=out)
        training = True
        stop_epoch_time = time.time()
        stop_epoch_time = stop_epoch_time - start_epoch_time
        with open(results, 'a') as out:
            print("Epoch time: " + str(stop_epoch_time) + "\n------------------------------------------------------", file=out)
        with open(weights_bias, 'a') as out:
            print("Epoch time: " + str(stop_epoch_time) + "\n------------------------------------------------------", file=out)
    print("Done")
