import numpy as np
import tensorflow as tf
import datetime
import os
import time
from tensorflow.python import debug as tf_debug

# Training Data
x_train = np.load("./saved_data/train/branch_arrays.npy")
mask_train = np.load("./saved_data/train/mask.npy")
y_train = np.load("./saved_data/train/padlabel.npy")
rmdoublemask_train = np.load("./saved_data/train/rmdoublemask.npy")

# Dev Data
x_dev = np.load("./saved_data/dev/branch_arrays.npy")
mask_dev = np.load("./saved_data/dev/mask.npy")
y_dev = np.load("./saved_data/dev/padlabel.npy")
rmdoublemask_dev = np.load("./saved_data/dev/rmdoublemask.npy")

# Result file
path_to_results_file = "./results"
path_to_saved_models = "./saved-models"

# Parameters
lstm_units = 50 # nur zu Testzwecken, um Verwirrungen mit mb_size zu verhindern -> Ursprungswert: 100
lstm_layers = 2
dense_units = 500
dense_layers = 2
num_epochs = 15
learn_rate = 0.001
mb_size = 100
l2reg = 0.0
rng_seed = 364

dropout_wrapper = True
prediction_layers = "tf.layers.dense AND all tf.layers.dropout"
los_seq_mask = False
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
        print("Loss via sequence mask: " + str(los_seq_mask), file=out)
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
    cross_entropy /= tf.reduce_sum(mask, 0)
    return tf.reduce_mean(cross_entropy)


def get_batches(x, y, mask, rmd, mb_size, shuffle):
    if shuffle:
        n_batches = len(x) // mb_size
        np.random.seed(rng_seed)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        for i in range(0, len(x) - mb_size + 1, mb_size):
            excerpt = indices[i:i+mb_size]
            yield x[excerpt], y[excerpt], mask[excerpt], rmd[excerpt]
    else:
        n_batches = len(x) // mb_size
        x, y = x[:n_batches * mb_size], y[:n_batches * mb_size]
        for i in range(0, len(x), mb_size):
            yield x[i:i + mb_size], y[i:i + mb_size], mask[i:i+mb_size], rmd[i:i+mb_size]
        # return x[0:mb_size], y[0:mb_size]


# Create default graph
tf.reset_default_graph()

# Declare variables
inputs = tf.placeholder(tf.float32, [mb_size, branch_length, tweet_length])
labels = tf.placeholder(tf.float32, [mb_size, branch_length, num_classes])
mask = tf.placeholder(tf.float32, [mb_size, branch_length])
rmdmask = tf.placeholder(tf.float32, [mb_size, branch_length])

# LSTM Cell
def lstm_cell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    if dropout_wrapper: # Semantischer Fehler true/false
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    else:
        return lstm_cell

# Several LSTM layers
cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
initial_state = cells.zero_state(mb_size, tf.float32)
print("Cells, InitialState")

# Unrolling the network
seq_length = length(inputs)
# outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state, sequence_length=seq_length) # sequence_length = branch_length -> branch_length funktioniert nicht
outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state)
# outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state)
# outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_train, dtype=tf.float32)
print("Outputs, FinalState")

# Predictions
if prediction_layers == "tf.layers.dense":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(inputs=hidden_1, units=num_classes, activation=tf.nn.relu)

elif prediction_layers == "test":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu, use_bias=True)
    predictions = tf.layers.dense(inputs=hidden_1, units=num_classes, activation=tf.nn.relu, use_bias=True)

elif prediction_layers == "tf.contrib.layers.fully_connected":
    predictions = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=tf.nn.relu)

elif prediction_layers == "tf.layers.dense AND all tf.layers.dropout":
    X_drop = tf.layers.dropout(inputs=outputs, rate=keep_prob, training=training)
    hidden_1 = tf.layers.dense(inputs=X_drop, units=dense_units, activation=tf.nn.relu, use_bias=True)
    hidden_1_drop = tf.layers.dropout(inputs=hidden_1, rate=keep_prob, training=training)
    hidden_2 = tf.layers.dense(inputs=hidden_1_drop, units=num_classes, activation=tf.nn.relu, use_bias=True)
    predictions = tf.layers.dropout(hidden_2, rate=keep_prob, training=training)

elif prediction_layers == "tf.layers.dense AND single tf.layers.dropout":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
    predictions = tf.layers.dense(inputs=hidden_1, units=num_classes, activation=tf.nn.relu)
else:
    predictions = get_predictions(outputs)
predictions = tf.Print(input_=predictions, data=[predictions], message="Predictions", summarize=100)
print("Predictions")

# Loss
if los_seq_mask:
    # mask = tf.sequence_mask(maxlen=mb_size, lengths=branch_length, dtype=tf.float32)
    loss = cost(predictions=predictions, labels=labels)
    loss *= mask
    loss *= rmdmask
    # loss = tf.Print(input_=loss, data=[loss], message="Loss", first_n=50)
else:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
    loss *= mask
    loss *= rmdmask
# loss = tf.Print(input_=loss, data=[loss], message="Loss", first_n=50)
print("Loss")

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
print("Optimizer")

preds = tf.round(predictions)
preds = tf.Print(input_=preds, data=[preds], message="Preds", summarize=100)
correct_pred = tf.equal(tf.cast(preds, tf.float32), labels) # argmax statt round
correct_pred = tf.Print(input_=correct_pred, data=[correct_pred], message="Correct Pred", summarize=100)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.Print(input_=accuracy, data=[accuracy], message="Accuracy", summarize=100)
print("Accuracy")

# Initialise variables
init = tf.global_variables_initializer()

# Training session
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    init.run()
    saver = tf.train.Saver()

    result_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".txt")
    results = os.path.join(path_to_results_file, result_file).replace(os.path.sep, '/')
    saved_model_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".ckpt")
    saved_model = os.path.join(path_to_saved_models, saved_model_file).replace(os.path.sep, '/')

    write_parameters(results)

    # for epoch in num_epochs:
    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        print(">>EPOCH: " + str(epoch + 1) + "/" + str(num_epochs) + " <<")
        with open(results, 'a') as out:
            print(">> EPOCH: " + str(epoch + 1) + " <<", file=out)
        for batch in get_batches(x_train, y_train, mask_train, rmdoublemask_train, mb_size, shuffle):
            x_batch, y_batch, mask_batch, rmdmask_batch = batch
            o, l, p = sess.run([optimizer, loss, predictions], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmdmask_batch}) # ausgeben und speichern
            mse = loss.eval(feed_dict={inputs: x_batch, labels:y_batch, mask: mask_batch, rmdmask: rmdmask_batch}) # ausgeben und speichern
            with open(results, 'a') as out:
                print("Accuracy for this batch: ", sess.run(accuracy, {inputs:x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmdmask_batch}), file=out)
            if (epoch + 1 % saving == 0 and epoch != 0):
                print("Model should save")
                with open(results, 'a') as out:
                    print("Model should save", file=out)
                save_path = saver.save(sess, saved_model, global_step=epoch)
                save_path = saver.save(sess, "models/pretrained_lstm.ckpt")
                with open(results, 'a') as out:
                    print("Saved to %s" % save_path)
                    print("Saved to %s" % save_path, file=out)
        stop_epoch_time = time.time()
        stop_epoch_time = stop_epoch_time - start_epoch_time
        with open(results, 'a') as out:
            print("Epoch time: " + str(stop_epoch_time) + "\n-------------------------------------------------------", file=out)
#
    print("Model should saveOuter")
    with open(results, 'a') as out:
        print("Model should saveOuter", file=out)
    save_path = saver.save(sess, saved_model, global_step=epoch)
    save_path = saver.save(sess, "models/pretrained_lstm.ckpt")
    with open(results, 'a') as out:
        print("OuterSaved to %s" % save_path)
        print("outerSaved to %s" % save_path, file=out)
    print("Training completed, prediction starts")
#
with open(results, 'a') as out:
    print("Training completed, prediction starts\n-------------------------------------------------------", file=out)

training = False

start_pred_time = time.time()

with tf.Session() as sess:
    saver = tf.train.Saver()
    # saver.restore(sess, saved_model)
    saver.restore(sess, tf.train.latest_checkpoint(path_to_saved_models))
    for iter in range(iterations):
        # for batch in get_batches(x_dev, y_dev, mb_size):
        for batch in get_batches(x_dev, y_dev, mask_dev, rmdoublemask_dev, mb_size, shuffle):
            x_batch, y_batch, mask_batch, rmdmask_batch = batch
            with open(results, 'a') as out:
                print("Accuracy for this batch: ", sess.run(accuracy, {inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmdmask_batch}), file=out)

stop_pred_time = time.time()
stop_pred_time = stop_pred_time - start_pred_time

with open(results, 'a') as  out:
    print("Prediction time: " + str(stop_pred_time), file=out)

print("Done")
