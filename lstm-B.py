import numpy as np
import tensorflow as tf
import datetime
import os
import time
import utils
import ipdb
from tensorflow.python import debug as tf_debug

# Training Data
x_train = np.load("./saved_data/B/mod/train/branch_arrays.npy")
mask_train = np.load("./saved_data/B/mod/train/mask.npy")
y_train = np.load("./saved_data/B/mod/train/padlabel.npy")
rmdoublemask_train = np.load("./saved_data/B/mod/train/rmdoublemask.npy")

# Dev Data
x_dev = np.load("./saved_data/B/mod/dev/branch_arrays.npy")
mask_dev = np.load("./saved_data/B/mod/dev/mask.npy")
y_dev = np.load("./saved_data/B/mod/dev/padlabel.npy")
rmdoublemask_dev = np.load("./saved_data/B/mod/dev/rmdoublemask.npy")

# Result file
# path_to_results_file = "./results/B"
# path_to_saved_models = "./saved-models/B"
date_folder = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
results_folder = "D:/Zwischenablage/results/B"
saved_models_folder = "D:/Zwischenablage/saved-models/B"
path_to_results_file = os.path.join(results_folder, date_folder).replace(os.path.sep, "/")
path_to_saved_models = os.path.join(saved_models_folder, date_folder).replace(os.path.sep, "/")
os.makedirs(path_to_results_file)
os.makedirs(path_to_saved_models)

# Parameters
lstm_units = 100
lstm_layers = 2
dense_units = 500
dense_layers = 2
num_epochs = 100
learn_rate = 0.001
mb_size = 10
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
np_print_threshold = False

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
        print("Loss via sequence mask: " + str(loss_calculations), file=out)
        print("Shuffle: " + str(shuffle), file=out)
        print("Keep prob: " + str(keep_prob), file=out)
        print("Iterations: " + str(iterations), file=out)
        print("Savings iterations: " + str(saving), file=out)
        print("-------------------------------------------------------", file=out)

# Create default graph
tf.reset_default_graph()

# Declare variables
inputs = tf.placeholder(tf.float32, [mb_size, branch_length, tweet_length])
labels = tf.placeholder(tf.float32, [mb_size, branch_length, num_classes])
mask = tf.placeholder(tf.float32, [mb_size, branch_length])
rmdmask = tf.placeholder(tf.float32, [mb_size, branch_length])

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
print("Cells, InitialState")

# Unrolling the network
outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, initial_state=initial_state, sequence_length=tf.reduce_sum(mask, 1))
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
    # hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_1 = tf.layers.dense(inputs=tf.reshape(outputs, [-1, lstm_units]), units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_1_drop = tf.layers.dropout(inputs=hidden_1, rate=keep_prob, training=training)
    hidden_2 = tf.layers.dense(inputs=hidden_1_drop, units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_2_drop = tf.layers.dropout(hidden_2, rate=keep_prob, training=training)
    scores = tf.layers.dense(inputs=hidden_2_drop, units=num_classes, activation=tf.nn.softmax, use_bias=True, trainable=training)
    scores = tf.reshape(scores, [mb_size, branch_length, num_classes])
elif prediction_layers == "tf.layers.dense AND single tf.layers.dropout - softmax":
    hidden_1 = tf.layers.dense(inputs=outputs, units=dense_units, activation=tf.nn.relu)
    hidden_1_norm = tf.layers.batch_normalization(inputs=hidden_1, training=training)
    hidden_2 = tf.layers.dense(inputs=hidden_1, units=dense_units, activation=tf.nn.relu, use_bias=True, trainable=training)
    hidden_2_norm = tf.layers.batch_normalization(inputs=hidden_2, training=training)
    hidden_2_norm_drop = tf.layers.dropout(inputs=hidden_2_norm, rate=keep_prob, training=training)
    scores = tf.layers.dense(inputs=hidden_2_norm_drop, units=num_classes, activation=tf.nn.softmax, use_bias=True, trainable=training)

else:
    print(prediction_layers)
print("Scores")

# Loss
if loss_calculations == "Softmax cross entropy with logits":
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores)
elif loss_calculations == "Softmax cross entropy":
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=scores)
elif loss_calculations == "Cross entropy":
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=scores, pos_weight=1)
elif loss_calculations == "Categorical cross entropy":
    loss = tf.contrib.keras.backend.categorical_crossentropy(output=scores, target=labels)
else:
    print(loss_calculations)
loss *= mask
# loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(rmdmask, 1)
loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(mask, 1)
loss = tf.reduce_mean(loss)
# ipdb.set_trace()
print("Loss")

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
print("Optimizer")

# Evaluation
predictions = tf.cast(tf.argmax(scores, 2), tf.float32)
correct_pred = tf.cast(tf.equal(predictions, tf.cast(tf.argmax(labels, 2), tf.float32)), tf.float32)
correct_pred *= mask
# correct_pred *= rmdmask
# accuracy = tf.reduce_sum(correct_pred, 1) / tf.reduce_sum(rmdmask, 1)
accuracy = tf.reduce_sum(correct_pred, 1) / tf.reduce_sum(mask, 1)
accuracy = tf.reduce_mean(accuracy, 0)
print("Accuracy")


saved_model_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".ckpt")
save_path = os.path.join(path_to_saved_models, saved_model_file).replace(os.path.sep, '/')
accuracy_dev_batch_file = os.path.join(path_to_results_file, "accuracy_dev_batch.txt").replace(os.path.sep, '/')
accuracy_dev_epoch_file = os.path.join(path_to_results_file, "accuracy_dev_epoch.txt").replace(os.path.sep, '/')
accuracy_train_batch_file = os.path.join(path_to_results_file, "accuracy_train_batch.txt").replace(os.path.sep, '/')
accuracy_train_epoch_file = os.path.join(path_to_results_file, "accuracy_train_epoch.txt").replace(os.path.sep, '/')
loss_dev_batch_file = os.path.join(path_to_results_file, "loss_dev_batch.txt").replace(os.path.sep, '/')
loss_dev_epoch_file = os.path.join(path_to_results_file, "loss_dev_epoch.txt").replace(os.path.sep, '/')
loss_train_batch_file = os.path.join(path_to_results_file, "loss_train_batch.txt").replace(os.path.sep, '/')
loss_train_epoch_file = os.path.join(path_to_results_file, "loss_train_epoch.txt").replace(os.path.sep, '/')
predictions_dev_file = os.path.join(path_to_results_file, "predictions_dev.txt").replace(os.path.sep, '/')
predictions_train_file = os.path.join(path_to_results_file, "predictions_train.txt").replace(os.path.sep, '/')
parameters_file = os.path.join(path_to_results_file, "parameters.txt").replace(os.path.sep, '/')
weights_bias_file = os.path.join(path_to_results_file, "weights_bias.txt").replace(os.path.sep, '/')
predictions_folder_train = os.path.join(path_to_results_file, "predictions_train").replace(os.path.sep, '/')
predictions_folder_dev = os.path.join(path_to_results_file, "predictions_dev").replace(os.path.sep, '/')
os.makedirs(predictions_folder_train)
os.makedirs(predictions_folder_dev)

write_parameters(parameters_file)

# Initialise variables
init = tf.global_variables_initializer()
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

# Training session
with tf.Session() as sess:
    init.run()
    saver = tf.train.Saver()

    for epoch in range(num_epochs):
        accuracy_train = 0
        loss_train = 0
        count_train = 0
        accuracy_dev = 0
        loss_dev = 0
        count_dev = 0
        start_epoch_time = time.time()
        print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
        for batch in utils.get_batches(x=x_train, y=y_train, mask=mask_train, rmd=rmdoublemask_train, mb_size=mb_size, shuffle=shuffle, rng_seed=rng_seed):
            x_batch, y_batch, mask_batch, rmd_batch = batch
            sess.run(fetches=optimizer, feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k, v in zip(variable_names, values):
                with open(weights_bias_file, 'a') as out:
                    print(epoch, ":", k, v, file=out)
            accuracy_run, loss_run, predictions_run = sess.run(fetches=[accuracy, loss, predictions], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            accuracy_train += accuracy_run
            loss_train += loss_run
            count_train += 1
            if np_print_threshold:
                np.set_printoptions(threshold=np.nan)
            with open(accuracy_train_batch_file, 'a') as out:
                print(epoch, ";", count_train, ":", accuracy_run, file=out)
            with open(loss_train_batch_file, 'a') as out:
                print(epoch, ";", count_train, ":", loss_run, file=out)
                # print(loss_run, file=out)
            with open(predictions_train_file, 'a') as out:
                print(epoch, ";", count_train, ":", predictions_run, file=out)
            np.save(os.path.join(predictions_folder_train, str(epoch) + "-" + str(count_train)), predictions_run)
        print("Save model")
        with open(accuracy_train_epoch_file, 'a') as out:
            print(epoch, ":", (accuracy_train / count_train), file=out)
        with open(loss_train_epoch_file, 'a') as out:
            print(epoch, ":", (loss_train / count_train), file=out)
            # print((loss_train / count_train), file=out)
        saved = saver.save(sess=sess, save_path=save_path, global_step=epoch)

        # Evaluation on dev data
        training = False
        print("Start evaluation on dev data")
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(path_to_saved_models))
        for batch in utils.get_batches(x=x_dev, y=y_dev, mask=mask_dev, rmd=rmdoublemask_dev, mb_size=mb_size, shuffle=shuffle, rng_seed=rng_seed):
            x_batch, y_batch, mask_batch, rmd_batch = batch
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            # for k, v in zip(variable_names, values):
            #     with open(weights_bias_file, 'a') as out:
            #         print(k, v, file=out)
            accuracy_run, loss_run, predictions_run = sess.run([accuracy, loss, predictions], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
            accuracy_dev += accuracy_run
            loss_dev += loss_run
            count_dev += 1
            with open(accuracy_dev_batch_file, 'a') as out:
                print(epoch, ";", count_dev, ":", accuracy_run, file=out)
            with open(loss_dev_batch_file, 'a') as out:
                print(epoch, ";", count_dev, ":", loss_run, file=out)
                # print(loss_run, file=out)
            with open(predictions_dev_file, 'a') as out:
                print(epoch, ";", count_dev, ":", predictions_run, file=out)
            np.save(os.path.join(predictions_folder_dev, str(epoch) + "-" + str(count_dev)), predictions_run)
        training = True
        stop_epoch_time = time.time()
        stop_epoch_time = stop_epoch_time - start_epoch_time
        with open(accuracy_dev_epoch_file, 'a') as out:
            print(epoch, ":", (accuracy_dev / count_dev), file=out)
        with open(loss_dev_epoch_file, 'a') as out:
            print(epoch, ":", (loss_dev / count_dev), file=out)
            # print((loss_dev / count_dev), file=out)

    # plots.plot_data(set="train", metric="accuracy", path=path_to_results_file, file=accuracy_train_epoch_file)
    # plots.plot_data(set="dev", metric="accuracy", path=path_to_results_file, file=accuracy_dev_epoch_file)
    # plots.plot_data(set="train", metric="loss", path=path_to_results_file, file=loss_train_epoch_file)
    # plots.plot_data(set="dev", metric="loss", path=path_to_results_file, file=loss_dev_epoch_file)
    print("Done")


# # Prepare files for results, weights+biases and saved models
# result_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".txt")
# results = os.path.join(path_to_results_file, result_file).replace(os.path.sep, '/')
# saved_model_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + ".ckpt")
# save_path = os.path.join(path_to_saved_models, saved_model_file).replace(os.path.sep, '/')
# weights_bias_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-weights-bias.txt")
# weights_bias = os.path.join(path_to_results_file, weights_bias_file).replace(os.path.sep, '/')
#
# write_parameters(results)
#
# # Initialise variables
# init = tf.global_variables_initializer()
# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
#
# # Training session
# with tf.Session() as sess:
#     init.run()
#     saver = tf.train.Saver()
#
#     for epoch in range(num_epochs):
#         loss_train = 0
#         count_train = 0
#         loss_dev = 0
#         count_dev = 0
#         start_epoch_time = time.time()
#         print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<")
#         with open(results, 'a') as out:
#             print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<", file=out)
#         with open(weights_bias, 'a') as out:
#             print(">>Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "<<", file=out)
#         for batch in utils.get_batches(x=x_train, y=y_train, mask=mask_train, rmd=rmdoublemask_train, mb_size=mb_size, shuffle=shuffle, rng_seed=rng_seed):
#             x_batch, y_batch, mask_batch, rmd_batch = batch
#             sess.run(fetches=optimizer, feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
#             variable_names = [v.name for v in tf.trainable_variables()]
#             values = sess.run(variable_names)
#             for k, v in zip(variable_names, values):
#                 with open(weights_bias, 'a') as out:
#                     print(k, v, file=out)
#             accuracy_run, loss_run = sess.run(fetches=[accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
#             loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
#             loss_train += loss_run
#             count_train += 1
#             with open(results, 'a') as out:
#                 print("Accuracy for this batch:", accuracy_run, file=out)
#         print("Save model")
#         with open(results, 'a') as out:
#             print("Training loss for this epoch:", loss_train, file=out)
#             print("Average training loss for this epoch:", (loss_train / count_train), file=out)
#             print("Save model", file=out)
#         saved = saver.save(sess=sess, save_path=save_path, global_step=epoch)
#
#         # Evaluation on dev data
#         training = False
#         print("Start evaluation on dev data")
#         with open(results, 'a') as out:
#             print("-----------------------\nStart evaluation on dev data", file=out)
#         with open(weights_bias, 'a') as out:
#             print("-----------------------\nStart evaluation on dev data", file=out)
#         saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(path_to_saved_models))
#         for batch in utils.get_batches(x=x_dev, y=y_dev, mask=mask_dev, rmd=rmdoublemask_dev, mb_size=mb_size, shuffle=shuffle, rng_seed=rng_seed):
#             x_batch, y_batch, mask_batch, rmd_batch = batch
#             variable_names = [v.name for v in tf.trainable_variables()]
#             values = sess.run(variable_names)
#             for k, v in zip(variable_names, values):
#                 with open(weights_bias, 'a') as out:
#                     print(k, v, file=out)
#             accuracy_run, loss_run = sess.run([accuracy, loss], feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
#             loss_eval = loss.eval(feed_dict={inputs: x_batch, labels: y_batch, mask: mask_batch, rmdmask: rmd_batch})
#             loss_dev += loss_run
#             count_dev += 1
#             with open(results, 'a') as out:
#                 print("Accuracy for this batch:", accuracy_run, file=out)
#         training = True
#         stop_epoch_time = time.time()
#         stop_epoch_time = stop_epoch_time - start_epoch_time
#         with open(results, 'a') as out:
#             print("Dev loss for this epoch:", loss_dev, file=out)
#             print("Average dev loss for this epoch:", (loss_dev / count_dev), file=out)
#             print("Epoch time: " + str(stop_epoch_time) + "\n------------------------------------------------------", file=out)
#         with open(weights_bias, 'a') as out:
#             print("Epoch time: " + str(stop_epoch_time) + "\n------------------------------------------------------", file=out)
#     print("Done")