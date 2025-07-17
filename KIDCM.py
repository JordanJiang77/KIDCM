import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import os
import numpy.linalg as la
from input_data import preprocess_data, load_sz_data, load_los_data
from tgcn import tgcnCell
from visualization1 import plot_result, plot_result1, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 300, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 6, 'time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 33, 'batch size.')
flags.DEFINE_string('dataset', 'sz', 'sz or los.')
flags.DEFINE_string('model_name', 'GRU', 'TGCN or GRU or GCN.')

model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
data = pd.read_excel(r'E:\I-24motiondata/dataset/flowdemo.xlsx')
adjor = pd.read_excel(r'E:\I-24motiondata/dataset/adjdemo1.xlsx')
weights = pd.read_excel(r'E:\I-24motiondata/dataset/weight5.xlsx')
bias = pd.read_excel(r'E:\I-24motiondata/dataset/biases5.xlsx')
data = np.array(data)
adjor = np.array(adjor)
# weights = np.array(weights)
# bias = np.array(bias)

weights_str = weights.iloc[0, 0]
bias = list(bias)
bias = np.array(bias)

weights = np.array(weights)
# print(weights)

# weights = np.array([[float(x) for x in weights_str.strip('[]\n').split(']\n [')]]).T

timelen = data.shape[0]
num_nodes = data.shape[1]

data1 = np.mat(data, dtype=np.float32) * 36
time_len = data1.shape[0]
# noise = np.random.normal(0,0.2,size=data1.shape)
# noise = np.random.poisson(8,size=data.shape)
# scaler = MinMaxScaler()
# scaler.fit(noise)
# noise = scaler.transform(noise)
# data1 = data1 + noise
print(timelen)


def introduce_random_missing(data, missing_rate=0.1):
    mask = np.random.random(data.shape)
    data_with_missing = data.copy()
    data_with_missing[mask < missing_rate] = np.nan
    return data_with_missing


def fill_missing_with_neighbor_mean(data_with_missing):
    filled_data = data_with_missing.copy()
    rows, cols = data_with_missing.shape

    for i in range(rows):
        for j in range(cols):
            if np.isnan(filled_data[i, j]):
                # 收集周围8个邻居的值
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # 跳过自身
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(filled_data[ni, nj]):
                            neighbors.append(filled_data[ni, nj])

                # 如果有邻居值，用均值填充；否则保持nan
                if neighbors:
                    filled_data[i, j] = np.mean(neighbors)
                # 如果没有邻居值，暂时保持nan（可以在外层再处理）

    return filled_data


# 缺失比例设定
missing_rate = 0.1

# 引入随机缺失值
data_with_missing = introduce_random_missing(data1, missing_rate=missing_rate)

# 填充缺失值
data1 = fill_missing_with_neighbor_mean(data_with_missing)

# 检查是否还有缺失值 (边界情况可能无法填充)
if np.isnan(data1).any():
    # 对于无法用邻居填充的值，使用全局均值填充
    global_mean = np.nanmean(data1)
    filled_data = np.nan_to_num(data1, nan=global_mean)

def rotate_x(matrix, angle):
    # 将角度转换为弧度
    theta = np.radians(angle)

    # 提取矩阵的坐标
    x = matrix  # 假设 x 坐标为矩阵本身
    y = np.zeros_like(x)  # 假设 y 坐标为 0
    z = np.zeros_like(x)  # 假设 z 坐标为 0

    # 计算旋转后的坐标
    x_new = x
    y_new = y * np.cos(theta) - z * np.sin(theta)
    z_new = y * np.sin(theta) + z * np.cos(theta)

    # 将结果重新组合成矩阵
    rotated_matrix = np.stack((x_new, y_new, z_new), axis=-1)

    # 返回旋转后的矩阵（忽略 z 坐标）
    return rotated_matrix[:, :, 0]

def rotate_y(matrix, angle):
    # 将角度转换为弧度
    theta = np.radians(angle)

    # 提取矩阵的坐标
    x = matrix  # 假设 x 坐标为矩阵本身
    y = np.zeros_like(x)  # 假设 y 坐标为 0
    z = np.zeros_like(x)  # 假设 z 坐标为 0

    # 计算旋转后的坐标
    x_new = x * np.cos(theta) + z * np.sin(theta)
    y_new = y
    z_new = -x * np.sin(theta) + z * np.cos(theta)

    # 将结果重新组合成矩阵
    rotated_matrix = np.stack((x_new, y_new, z_new), axis=-1)

    # 返回旋转后的矩阵（忽略 z 坐标）
    return rotated_matrix[:, :, 0]


def correlation(x1, x2):
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    c1 = np.std(x1)
    c2 = np.std(x2)
    cor = np.mean((x1 - m1) * (x2 - m2))
    autoco = cor / np.sqrt(c1 * c2)
    return autoco

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

time_gap = 12
time_steps = int(time_len / time_gap)
print(time_steps)

Spatialgraph = []
for a in range(1, time_steps):
    datagraph = data[(a - 1) * time_gap:a * time_gap]
    sgraph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            a = datagraph[:, i]
            b = datagraph[:, j]
            sgraph[i, j] = correlation(a, b.T)
    Spatialgraph.append(sgraph)
Spatialgraph = np.array(Spatialgraph)

degree1 = 10
degree2 = 70
Graphlen = Spatialgraph.shape[0]
trainlen = int(Graphlen * train_rate)
Graphtrain = Spatialgraph[:trainlen]
Graph_period1 = Graphtrain[trainlen-1:]
Graph_period1 = np.reshape(Graph_period1, (num_nodes, num_nodes))
Graph_period2 = Graphtrain[trainlen-2:trainlen-1]
Graph_period2 = np.reshape(Graph_period2, (num_nodes, num_nodes))
Graph_use = 4.1942 + -0.0402 * Graph_period1 + 0.4816 * Graph_period2
print(Graph_use.shape)
Graph_use = rotate_y(rotate_x(Graph_use, degree1), degree2)
print(Graph_use.shape)
Graph_use = np.reshape(Graph_use, (num_nodes, num_nodes))

adj = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if adjor[i, j] == 0:
            adj[i, j] = 0
        else:
            adj[i, j] = Graph_use[i, j]
adj = adj

array_np = np.array(adj)
array_cleaned = np.where(np.isfinite(array_np), array_np, 0)
adj = array_np

def TGCN(_X, _weights, _biases):
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights) + _biases
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states

trainX1, trainY1, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
dropda = int(trainX1.shape[0] * (1-0.2))

trainX = trainX1[dropda:]
trainY = trainY1[dropda:]


totalbatch = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)

print(testX.shape[0])
print(testY.shape)

###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

weights = tf.cast(weights, tf.float32)
bias = tf.cast(bias, tf.float32)

pred, ttto, alpha = TGCN(inputs, weights, bias)
y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s' % (model_name)

###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = np.abs((a - b) / b).mean() * 100
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, F_norm, r2, var

x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []
time_start = time.time()

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1)

    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs: testX, labels: testY})
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label
    test_output1 = test_output
    test_loss.append(loss2)
    test_rmse.append(rmse)
    test_mae.append(mae)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))

time_end = time.time()
print(time_end - time_start, 's')

############## visualization ###############
b = int(len(batch_rmse) / totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
plot_result1(test_result, test_label1)

# 计算模型浮点数
# def calculate_flops():
#     # 创建一个计算 FLOPs 的上下文
#     with tf.profiler.experimental.Profile('logdir'):
#         # 运行一次前向传播
#         sess.run(y_pred, feed_dict={inputs: testX, labels: testY})
#
#     # 读取 FLOPs 数据
#     options = option_builder.ProfileOptionBuilder.float_operation()
#     flops = tf.profiler.experimental.profile(
#         'logdir',
#         options=options
#     )
#     return flops.total_float_ops
#
# # 在训练结束后调用计算 FLOPs
# flops = calculate_flops()
# print(f"Total FLOPs: {flops}")
#
# # 在训练过程中插入内存占用计算
# def get_memory_usage():
#     if tf.config.list_physical_devices('GPU'):
#         device = '/GPU:0'
#     else:
#         device = '/CPU:0'
#
#     memory_info = tf.config.experimental.get_memory_info(device)
#     memory_used = memory_info['current'] / (1024 ** 2)  # 转换为 MB
#     memory_total = memory_info['peak'] / (1024 ** 2)    # 转换为 MB
#     return memory_used, memory_total
#
# # 在训练结束后调用计算内存占用
# memory_used, memory_total = get_memory_usage()
# print(f"Memory used: {memory_used:.2f} MB")
# print(f"Peak memory usage: {memory_total:.2f} MB")
#
print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])

# df1 = pd.DataFrame(test_result)
# df2 = pd.DataFrame(test_label1)
#
# df1.to_excel('Fresult.xlsx', index=False)
# df2.to_excel('Frealda.xlsx', index=False)