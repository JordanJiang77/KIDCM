import pandas as pd
import numpy as np
import math
import os
from sklearn.cluster import KMeans
import time
from statsmodels.tsa.arima.model import ARIMA
from collections import defaultdict
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.optimize import curve_fit

time_start = time.time()


def fun(x, a=0.827402, b=0.013296):
    return a * x * np.exp(-b * x)


###### load data ######
density = pd.read_excel(r'E:\I-24motiondata/densitydemo.xlsx')
adjor = pd.read_excel(r'E:\I-24motiondata/adjdemo1.xlsx')
data = fun(density) * 10
data = np.array(data)
adjor = np.array(adjor)
density = np.array(density)
# print(data)

time_len = data.shape[0]
num_nodes = data.shape[1]


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

Spatialgraph = []
for a in range(1, time_steps):
    datagraph = data[(a - 1) * time_gap:a * time_gap]
    sgraph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            a = datagraph[:, i]
            b = datagraph[:, j]
            sgraph[i, j] = correlation(a, b.T)
    # print('......................')
    # print(sgraph)
    Spatialgraph.append(sgraph)
Spatialgraph = np.array(Spatialgraph)
# print(Spatialgraph)

print('.......................')

# 使用K-means将每个160x160数组映射到有限数量的状态
num_states = 10
kmeans = KMeans(n_clusters=num_states, random_state=0, n_init=10)
flattened_spatialgraph = Spatialgraph.reshape(Spatialgraph.shape[0], -1)

# 使用K-means进行聚类
kmeans.fit(flattened_spatialgraph)
state_labels = kmeans.predict(flattened_spatialgraph)

# 计算二阶转移概率矩阵
transition_matrix = np.zeros((num_states, num_states, num_states))

# 计算转移次数
for t in range(2, len(state_labels)):
    prev_state1 = state_labels[t - 2]
    prev_state2 = state_labels[t - 1]
    current_state = state_labels[t]
    transition_matrix[prev_state1, prev_state2, current_state] += 1

# 归一化得到转移概率
for i in range(num_states):
    for j in range(num_states):
        total = transition_matrix[i, j].sum()
        if total > 0:
            transition_matrix[i, j] /= total
        else:
            transition_matrix[i, j] = 1 / num_states  # 如果没有转移，均匀分布

# 生成初始状态
reconstructed_states = [state_labels[0], state_labels[1]]

# 根据二阶转移概率矩阵生成后续状态
for _ in range(2, len(state_labels)):
    prev_state1 = reconstructed_states[-2]
    prev_state2 = reconstructed_states[-1]
    next_state = np.random.choice(num_states, p=transition_matrix[prev_state1, prev_state2])
    reconstructed_states.append(next_state)

# 将重构的状态映射回原始空间
reconstructed_spatialgraph = kmeans.cluster_centers_[reconstructed_states].reshape(-1, 160, 160)
print(reconstructed_spatialgraph.shape)
print(Spatialgraph.shape)

# 计算重构误差
reconstructed_derivation = reconstructed_spatialgraph - Spatialgraph
print(reconstructed_derivation)

# 使用K-means将每个160x160数组映射到有限数量的状态
num_states = 10
kmeans = KMeans(n_clusters=num_states, random_state=0, n_init=10)
flattened_spatialgraph = reconstructed_spatialgraph.reshape(reconstructed_spatialgraph.shape[0], -1)

# 使用K-means进行聚类
kmeans.fit(flattened_spatialgraph)
state_labels = kmeans.predict(flattened_spatialgraph)

# 计算二阶转移概率矩阵
transition_matrix = np.zeros((num_states, num_states, num_states))

# 计算转移次数
for t in range(2, len(state_labels)):
    prev_state1 = state_labels[t - 2]
    prev_state2 = state_labels[t - 1]
    current_state = state_labels[t]
    transition_matrix[prev_state1, prev_state2, current_state] += 1

# 归一化得到转移概率
for i in range(num_states):
    for j in range(num_states):
        total = transition_matrix[i, j].sum()
        if total > 0:
            transition_matrix[i, j] /= total
        else:
            transition_matrix[i, j] = 1 / num_states  # 如果没有转移，均匀分布

# 将 state_labels 转换为时间序列数据
time_series = pd.Series(state_labels)

# 使用 ARIMA 模型进行时间序列建模
# model = ARIMA(time_series, order=(2, 1, 0))  # ARIMA(2, 1, 0) 模型
model = AutoReg(time_series, lags=2)  # AR(2) 模型
model_fit = model.fit()

# 获取模型的系数
coefficients = model_fit.params
print("Model Coefficients:", coefficients)

# 输出每一个时间步的状态和前两个时间步的状态之间的关系表达式
print("Relationship Expression:")
print(
    f"Current State = {coefficients[0]:.4f} + {coefficients[1]:.4f} * State(t-1) + {coefficients[2]:.4f} * State(t-2)")
# print(f"Current State = {coefficients[0]:.4f} * (State(t-1)-State(t-2)) + {coefficients[1]:.4f} * (State(t-2)-State(t-3)) + {coefficients[2]:.4f}")

# 预测未来状态
forecast = model_fit.predict(start=len(time_series), end=len(time_series) + 9)  # 预测未来10个时间步的状态
print("Forecasted states:", forecast)

# 绘制原始时间序列和预测结果
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original')
plt.plot(np.arange(len(time_series), len(time_series) + len(forecast)), forecast, label='Forecast')
plt.title('Time Series and Forecast')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.legend()
plt.show()

# 定义激活函数
def activation_function(x):
    return tf.nn.tanh(x)  # 你可以选择其他激活函数，如tf.nn.sigmoid, tf.nn.tanh等

# 修改后的X_around函数
def X_around(graph, degree):
    cos_degree = tf.cos(degree)
    sin_degree = tf.sin(degree)
    rotation_matrix = tf.convert_to_tensor([
        [1, 0, 0],
        [0, cos_degree, -sin_degree],
        [0, sin_degree, cos_degree]
    ], dtype=tf.float32)

    points = tf.zeros((num_nodes, num_nodes, 3), dtype=tf.float32)
    indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(num_nodes), tf.range(num_nodes)), axis=-1), (-1, 2))
    updates = tf.tile(tf.reshape(graph, (-1, 1)), [1, 3])  # 确保updates的形状为[num_nodes * num_nodes, 3]
    updates = tf.cast(updates, tf.float32)  # 确保updates的数据类型与points一致
    points = tf.tensor_scatter_nd_update(points, indices, updates)

    points_reshaped = tf.reshape(points, (-1, 3))

    rotated_points = tf.matmul(points_reshaped, rotation_matrix, transpose_b=True)

    rotated_points = tf.reshape(rotated_points, (num_nodes, num_nodes, 3))
    rotated_matrix = rotated_points[:, :, 2]

    return activation_function(rotated_matrix)  # 添加激活函数

# 修改后的Y_around函数
def Y_around(graph, degree):
    cos_degree = tf.cos(degree)
    sin_degree = tf.sin(degree)
    rotation_matrix = tf.convert_to_tensor([
        [cos_degree, 0, sin_degree],
        [0, 1, 0],
        [-sin_degree, 0, cos_degree]
    ], dtype=tf.float32)

    points = tf.zeros((num_nodes, num_nodes, 3), dtype=tf.float32)
    indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(num_nodes), tf.range(num_nodes)), axis=-1), (-1, 2))
    updates = tf.tile(tf.reshape(graph, (-1, 1)), [1, 3])  # 确保updates的形状为[num_nodes * num_nodes, 3]
    updates = tf.cast(updates, tf.float32)  # 确保updates的数据类型与points一致
    points = tf.tensor_scatter_nd_update(points, indices, updates)

    points_reshaped = tf.reshape(points, (-1, 3))

    rotated_points = tf.matmul(points_reshaped, rotation_matrix, transpose_b=True)

    rotated_points = tf.reshape(rotated_points, (num_nodes, num_nodes, 3))
    rotated_matrix = rotated_points[:, :, 2]

    return activation_function(rotated_matrix)  # 添加激活函数

# 定义损失函数
def compute_loss(predicted, actual):
    return tf.reduce_mean(tf.square(predicted - actual))  # 均方误差损失

# 定义不同的学习率
learning_rate_Xdegree = 0.05
learning_rate_Ydegree = 0.05
learning_rate_soft_attention = 0.01

# 定义优化器
optimizer_Xdegree = tf.train.RMSPropOptimizer(learning_rate=learning_rate_Xdegree)
optimizer_Ydegree = tf.train.RMSPropOptimizer(learning_rate=learning_rate_Ydegree)

optimizer_soft_attention = tf.train.RMSPropOptimizer(learning_rate=learning_rate_soft_attention)

# 定义Soft Attention层
def soft_attention(inputs):
    # 计算 Query, Key, Value
    query = tf.matmul(inputs, query_weights)
    key = tf.matmul(inputs, key_weights)
    value = tf.matmul(inputs, value_weights)

    # 计算注意力分数
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    # 计算加权和
    attention_output = tf.matmul(attention_scores, value)

    # 线性变换
    output_layer = tf.matmul(attention_output, output_weights)

    return output_layer

# 训练循环
def train_step(Xdegree, Ydegree, reconstructed_spatialgraph, session):
    total_loss = 0.0
    all_variables = []

    for i in range(reconstructed_spatialgraph.shape[0] - 2):
        # 使用前两个时间步的数据来预测下一个时间步的数据
        Autoregressive = coefficients[1] * reconstructed_spatialgraph[i] + coefficients[2] * reconstructed_spatialgraph[i + 1]
        intermediate_output = tf.nn.relu(Y_around(X_around(Autoregressive, Xdegree), Ydegree))
        predicted_next_step = soft_attention(intermediate_output)
        predicted_next_step = tf.reshape(predicted_next_step, (num_nodes, num_nodes))

        loss = compute_loss(predicted_next_step, reconstructed_spatialgraph[i + 2])
        total_loss += loss

        # 收集所有变量
        all_variables.extend(tf.trainable_variables())

    # 计算平均损失
    average_loss = total_loss / (reconstructed_spatialgraph.shape[0] - 2)

    # 计算梯度
    grads = tf.gradients(average_loss, [Xdegree, Ydegree] + all_variables)

    # 分离梯度
    grads_Xdegree = grads[0]
    grads_Ydegree = grads[1]
    grads_soft_attention = grads[2:]

    # 应用不同的学习率
    optimizer_Xdegree.apply_gradients([(grads_Xdegree, Xdegree)])
    optimizer_Ydegree.apply_gradients([(grads_Ydegree, Ydegree)])
    optimizer_soft_attention.apply_gradients(zip(grads_soft_attention, all_variables))

    return average_loss, all_variables

# 初始化参数
num_nodes = reconstructed_spatialgraph.shape[1]  # 假设 reconstructed_spatialgraph 是一个 numpy 数组
hidden_units = 64
Xdegree = tf.Variable(30.0, dtype=tf.float32)
Ydegree = tf.Variable(120.0, dtype=tf.float32)
output_weights = tf.Variable(tf.random_normal([hidden_units, num_nodes]), name='output_weights')
query_weights = tf.Variable(tf.random_normal([num_nodes, hidden_units]), name='query_weights')
key_weights = tf.Variable(tf.random_normal([num_nodes, hidden_units]), name='key_weights')
value_weights = tf.Variable(tf.random_normal([num_nodes, hidden_units]), name='value_weights')

# 创建会话
session = tf.Session()
session.run(tf.global_variables_initializer())

# 训练过程
num_epochs = 20
for epoch in range(num_epochs):
    loss, all_variables = session.run(train_step(Xdegree, Ydegree, reconstructed_spatialgraph, session))
    print(f"Epoch {epoch + 1}, Loss: {loss}")
    print(session.run(Xdegree))
    print(session.run(Ydegree))
print(session.run(query_weights))
print(session.run(key_weights))
print(session.run(value_weights))

# 最终的Xdegree和Ydegree
print(f"Final Xdegree: {session.run(Xdegree)}")
print(f"Final Ydegree: {session.run(Ydegree)}")