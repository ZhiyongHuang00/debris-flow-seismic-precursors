# -*- coding: utf-8 -*-
'''
@Time    : 2025/7/8 2:05
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_DEC.py
@Function: {}:
'''
# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


from datetime import date

today = str(date.today())
print("Today's date:", today)
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import Model_AE
from plot_distribution import plot_embedding, Visualization_2Ddistribution, cluster_assignments
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import metrics
import seaborn as sns
import csv
import glob
import pandas as pd
from datetime import datetime

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, \
    UpSampling1D, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, \
    CSVLogger
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_style('darkgrid')
sns.set_palette('dark')
print(tf.config.list_physical_devices())
print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   #Invalid device or cannot modify virtual devices once initialized.
#   pass

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 获取GPU设备

# 如果存在GPU设备，设置内存增长
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print("Error setting memory growth:", e)
# In[2]:
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
# Setting Config

import glob
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(87)

def resolve_model_weights(config, event_name):
    resolved_paths = []
    for path_template in config['WeightsFileName']:
        pattern = path_template.format(event_name=event_name)
        matches = glob.glob(pattern)
        resolved_paths.append(matches[0])  # 默认取第一个匹配

    return resolved_paths  # [encoder_path, autoencoder_path, decoder_path]


def load_and_concatenate(files, column_index=None):
    data_list = []
    i = 0
    for file in files:
        data = np.load(file, allow_pickle=True)
        if column_index is not None:
            data = data[:, column_index]

        # if len(data) > 9000:
        #     data = data[5000:10000]
        # else:
        #     data = data[:5000]
        data = data[:3000]
        data_list.append(data)
        print(file)
        del data
        i = i + 1
    return np.concatenate(data_list, axis=0)



import os
import re
# optimal_clluster = pd.read_excel(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt\02clustering\result\summary_cluster_selection.xlsx")
# time_file = pd.read_excel(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\station_time_windows_with_amp.xlsx")
time_file = pd.read_excel(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\新版完整数据集时间.xlsx")
cluster_K = pd.read_excel(r".\result\Figure\\summary_best_k.xlsx")
# 定义文件夹路径
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_40\tmp_std_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_40\tmp_std_40s\*\*.npy")
train_files = glob.glob(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\STFT_oneday_all\Z\*.npz")
for file in train_files:
    data = np.load(file, allow_pickle=True)
    x_train = data['stft']
    x_file = data['files']
    # label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_60\aug_maxmin\*\*.npy")
    # x_train = load_and_concatenate(train_files)
    # y_train = load_and_concatenate(label_files, 0)
    # file_position = load_and_concatenate(label_files, 1)
    # y_train = y_train.astype(np.float16)
    match = re.search(r'event\d+', x_file[0])
    event_name = match.group(0)
    # event_name = os.path.normpath(x_file[0]).split(os.sep)[-3]

    x_train = np.expand_dims(x_train, axis=-1)
    x_train = x_train.astype(np.float16)

    save_dirs = r'.\result\model_save'
    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
    config = {"dirname": "iter3e4",
              "WeightsFileName": [rf".\result\model_save\\{event_name}_*Pretrain_encoder.h5",
                                  rf".\result\model_save\\{event_name}_*_Pretrain_autoencoder.h5",
                                  rf".\result\model_save\\{event_name}_*_Pretrain_decoder.h5"],
              "n_clusters": 12,
              "gamma": 0.1,
              "batch_size": 256,
              "maxiter": 20000,
              "tol": 0.001,  # tolerance threshold to stop training
              "update_interval": 10,
              "save_interval": 5000
              }
    matching_row = time_file[time_file['Event'] == event_name]
    manual_label_start = matching_row.iloc[0]['Manual_Label_start']
    matching_K = cluster_K[cluster_K['Event'] == event_name]
    clusters =matching_K.iloc[0]['WCSS']

    # clusters = matching_row.iloc[0]['Optimal_Cluster']
    # aic_row = optimal_clluster[optimal_clluster['File'] == event_name]
    # clusters = aic_row.iloc[0]['AIC']
    config["n_clusters"] = clusters
    n_clusters = config["n_clusters"]
    # y_true = y_train
    DEC_weightsdir = config["dirname"]

    # save_dir = rf'D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_30s\02clustering\result\DEC_result/{event_name}/'.format(
    #     str(config["gamma"]).split('.')[1], config["dirname"], str(config["n_clusters"]))
    save_dir = rf'.\result\DEC_result'
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir + "/image/"):
        os.makedirs(save_dir + "/image/")
    save_imagedir = save_dir + "/image/"

    print(save_dir)
    print(x_train.shape)
    encoder_path, autoencoder_path, decoder_path = resolve_model_weights(config, event_name)

    # # Buliding the model
    # Buliding the autoencoder
    encoder, decoder, autoencoder = Model_AE.CAE(input_shape=(64, 512, 1), filters=[8, 16, 32, 64, 128], encoding_filters=32,
                                                set_seed=[True, 87], summary=True)
    encoder.load_weights(encoder_path)
    autoencoder.load_weights(autoencoder_path)
    decoder.load_weights(decoder_path)


    # ## adding the clustering layer into the bottelneck layer

    # In[5]:

    class ClusteringLayer(Layer):
        def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super(ClusteringLayer, self).__init__(**kwargs)
            self.n_clusters = n_clusters
            self.alpha = alpha
            self.initial_weights = weights
            self.input_spec = InputSpec(ndim=2)

        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]
            self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                            name='clusters')
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            self.built = True

        def call(self, inputs, **kwargs):
            q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
            q **= (self.alpha + 1.0) / 2.0
            q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
            return q

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            return input_shape[0], self.n_clusters

        def get_config(self):
            config = {'n_clusters': self.n_clusters}
            base_config = super(ClusteringLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


    print('...Finetuning...')
    # Define DCEC model
    # clustering_layer = Model_AE.ClusteringLayer(config["n_clusters"], name='clustering')(encoder.output)
    clustering_layer = ClusteringLayer(config["n_clusters"], name='clustering')(encoder.output)
    DECmodel = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output], name='DEC')
    DECmodel.compile(loss=['kld', 'mse'], loss_weights=[config["gamma"], 1], optimizer='adam')
    DECmodel.summary()

    # ## initializing the weights
    ### initializing the weights using Kmean and assigning them to the model
    features = encoder.predict(x_train)
    kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
    y_pred = kmeans.fit_predict(features)
    y_pred_last = np.copy(y_pred)
    DECmodel.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    ## parameters for the finetuning
    loss = [0, 0, 0]
    index = 0
    index_array = np.arange(x_train.shape[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###############################################################################
    ### simultaneous optimization and clustering
    # logfile = open(save_dir + 'scatter/dcec_log.csv', 'w')
    # logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'purity', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
    # logwriter.writeheader()


    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T



    tol = 0.0001  # tolerance threshold to stop training
    loss_history = []

    for ite in range(int(config["maxiter"])):
        if ite % config["update_interval"] == 0:
            if len(x_train) > 15000:
                q1, _ = DECmodel.predict(x_train[0:15000], verbose=0)
                q2, _ = DECmodel.predict(x_train[15000:], verbose=0)
                q = np.concatenate((q1, q2), axis=0)
            else:
                q, _ = DECmodel.predict(x_train, verbose=0)
            # ===== 新增：获取簇中心并计算距离 =====
            cluster_centers = DECmodel.get_layer(name='clustering').get_weights()[0]  # shape=(n_clusters, embedding_dim)
            # 获取样本的嵌入向量（即输入到cluster_layer的特征）
            # features = DECmodel.get_layer(name='encoder').predict(x_train, verbose=0)
            encoder_model = Model(inputs=DECmodel.input, outputs=DECmodel.get_layer('encoded').output)
            features = encoder_model.predict(x_train, verbose=0)  # (n_samples, embedding_dim)

            # 计算每个样本到每个簇中心的欧氏距离
            distances = np.linalg.norm(features[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2)
            # update the auxiliary target distribution p
            p = target_distribution(q)
            # evaluate the clustering performance
            y_pred = q.argmax(1)
            print('Iter %d: ' % (ite), ' ; loss=', loss)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < config["tol"]:
                print('delta_label ', delta_label, '< tol ', config["tol"])
                break

        idx = index_array[index * config["batch_size"]: min((index + 1) * config["batch_size"], x_train.shape[0])]
        loss = DECmodel.train_on_batch(x=x_train[idx], y=[p[idx], x_train[idx]])
        loss_history.append(loss)  # 保存每次的损失

        index = index + 1 if (index + 1) * config["batch_size"] <= x_train.shape[0] else 0
        # tf.keras.backend.clear_session()
    # save DCEC model checkpoints
    # logfile.close()
    model_weight_path = os.path.join(save_dir, 'model_weight')
    os.makedirs(model_weight_path, exist_ok=True)  # 如果已存在不会报错
    print('saving model to:', model_weight_path, f'{event_name}_dcec_model_final.h5')
    model_path = os.path.join(model_weight_path, f'{event_name}_dcec_model_final.h5')
    DECmodel.save_weights(model_path)
    # DECmodel.save_weights(save_dir + '/dcec_model_final.h5')
    print('finish :' + str(ite))
    loss_history = np.array(loss_history)  # 转成数组，shape可能是 (step, 3)
    df_loss = pd.DataFrame(loss_history, columns=['total_loss', 'cluster_loss', 'reconstruction_loss'])
    losscsv_dir = os.path.join(save_dir, 'loss_csv')
    os.makedirs(losscsv_dir, exist_ok=True)  # 如果已存在不会报错

    df_loss.to_csv(losscsv_dir + f'/{event_name}training_loss.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df_loss['total_loss'], label='Total Loss')
    plt.plot(df_loss['cluster_loss'], label='Cluster Loss')
    plt.plot(df_loss['reconstruction_loss'], label='Reconstruction Loss')

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    loss_dir = os.path.join(save_imagedir, 'loss')
    os.makedirs(loss_dir, exist_ok=True)  # 如果已存在不会报错

    plt.savefig(os.path.join(loss_dir, f'{event_name}_scatter.png'))
    # plt.savefig(save_imagedir + f'loss/{event_name}_loss_curve.png')
    # plt.show()
    plt.close()
    import csv

    DECmodel.load_weights(model_path)
    print(save_dir + f'/{event_name}_dcec_model_final.h5')
    features = encoder.predict(x_train)
    reconstruction_all = decoder.predict(features)

    y_pred_last = np.copy(y_pred)

    redu = TSNE(n_components=2, perplexity=40, random_state=74).fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(redu[:, 0], redu[:, 1], c=y_pred_last, cmap='tab10', s=15, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(f'{event_name} t-SNE Clustering Scatter Plot')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    scatter_dir = os.path.join(save_imagedir, 'scatter')
    os.makedirs(scatter_dir, exist_ok=True)  # 如果已存在不会报错

    plt.savefig(os.path.join(scatter_dir, f'{event_name}_scatter.png'))
    # plt.savefig(save_imagedir + f'scatter/{event_name}_scatter.png')
    # plt.show()
    plt.close()


    save_mode_output = {
        'x_file': x_file,
        "spectral": x_train,
        'feature': features,
        'redu': redu,
        'respectral' : reconstruction_all,
        "y_pred_last": y_pred_last,
        "n_clusters": n_clusters,
        'probalities': q,
        'distances': distances,
    }
    # all_parement
    all_parement_dir = os.path.join(save_dir, 'all_parement')
    os.makedirs(all_parement_dir, exist_ok=True)  # 如果已存在不会报错
    save_name_oup = os.path.join(all_parement_dir, f'{event_name}_all_parement.npz')
    # f'D:\\Pycharm_Project\\Classification\\clustering_method\\DEC\\Model_output\\{current_datetime}_{n_clusters}_{purity}_DEC_model.npz'
    np.savez(save_name_oup, **save_mode_output)
    print('save to file：', save_name_oup)

    # In[ ]:




