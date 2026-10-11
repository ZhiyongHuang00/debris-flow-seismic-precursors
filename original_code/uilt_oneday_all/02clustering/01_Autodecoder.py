# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/16 22:34
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_Autodecoder.py
@Function: {}:
'''
# -*- coding: utf-8 -*-
'''
@Time    : 2024/9/6 10:59
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_Autoencoder_60s.py
@Function: {}:
'''
# !/usr/bin/env python
# coding: utf-8
from datetime import datetime
import pandas as pd
# 打印当前的日期和时间
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import Model_AE
import glob
from plot_distribution import plot_embedding, Visualization_2Ddistribution, cluster_assignments
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import trange
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split

sns.set_style('darkgrid')
sns.set_palette('muted')
print(tf.config.list_physical_devices('GPU'))

print(tf.config.list_physical_devices())
print(tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

save_dir = "./result"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 获取当前的日期和时间
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d%H%M%S")



def save_loss(train_loss,valid_loss,name_no_ext):
    # 保存损失值到CSV文件
    loss_df = pd.DataFrame({
        'Training Loss': train_loss,
        'Validation Loss': valid_loss
    })
    train_loss = pd.DataFrame(train_loss, columns=['Loss'])
    valid_loss = pd.DataFrame(valid_loss, columns=['Loss'])
    save_path = f'D:\Pycharm_Project\Find_precursor_signs\model\DEC\Figure\loss//{name_no_ext}_Loss.csv'
    loss_df.to_csv(save_path, index=False)    # 绘制损失曲线D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}Number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    img_path = f'D:\Pycharm_Project\Find_precursor_signs\model\DEC\Figure\loss/{name_no_ext}_Loss.png'
    plt.savefig(img_path)
    plt.close()
# 定义文件夹路径
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_aug_40\maxmin_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_aug_40\maxmin_40s\*\*.npy")
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\Wavelet\tmp_std_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\Wavelet\tmp_std_40s\label\*.npy")
train_files = glob.glob(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\STFT_oneday_all\Z\*Z.npz")
# train_files = glob.glob(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\STFT_oneday_all\Z\event35*Z.npz")


for train_file in train_files:
    base_name = os.path.basename(train_file)
    name_no_ext = os.path.splitext(base_name)[0]

    data = np.load(train_file, allow_pickle=True)
    x_train = data['stft']
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = x_train.astype(np.float32)

    print(x_train.shape)
    file_paths = data['files']
    file_path = file_paths[1]
    folder_name = os.path.basename(os.path.dirname(file_path))
    ###make noisy data
    def noise(array):
        """
        Adds random noise to each image in the supplied array.
        """
        noise_factor = 0.2
        noisy_array = array + noise_factor * np.random.normal(
            loc=0.0, scale=3.0, size=array.shape
        )
        return noisy_array
    def plot_decoder_output(origianl_data,output_data):
        # Plot the input spectrogram
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(origianl_data), aspect='auto', cmap='viridis')
        plt.title('Input Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()

        # Plot the output spectrogram
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(output_data), aspect='auto', cmap='viridis')
        plt.title('Output Spectrogram (Autoencoder)')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()

        plt.suptitle('Autoencoder Input vs Output Spectrogram')
        plt.tight_layout()
        plt.savefig(r"D:\Desktop\aiting.png")
        # Save the figure
        plt.show()
        plt.close()
    # x_train_noisy = noise(x_train)
    x_train_split, x_test_split = train_test_split(x_train, test_size=0.3, random_state=42, shuffle=True)
    ###building model
    encoder, decoder, autoencoder = Model_AE.CAE(input_shape=(64, 512, 1), filters=[8, 16, 32, 64, 128], encoding_filters=32,
                                                set_seed=[False, 87], summary=True)
    autoencoder.compile(optimizer='adam', loss='mse')
    save_dir = r'./result\AE'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_logger = tf.keras.callbacks.CSVLogger(rf'D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_oneday_all\02clustering\result\AE/{name_no_ext}_pretrain_log.csv')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=1e-4,  restore_best_weights=True) # min_delta=1e-5,
    print(2)

    history = autoencoder.fit(x_train_split, x_train_split,
                              validation_data=(x_test_split, x_test_split),
                              batch_size=128,
                              epochs=500,
                              callbacks=[csv_logger, early_stopping]
                              )
    # 获取 loss 和 val_loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    # loss_values = history.history['loss']
    # save_loss(train_loss, val_loss, name_no_ext)
    output_data = autoencoder.predict(x_train)
    loss_dict = {'train_loss': train_loss, 'val_loss': val_loss}
    save_dir = r'./result\model_save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    WeightsFileName = [fr'.\result\model_save//{name_no_ext}ACE_Pretrain_encoder.h5',
                       fr'.\result\model_save//{name_no_ext}ACE_Pretrain_autoencoder.h5',
                       fr'.\result\model_save//{name_no_ext}ACE_Pretrain_decoder.h5']
    encoder.save_weights(WeightsFileName[0])
    autoencoder.save_weights(WeightsFileName[1])
    decoder.save_weights(WeightsFileName[2])
    features = encoder.predict(x_train)
    # 保存 features 为 .npy 文件
    save_dir = r'./result\feature'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = f'.\\result\\feature\\{name_no_ext}_features.npy'  # 修改为实际保存路径

    save_mode_output = {
        'spectral': x_train,
        'features': features,
        're_spectral': output_data,
        'wave_file': file_paths,
        'loss': loss_dict
    }
    save_dir = r'./result\all_parameters'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name_oup = fr'.\result\\all_parameters\{name_no_ext}_ACE_model.npz'
    np.savez(save_name_oup, **save_mode_output)
    print(f'Save model to {save_name_oup}')
    a = 1