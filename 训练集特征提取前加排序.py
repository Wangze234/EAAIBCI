import numpy as np
# 获取排序后的索引
import scipy.stats as sp
import scipy.io as scio
import csv


def read_EEG_values(file_path):
    # 获取预处理滤波后的数据
    processdata = scio.loadmat(file_path)
    processdata = processdata['EEG']
    traindataProc = np.double(processdata[0][0][15])
    # 获取滤波后的脑电数据
    arr_transposed = np.transpose(traindataProc, (1, 0, 2))
    return arr_transposed

def Getlabel(filepath):
    # 获取原始数据
    rawdata = scio.loadmat(filepath)
    # 获取原始数据中的label
    traindataraw = rawdata['epo_train']
    label = np.double(traindataraw[0][0][5])
    label = np.transpose(label)
    # 将 one-hot 编码的标签数组转换为整数标签
    integer_labels = np.argmax(label, axis=1)
    return integer_labels



def hjorth(input):                                             # function for hjorth
    realinput = input
    hjorth_activity = np.zeros((3, 6))
    hjorth_mobility = np.zeros((3, 6))
    hjorth_diffmobility = np.zeros((3, 6))
    hjorth_complexity = np.zeros((3, 6))
    for i in range(3):
        for j in range(6):
            hjorth_activity[i, j] = np.var(realinput[i * 265:(i + 1) * 265, j])
    diff_input = np.diff(realinput, axis=0)
    diff_diffinput = np.diff(diff_input, axis=0)
    for i in range(3):
        for j in range(6):
            hjorth_mobility[i, j] = np.sqrt(
                np.var(diff_input[i * 265:(i + 1) * 265, j]) / hjorth_activity[i, j])
            hjorth_diffmobility[i, j] = np.sqrt(
                np.var(diff_diffinput[i * 264:(i + 1) * 264, j]) / np.var(diff_input[i * 265:(i + 1) * 265, j]))
            hjorth_complexity[i, j] = hjorth_diffmobility[i, j] / hjorth_mobility[i, j]
    return np.sum(hjorth_activity, axis=0) / 8, np.sum(hjorth_mobility, axis=0) / 8, np.sum(
        hjorth_complexity, axis=0) / 8

# 提取对应维度的特征
def Get_Features(dim, data):

    features2 = []
    for i in range(dim):
        hjorthfeatures = hjorth(data[:, :, i])
        combined_features = np.hstack((hjorthfeatures))
        features2.append(combined_features)

    features2 = np.array(features2)
    print(features2.shape)
    lines = [l for l in features2]

    for i in range(len(lines[0]) - 1):
        columns = []
        for j in range(0, len(lines)):
            columns.append(float(lines[j][i]))
        mean = np.mean(columns, axis=0)
        std_dev = np.std(columns, axis=0)

        for j in range(0, len(lines)):
            lines[j][i] = (float(lines[j][i]) - mean) / std_dev
    lines = np.array(lines)
    print(lines.shape)
    return lines

def main():
    # 包含标签的原始训练集数据
    rawdataFile = 'F:\\EAAIBCI\\EAAIBCI\\Training set\\Data_Sample15.mat'
    # eeglab处理好之后的训练集数据
    procdatafile = 'F:\\EAAIBCI\\EAAIBCI\\TraindataAfilter\\Train_fil\\Subj15.mat'

    integer_labels = Getlabel(rawdataFile)
    # 获取预处理滤波后的数据
    arr_transposed = read_EEG_values(procdatafile)
    # 根据整数标签对数据和标签数组进行排序
    sorted_indices = np.argsort(integer_labels)
    sorted_data = arr_transposed[:, :, sorted_indices]
    third_dim_size = sorted_data.shape[2]
    features = Get_Features(third_dim_size, sorted_data)
    writer = csv.writer(
        open('TrainFeatures/Normalizedfeatures15.csv', 'w', newline=''))  # This file will store the normalized features
    writer.writerows(features)
    # dataNew = 'D:\\1xuexi\\daima\\naojijiekou\\autism\\启弘启惠\\data001features2.mat'
    # scio.savemat(dataNew, {'SHUJU':features2 })
main()