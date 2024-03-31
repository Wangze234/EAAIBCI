import numpy as np
# 获取排序后的索引
import scipy.stats as sp
import scipy.io as scio
import csv

from scipy.signal import welch
# 包含标签的原始训练集数据
rawdataFile = 'F:\\EAAIBCI\\EAAIBCI\\Training set\\Data1.mat'
# eeglab处理好之后的训练集数据
procdatafile = 'F:\\EAAIBCI\\EAAIBCI\\TraindataAfilter\\6\\Sub01-data.mat'
# 获取原始数据
rawdata = scio.loadmat(rawdataFile)
# 获取原始数据中的label
traindataraw = rawdata['epo_train']
label = np.double(traindataraw[0][0][5])
label = np.transpose(label)
# 将 one-hot 编码的标签数组转换为整数标签
integer_labels = np.argmax(label, axis=1)


# 获取预处理滤波后的数据
processdata = scio.loadmat(procdatafile)
processdata = processdata['EEG']
traindataProc = np.double(processdata[0][0][15])
# 获取滤波后的脑电数据
arr_transposed = np.transpose(traindataProc, (1, 0, 2))


# 根据整数标签对数据和标签数组进行排序
sorted_indices = np.argsort(integer_labels)
sorted_data = arr_transposed[:, :, sorted_indices]


def hjorth(input):                                             # function for hjorth
    # realinput = input
    # hjorth_activity = np.zeros(len(realinput))
    # hjorth_mobility = np.zeros(len(realinput))
    # hjorth_diffmobility = np.zeros(len(realinput))
    # hjorth_complexity = np.zeros(len(realinput))
    # diff_input = np.diff(realinput)
    # diff_diffinput = np.diff(diff_input)
    # k = 0
    # for j in realinput:
    #     hjorth_activity[k] = np.var(j)
    #     hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
    #     hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
    #     hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
    #     k = k+1
    # return np.sum(hjorth_activity)/8, np.sum(hjorth_mobility)/8, np.sum(hjorth_complexity)/8
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
def my_kurtosis(a):
    b = a # Extracting the data from the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    k = 0; # For counting the current row no.
    for i in b:
        mean_i = np.mean(i) # Saving the mean of array i
        std_i = np.std(i) # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j-mean_i)/std_i,4)-3)
        kurtosis_i = t/len(i) # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return np.sum(output)/14
def skewness(arr):

        data = arr
        skew_array = np.zeros(len(data))  # Initialinling the array as all 0s
        index = 0;  # current cell position in the output array

        for i in data:
            skew_array[index] = sp.skew(i, axis=0, bias=True)
            index += 1  # updating the cell position
        return np.sum(skew_array) / 8
def coeff_var(a):
    b = a #Extracting the data from the 14 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for i in b:
        mean_i = np.mean(i) #Saving the mean of array i
        std_i = np.std(i) #Saving the standard deviation of array i
        output[k] = std_i/mean_i #computing coefficient of variation
        k=k+1
    return np.sum(output)/8
def maxPwelch(data_win, Fs):
    BandF = [0.1, 3, 7, 12, 30]
    # BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([8, (len(BandF) - 1)]);

    for j in range(8):
        f, Psd = welch(data_win[j, :], Fs)
        for i in range(len(BandF) - 1):
            fr = np.where((f > BandF[i]) & (f <= BandF[i + 1]))
            PMax[j, i] = np.max(Psd[fr])

    return np.sum(PMax[:, 0]) / 8, np.sum(PMax[:, 1]) / 8, np.sum(PMax[:, 2]) / 8, np.sum(PMax[:, 3]) / 8
def main():
    third_dim_size =sorted_data.shape[2]
    second_dim_size = sorted_data.shape[1]
    features2 = []

    for i in range(third_dim_size):

        hjorthfeatures = hjorth(sorted_data[:, :, i])
        # my_kurtosisfeatures = my_kurtosis(sorted_data[:, :, i])
        # wrapper2features = skewness(sorted_data[:, :, i])
        # coeff_varfeatures = coeff_var(sorted_data[:, :, i])
        # maxPwelchfeatures = maxPwelch(sorted_data[:, :, i], 256)
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

    writer = csv.writer(
        open('TrainFeatures/Normalizedfeatures1-2.csv', 'w', newline=''))  # This file will store the normalized features
    writer.writerows(lines)
    # dataNew = 'D:\\1xuexi\\daima\\naojijiekou\\autism\\启弘启惠\\data001features2.mat'
    # scio.savemat(dataNew, {'SHUJU':features2 })
main()