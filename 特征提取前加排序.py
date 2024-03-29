import numpy as np
# 获取排序后的索引
import scipy as sp
import scipy.io as scio
import csv

from scipy.signal import welch

labeldataFile = 'F:\\EAAIBCI\\Trainingset\\Data_Sample01.mat'
labeldata = scio.loadmat(labeldataFile)
traindata = labeldata['epo_train']
data = np.double(traindata[0][0][4])
label = np.double(traindata[0][0][5])
label = np.transpose(label)

# 将 one-hot 编码的标签数组转换为整数标签
integer_labels = np.argmax(label, axis=1)

# 根据整数标签对数据和标签数组进行排序
sorted_indices = np.argsort(integer_labels)
sorted_labels = label[sorted_indices]
sorted_data = data[:, :, sorted_indices]


def hjorth(input):                                             # function for hjorth
    realinput = input
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for j in realinput:
        hjorth_activity[k] = np.var(j)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
        k = k+1
    return np.sum(hjorth_activity)/8, np.sum(hjorth_mobility)/8, np.sum(hjorth_complexity)/8
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
    BandF = [0.1, 5, 7, 12, 30]
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
    features2 = []

    for i in range(third_dim_size):
        hjorthfeatures = hjorth(sorted_data[:, :, i])
        my_kurtosisfeatures = my_kurtosis(sorted_data[:, :, i])
        wrapper2features = skewness(sorted_data[:, :, i])
        coeff_varfeatures = coeff_var(sorted_data[:, :, i])
        maxPwelchfeatures = maxPwelch(sorted_data[:, :, i], 300)
        combined_features = np.hstack((hjorthfeatures, my_kurtosisfeatures,wrapper2features,coeff_varfeatures,maxPwelchfeatures))
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
        open('Features/Normalizedfeatures5.csv', 'w', newline=''))  # This file will store the normalized features
    writer.writerows(lines)
    # dataNew = 'D:\\1xuexi\\daima\\naojijiekou\\autism\\启弘启惠\\data001features2.mat'
    # scio.savemat(dataNew, {'SHUJU':features2 })
main()