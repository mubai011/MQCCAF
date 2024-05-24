import scipy.io
import pandas as pd
import numpy as np


def read_mat(path):
    mat = scipy.io.loadmat(path)
    list_keys = list(mat.keys())
    DE_name = list_keys[3]
    data = mat[DE_name]
    data = pd.DataFrame(data)
    return data


def convert_data(data, Fault_type, types, save_path):
    """
    Input must be training data set and test data set,
    because the training data needs to do argumentation operations
    """
    data_samples = 200
    sample_size = 2048

    data_stride = int((len(data) - sample_size) / (data_samples - 1))
    pd1 = pd.DataFrame()
    start = 0
    end = 2048
    for i in range(data_samples):
        df = np.array(data)[start:end].ravel()
        pd1[i] = df
        start = start + data_stride
        end = start + sample_size
    data_convert = pd1.T
    data_convert[Fault_type] = types
    data_convert.to_csv(save_path)
    return data_convert
# health
base1 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-A-1.mat")
base2 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-A-2.mat")
base3 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-A-3.mat")
base4 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-B-1.mat")
base5 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-B-2.mat")
base6 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-B-3.mat")
base7 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-C-1.mat")
base8 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-C-2.mat")
base9 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-C-3.mat")
base10 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-D-1.mat")
base11 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-D-2.mat")
base12 = read_mat("./Ottawa/1 Data collected from a healthy bearing/H-D-3.mat")

# I

I1 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-A-1.mat")
I2 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-A-2.mat")
I3 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-A-3.mat")
I4 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-B-1.mat")
I5 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-B-2.mat")
I6 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-B-3.mat")
I7 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-C-1.mat")
I8 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-C-2.mat")
I9 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-C-3.mat")
I10 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-D-1.mat")
I11 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-D-2.mat")
I12 = read_mat("./Ottawa/2 Data collected from a bearing with inner race fault/I-D-3.mat")

# O
Or1 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-A-1.mat")
Or2 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-A-2.mat")
Or3 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-A-3.mat")
Or4 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-B-1.mat")
Or5 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-B-2.mat")
Or6 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-B-3.mat")
Or7 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-C-1.mat")
Or8 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-C-2.mat")
Or9 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-C-3.mat")
Or10 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-D-1.mat")
Or11 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-D-2.mat")
Or12 = read_mat("./Ottawa/3 Data collected from a bearing with outer race fault/O-D-3.mat")

# B
B1 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-A-1.mat")
B2 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-A-2.mat")
B3 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-A-3.mat")
B4 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-B-1.mat")
B5 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-B-2.mat")
B6 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-B-3.mat")
B7 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-C-1.mat")
B8 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-C-2.mat")
B9 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-C-3.mat")
B10 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-D-1.mat")
B11 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-D-2.mat")
B12 = read_mat("./Ottawa/4 Data collected from a bearing with ball fault/B-D-3.mat")

# C

C1 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-A-1.mat")
C2 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-A-2.mat")
C3 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-A-3.mat")
C4 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-B-1.mat")
C5 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-B-2.mat")
C6 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-B-3.mat")
C7 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-C-1.mat")
C8 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-C-2.mat")
C9 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-C-3.mat")
C10 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-D-1.mat")
C11 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-D-2.mat")
C12 = read_mat("./Ottawa/5 Data collected from a bearing with a combination of faults/C-D-3.mat")



base1_c = convert_data(base1, "fault", "0", "./Ottawa/convert/base1.csv")
base2_c = convert_data(base2, "fault", "0", "./Ottawa/convert/base2.csv")
base3_c = convert_data(base3, "fault", "0", "./Ottawa/convert/base3.csv")
base4_c = convert_data(base4, "fault", "0", "./Ottawa/convert/base4.csv")
base5_c = convert_data(base5, "fault", "0", "./Ottawa/convert/base5.csv")
base6_c = convert_data(base6, "fault", "0", "./Ottawa/convert/base6.csv")
base7_c = convert_data(base7, "fault", "0", "./Ottawa/convert/base7.csv")
base8_c = convert_data(base8, "fault", "0", "./Ottawa/convert/base8.csv")
base9_c = convert_data(base9, "fault", "0", "./Ottawa/convert/base9.csv")
base10_c = convert_data(base10, "fault", "0", "./Ottawa/convert/base10.csv")
base11_c = convert_data(base11, "fault", "0", "./Ottawa/convert/base11.csv")
base12_c = convert_data(base12, "fault", "0", "./Ottawa/convert/base12.csv")

Or1_c = convert_data(Or1, "fault", "1", "./Ottawa/convert/Or1.csv")
Or2_c = convert_data(Or2, "fault", "1", "./Ottawa/convert/Or2.csv")
Or3_c = convert_data(Or3, "fault", "1", "./Ottawa/convert/Or3.csv")
Or4_c = convert_data(Or4, "fault", "1", "./Ottawa/convert/Or4.csv")
Or5_c = convert_data(Or5, "fault", "1", "./Ottawa/convert/Or5.csv")
Or6_c = convert_data(Or6, "fault", "1", "./Ottawa/convert/Or6.csv")
Or7_c = convert_data(Or7, "fault", "1", "./Ottawa/convert/Or7.csv")
Or8_c = convert_data(Or8, "fault", "1", "./Ottawa/convert/Or8.csv")
Or9_c = convert_data(Or9, "fault", "1", "./Ottawa/convert/Or9.csv")
Or10_c = convert_data(Or10, "fault", "1", "./Ottawa/convert/Or10.csv")
Or11_c = convert_data(Or11, "fault", "1", "./Ottawa/convert/Or11.csv")
Or12_c = convert_data(Or12, "fault", "1", "./Ottawa/convert/Or12.csv")

I1_c = convert_data(I1, "fault", "2", "./Ottawa/convert/I1.csv")
I2_c = convert_data(I2, "fault", "2", "./Ottawa/convert/I2.csv")
I3_c = convert_data(I3, "fault", "2", "./Ottawa/convert/I3.csv")
I4_c = convert_data(I4, "fault", "2", "./Ottawa/convert/I4.csv")
I5_c = convert_data(I5, "fault", "2", "./Ottawa/convert/I5.csv")
I6_c = convert_data(I6, "fault", "2", "./Ottawa/convert/I6.csv")
I7_c = convert_data(I7, "fault", "2", "./Ottawa/convert/I7.csv")
I8_c = convert_data(I8, "fault", "2", "./Ottawa/convert/I8.csv")
I9_c = convert_data(I9, "fault", "2", "./Ottawa/convert/I9.csv")
I10_c = convert_data(I10, "fault", "2", "./Ottawa/convert/I10.csv")
I11_c = convert_data(I11, "fault", "2", "./Ottawa/convert/I11.csv")
I12_c = convert_data(I12, "fault", "2", "./Ottawa/convert/I12.csv")

B1_c = convert_data(B1, "fault", "3", "./Ottawa/convert/B1.csv")
B2_c = convert_data(B2, "fault", "3", "./Ottawa/convert/B2.csv")
B3_c = convert_data(B3, "fault", "3", "./Ottawa/convert/B3.csv")
B4_c = convert_data(B4, "fault", "3", "./Ottawa/convert/B4.csv")
B5_c = convert_data(B5, "fault", "3", "./Ottawa/convert/B5.csv")
B6_c = convert_data(B6, "fault", "3", "./Ottawa/convert/B6.csv")
B7_c = convert_data(B7, "fault", "3", "./Ottawa/convert/B7.csv")
B8_c = convert_data(B8, "fault", "3", "./Ottawa/convert/B8.csv")
B9_c = convert_data(B9, "fault", "3", "./Ottawa/convert/B9.csv")
B10_c = convert_data(B10, "fault", "3", "./Ottawa/convert/B10.csv")
B11_c = convert_data(B11, "fault", "3", "./Ottawa/convert/B11.csv")
B12_c = convert_data(B12, "fault", "3", "./Ottawa/convert/B12.csv")

C1_c = convert_data(C1, "fault", "4", "./Ottawa/convert/C1.csv")
C2_c = convert_data(C2, "fault", "4", "./Ottawa/convert/C2.csv")
C3_c = convert_data(C3, "fault", "4", "./Ottawa/convert/C3.csv")
C4_c = convert_data(C4, "fault", "4", "./Ottawa/convert/C4.csv")
C5_c = convert_data(C5, "fault", "4", "./Ottawa/convert/C5.csv")
C6_c = convert_data(C6, "fault", "4", "./Ottawa/convert/C6.csv")
C7_c = convert_data(C7, "fault", "4", "./Ottawa/convert/C7.csv")
C8_c = convert_data(C8, "fault", "4", "./Ottawa/convert/C8.csv")
C9_c = convert_data(C9, "fault", "4", "./Ottawa/convert/C9.csv")
C10_c = convert_data(C10, "fault", "4", "./Ottawa/convert/C10.csv")
C11_c = convert_data(C11, "fault", "4", "./Ottawa/convert/C11.csv")
C12_c = convert_data(C12, "fault", "4", "./Ottawa/convert/C12.csv")

He = np.concatenate(
    (base1_c, base2_c, base3_c, base4_c, base5_c, base6_c, base7_c, base8_c, base9_c, base10_c, base11_c, base12_c))

Or = np.concatenate((base1_c, base2_c, base3_c, Or4_c, Or5_c, Or6_c, Or7_c, Or8_c, Or9_c, Or10_c, Or11_c, Or12_c))
In = np.concatenate((I1_c, I2_c, I3_c, I4_c, I5_c, I6_c, I7_c, I8_c, I9_c, I10_c, I11_c, I12_c))
Ba = np.concatenate((B1_c, B2_c, B3_c, B4_c, B5_c, B6_c, B7_c, B8_c, B9_c, B10_c, B11_c, B12_c))
Co = np.concatenate((C1_c, C2_c, C3_c, C4_c, C5_c, C6_c, C7_c, C8_c, C9_c, C10_c, C11_c, C12_c))
E = np.concatenate((He, In, Or, Ba, Co))
# print(E.shape)

from sklearn.model_selection import train_test_split

Xe_train, Xe_test, Ye_train, Ye_test = train_test_split(E[:, 0:2048], E[:, 2048], test_size=0.2)

# # 保存Xe_train
np.save('./dealed_data/Ottawa/Xe_train.npy', Xe_train)

# 保存Xe_test
np.save('./dealed_data/Ottawa/Xe_test.npy', Xe_test)

# 保存Ye_train
np.save('./dealed_data/Ottawa/Ye_train.npy', Ye_train)

# 保存Ye_test
np.save('./dealed_data/Ottawa/Ye_test.npy', Ye_test)
