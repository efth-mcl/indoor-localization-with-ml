import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv
from io import StringIO
from tensorflow.keras.preprocessing.sequence import pad_sequences

WIFI_RTT_PATH = "data/Data.zip"



class WIFIRTTDataset(object):
    def __init__(self):
        self.__df, self.__new_m_ind = self.__load_df(WIFI_RTT_PATH)

    def load_dataset(self, w, batch_size):

        def preprocess_dataset():
            tods_ap = []
            pos_xy = []
            spectrs_ap_ant1 = []
            spectrs_ap_ant2 = []

            for k in range(len(self.__new_m_ind) - 1):
                dfmini = self.__df.loc[self.__new_m_ind[k] + 2:self.__new_m_ind[k + 1] + 1]
                j = 0
                while (j + 1) * w <= len(dfmini):
                    data = dfmini.iloc[j * w:(j + 1) * w]
                    sp1_ap_ant = []
                    sp2_ap_ant = []
                    tod_ap = []
                    for i in range(1, 13):
                        data_api = data[data["AP_index"] == i].iloc

                        sp1_ap_ant.append(data_api[:, 11:125])
                        sp2_ap_ant.append(data_api[:, 125:])
                        tod_ap.append(data["ToD_factor[m]"])

                    tods_ap.append(tod_ap)
                    spectrs_ap_ant1.append(sp1_ap_ant)
                    spectrs_ap_ant2.append(sp2_ap_ant)

                    pos_xy.append(data[[
                        'GroundTruthPositionX[m]',
                        'GroundTruthPositionY[m]'
                    ]].iloc[-1])
                    j += 1

            n_d = len(spectrs_ap_ant1)
            spectrs_ap_ant1 = [pad_sequences(spectrs_ap_ant1[i], maxlen=w, padding="post", dtype='float') for i in
                               range(n_d)]

            spectrs_ap_ant2 = [pad_sequences(spectrs_ap_ant2[i], maxlen=w, padding="post", dtype='float') for i in
                               range(n_d)]

            tods_ap = [pad_sequences(tods_ap[i], maxlen=w, padding="post", dtype='float') for i in range(n_d)]

            spectrs_ap_ant1 = tf.transpose(np.array(spectrs_ap_ant1), perm=[0, 2, 1, 3]).numpy()
            spectrs_ap_ant2 = tf.transpose(np.array(spectrs_ap_ant2), perm=[0, 2, 1, 3]).numpy()
            tods_ap = tf.transpose(np.array(tods_ap), perm=[0, 2, 1]).numpy()
            pos_xy = np.array(pos_xy)

            return spectrs_ap_ant1, spectrs_ap_ant2, tods_ap, pos_xy, n_d

        self.__spectrs_ap_ant1, self.__spectrs_ap_ant2, self.__tods_ap, self.__pos_xy, n_d = preprocess_dataset()

        # random shuffle and split
        np.random.seed(5287231)
        r_sort_array = np.random.choice(np.arange(0, n_d), replace=False, size=(1, n_d)).reshape(n_d)
        tods_ap = self.__tods_ap[r_sort_array]
        ps_xy = self.__pos_xy[r_sort_array]
        spectrs_ap_ant1 = self.__spectrs_ap_ant1[r_sort_array]
        spectrs_ap_ant2 = self.__spectrs_ap_ant2[r_sort_array]

        train_tod, valtest_tod, train_spectr_ant1, valtest_spectr_ant1, train_spectr_ant2, valtest_spectr_ant2, train_pxy, valtest_pxy = train_test_split(
            tods_ap, spectrs_ap_ant1, spectrs_ap_ant2, ps_xy, test_size=0.5)

        val_tod, test_tod, val_spectr_ant1, test_spectr_ant1, val_spectr_ant2, test_spectr_ant2, val_pxy, test_pxy = train_test_split(
            valtest_tod, valtest_spectr_ant1, valtest_spectr_ant2, valtest_pxy, test_size=0.5)

        # min max normalization for position (target)
        self.__min_xy, self.__max_xy = (np.min(train_pxy, axis=0), np.max(train_pxy, axis=0))
        train_pxy = np.divide(train_pxy - self.__min_xy, self.__max_xy - self.__min_xy)
        val_pxy = np.divide(val_pxy - self.__min_xy, self.__max_xy - self.__min_xy)
        test_pxy = np.divide(test_pxy - self.__min_xy, self.__max_xy - self.__min_xy)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(train_tod, tf.float32),
            *[tf.cast(train_spectr_ant1[:, :, i], tf.float32) for i in range(12)],
            *[tf.cast(train_spectr_ant2[:, :, i], tf.float32) for i in range(12)],
            tf.cast(train_pxy, tf.float32),
        )).batch(batch_size)

        self.val = tf.data.Dataset.from_tensor_slices((
            tf.cast(val_tod, tf.float32),
            *[tf.cast(val_spectr_ant1[:, :, i], tf.float32) for i in range(12)],
            *[tf.cast(val_spectr_ant2[:, :, i], tf.float32) for i in range(12)],
            tf.cast(val_pxy, tf.float32),
        )).batch(batch_size)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(test_tod, tf.float32),
            *[tf.cast(test_spectr_ant1[:, :, i], tf.float32) for i in range(12)],
            *[tf.cast(test_spectr_ant2[:, :, i], tf.float32) for i in range(12)],
            tf.cast(test_pxy, tf.float32),
        )).batch(batch_size)

    def __load_df(self, data_path):
        df = read_csv(
            StringIO(
                str(
                    np.load('../data/{}'.format(data_path))['RTT_data.csv'],
                    'utf-8'
                )))

        new_m_ind = [0] + list(np.where(np.gradient(df['%Timestamp[s]']) < -750)[0][::2]) + [len(df)]
        df.iloc[:, 11:239] = df.iloc[:, 11:239].applymap(lambda x: np.abs(complex(x.replace('i', 'j'))))
        df = df.iloc[:, :239]

        return df, new_m_ind

    def get_df(self):
        return self.__df

    def get_new_m_ind(self):
        return self.__new_m_ind

    def get_prepro_dataset(self):
        return self.__spectrs_ap_ant1, self.__spectrs_ap_ant2, self.__tods_ap, self.__pos_xy

    def get_min_max_pos(self):
        return self.__min_xy, self.__max_xy