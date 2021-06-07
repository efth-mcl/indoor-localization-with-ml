import json
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from thesispack.methods.graphs import list2graph
from thesispack.methods.nn import n_identity_matrix
from pandas import read_csv
from io import StringIO
from tensorflow.keras.preprocessing.sequence import pad_sequences

TRAIN_PATH = "train_data.json"

TEST_PATH = "test_data.json"


GAE_dataset ={
    'Labels': ['0', '1', '2', '3', '4', '5'],
    'listgraph': [[
        [0,1],
        [1,2],
        [1,3],
        [1,4],
        [1,5]
    ]]
}

SGAE_dataset ={
    'Labels': ['moving_divice', 'static0', 'static1', 'static2', 'static3'],
    'route_links': [[
        [0,1],
        [0,2],
        [0,3],
        [0,4]
    ]]
}

WIFI_RTT_PATH = "Data.zip"

class GaeDataset(object):
    def __init__(self):
       X, A, Atld = list2graph(GAE_dataset['listgraph'])

       self.train = tf.data.Dataset.from_tensor_slices((
            X, Atld, A
        )).batch(128)


class ZeroShotDataset(object):
    def __init__(self, embs_id, pathroot='..'):
       self.__a_mlb = MultiLabelBinarizer()
       self.__r_lb = LabelBinarizer()
       self.__pathroot = pathroot
       self.__r_emb_dict = np.load('{}/data/gae/gae-node-embs{}.npy'.format(self.__pathroot, embs_id), allow_pickle=True)[()]

    def load_data(self, seen_labels, unseen_labels):

        self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train = self.__load_data(TRAIN_PATH, seen_labels)
        self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val = self.__load_data(TEST_PATH, seen_labels)
        self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test = self.__load_merge_data(TRAIN_PATH, TEST_PATH, unseen_labels)

        avg = np.mean(self.__x_train)
        std = np.std(self.__x_train)

        self.__x_train = (self.__x_train-avg)/std
        self.__x_val = (self.__x_val-avg)/std
        self.__x_test = (self.__x_test-avg)/std

        self.__a_mlb.fit(self.__a_train)
        self.__a_train = self.__a_mlb.transform(self.__a_train)
        self.__a_test = self.__a_mlb.transform(self.__a_test)
        self.__a_val = self.__a_mlb.transform(self.__a_val)

        self.__r_lb.fit(self.__r_train+self.__r_test)
        self.__r_train = self.__r_lb.transform(self.__r_train)
        self.__r_val = self.__r_lb.transform(self.__r_val)
        self.__r_test = self.__r_lb.transform(self.__r_test)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_train, tf.float32),
            tf.cast(self.__r_emb_train, tf.float32),
            tf.cast(self.__r_train,tf.int32),
            tf.cast(self.__a_train, tf.int32)
        )).batch(128)

        self.val = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_val, tf.float32),
            tf.cast(self.__r_emb_val, tf.float32),
            tf.cast(self.__r_val, tf.int32),
            tf.cast(self.__a_val, tf.int32)
        )).batch(128)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_test, tf.float32),
            tf.cast(self.__r_emb_test, tf.float32),
            tf.cast(self.__r_test, tf.int32),
            tf.cast(self.__a_test, tf.int32)
        )).batch(128)

    def __load_data(self,path,labels):
        x, r, a = self.__load_file(path, labels)
        r_emb = []
        for r_label in r:
            r_emb.append(self.__r_emb_dict[r_label])
        
        r_emb = np.array(r_emb)
        return x, r_emb, r, a

    def __load_merge_data(self, train_path, test_path, labels):
        x1, r_emb1, r1, a1 = self.__load_data(train_path, labels)
        x2, r_emb2, r2, a2 = self.__load_data(test_path, labels)
        x = np.concatenate([x1, x2],axis=0)
        r = r1 + r2
        a = a1 + a2
        r_emb = np.concatenate([r_emb1, r_emb2],axis=0)

        return x, r_emb, r, a
    
    def __load_file(self, path, labels):
            # read file
            with open('{}/data/udp_dataset/{}'.format(self.__pathroot, path), 'r') as json_file:
                raw_data=json_file.read()

            # parse file
            obj= json.loads(raw_data)
            x = []
            r = []
            a = []
            for _, examples in obj.items():
                for key, example in examples.items():
                    if example["attribute"]["room"] in labels:
                        r.append(example["attribute"]["room"])
                        a.append([example["attribute"]["action"]])
                        udps = []
                        for _, data in example["udp"].items():
                            udps.append([])
                            data = list(data.values())
                            udps[-1].append(data)
                        udps = np.array(udps).T
                        x.append(udps)
            
            x = np.array(x).reshape(-1,24,4)
            return x, r, a
    
    def get_r_embs_dict(self):
        return self.__r_emb_dict
    
    def get_r_labels(self):
        return self.__r_lb.classes_
    
    def get_a_labels(self):
        return self.__a_mlb.classes_
    
    def get_r_onehot_from_labels(self,labels):
        return self.__r_lb.transform(labels)
    
    def get_a_onehot_from_labels(self, labels):
        return self.__a_mlb.transform(labels)
    
    def get_datatset(self):
        return (self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train), (self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val), (self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test)



class ENNDataset(ZeroShotDataset):
    def __init__(self, embid='32', pathroot='..', seen_labels=['0', '2'], unseen_labels=['1']):
        super(ENNDataset, self).__init__(embid, pathroot)
        self.load_data(seen_labels, unseen_labels)
        X, A, Atld = list2graph(SGAE_dataset['route_links'])
        Xpr = self.Pr0(Atld, X)

        (self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train), (self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val), (self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test) = super(ENNDataset, self).get_datatset()

        Ntrian = self.__x_train.shape[0]
        self.__X_train = np.concatenate(Ntrian*[Xpr],axis=0)
        self.__Atld_train = np.concatenate(Ntrian*[Atld], axis=0)
        self.__A_train = np.concatenate(Ntrian*[A],axis=0)
        Nval = self.__x_val.shape[0]
        self.__X_val = np.concatenate(Nval*[Xpr],axis=0)
        self.__Atld_val = np.concatenate(Nval*[Atld], axis=0)
        self.__A_val = np.concatenate(Nval*[A],axis=0)
        Ntest =self.__x_test.shape[0]
        self.__X_test = np.concatenate(Ntest*[Xpr],axis=0)
        self.__Atld_test = np.concatenate(Ntest*[Atld], axis=0)
        self.__A_test = np.concatenate(Ntest*[A],axis=0)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__X_train, tf.float32),
            tf.cast(self.__Atld_train, tf.float32),
            tf.cast(self.__x_train, tf.float32),

            tf.cast(self.__A_train, tf.float32),

            tf.cast(self.__r_emb_train, tf.float32),
            tf.cast(self.__r_train, tf.int8),
            tf.cast(self.__a_train, tf.int8)
        )).batch(128)

        self.val = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__X_val, tf.float32),
            tf.cast(self.__Atld_val, tf.float32),
            tf.cast(self.__x_val, tf.float32),

            tf.cast(self.__A_val, tf.float32),

            tf.cast(self.__r_emb_val, tf.float32),
            tf.cast(self.__r_val, tf.int8),
            tf.cast(self.__a_val, tf.int8)
        )).batch(128)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__X_test, tf.float32),
            tf.cast(self.__Atld_test, tf.float32),
            tf.cast(self.__x_test, tf.float32),

            tf.cast(self.__A_test, tf.float32),

            tf.cast(self.__r_emb_test, tf.float32),
            tf.cast(self.__r_test, tf.int8),
            tf.cast(self.__a_test, tf.int8)
        )).batch(128)


    def get_datatset(self):
        return (self.__X_train, self.__Atld_train, self.__x_train, self.__A_train, self.__r_emb_train, self.__r_train, self.__a_train
        ), (self.__X_val, self.__Atld_val, self.__x_val, self.__A_val, self.__r_emb_val, self.__r_val, self.__a_val
            ), (self.__X_test, self.__Atld_test, self.__x_test, self.__A_test, self.__r_emb_test, self.__r_test, self.__a_test
            )
    
    def Pr0(self, Atld, X):
        I0 = n_identity_matrix(Atld.shape[0])
        I2 = n_identity_matrix(Atld.shape[2])
        Xpr = tf.tensordot(I0, Atld, [[1],[0]])
        Xpr = tf.tensordot(Xpr, I2, [[3],[1]])
        Xpr = tf.tensordot(Xpr, X, [[1,4],[0,1]])
        return Xpr


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