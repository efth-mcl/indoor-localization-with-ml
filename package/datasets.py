import json
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tensorflow import cast
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter, gaussian_filter1d

TRAIN_PATHS = [
    "train_data.json",
]

TEST_PATHS = [
    "test_data.json",

]

class ZeroShotDataset(object):
    def __init__(self):
       self.__a_mlb = MultiLabelBinarizer()
       self.__r_lb = LabelBinarizer()
       self.__r_emb_dict = np.load('../data/gae/gae-node-embs.npy', allow_pickle=True)[()]

    def load_data(self, train_paths, test_paths, seen_labels, unseen_labels):

        self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train = self.__load_data(train_paths[0], seen_labels)
        self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val = self.__load_data(test_paths[0], seen_labels)
        self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test = self.__load_merge_data(train_paths[0], test_paths[0], unseen_labels)

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
            tf.cast(self.__r_train,tf.int64),
            tf.cast(self.__a_train, tf.int64)
        )).batch(128)

        self.val = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_val, tf.float32),
            tf.cast(self.__r_emb_val, tf.float32),
            tf.cast(self.__r_val, tf.int64),
            tf.cast(self.__a_val, tf.int64)
        )).batch(128)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_test, tf.float32),
            tf.cast(self.__r_emb_test, tf.float32),
            tf.cast(self.__r_test, tf.int64),
            tf.cast(self.__a_test, tf.int64)
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
            with open('../data/dataset/'+path, 'r') as json_file:
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



class Dataset(object):
    # x: data
    # r: room
    # a: action
    def __init__(self,svd_size = 2):
        self.__r_mlb = MultiLabelBinarizer()
        self.__a_mlb = MultiLabelBinarizer()
        # self.__mlb = MultiLabelBinarizer()


    def load_data(self, train_paths, test_paths, sigma=0):
        
        # get data and labels
        self.__x_train, self.__r_train, self.__a_train  = self.__load_data(train_paths[0], sigma)
        self.__x_test, self.__r_test, self.__a_test  = self.__load_data(test_paths[0], sigma)

        # normalize data
        avg = np.mean(self.__x_train)
        std = np.std(self.__x_train)

        self.__x_train = (self.__x_train-avg)/std
        self.__x_test = (self.__x_test-avg)/std

        # to one hot encoding
        self.__r_mlb.fit(self.__r_train)
        self.__a_mlb.fit(self.__a_train)
        
        self.__r_train = self.__r_mlb.transform(self.__r_train)
        self.__r_test = self.__r_mlb.transform(self.__r_test)

        self.__a_train = self.__a_mlb.transform(self.__a_train)
        self.__a_test = self.__a_mlb.transform(self.__a_test)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_train, tf.float32),
            tf.cast(self.__r_train, tf.int64),
            tf.cast(self.__a_train, tf.int64)
        )).batch(128)

        # self.a_train = tf.data.Dataset.from_tensor_slices((
        #     tf.cast(self.__x_train, tf.float32),
        #     tf.cast(self.__a_train, tf.int64)
        # )).batch(128)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_test, tf.float32),
            tf.cast(self.__r_test, tf.int64),
            tf.cast(self.__a_test, tf.int64)
        )).batch(128)

        # self.a_test = tf.data.Dataset.from_tensor_slices((
        #     tf.cast(self.__x_test, tf.float32),
        #     tf.cast(self.__a_test, tf.int64)
        # )).batch(128)

    def __load_data(self, path, sigma):
        # read file
        with open('dataset_simple/'+path, 'r') as json_file:
            raw_data=json_file.read()

        # parse file
        obj= json.loads(raw_data)
        x = []
        r = []
        a = []
        # y = []
        for _, examples in obj.items():
            for key, example in examples.items():
                r.append([example["attribute"]["room"]])
                a.append([example["attribute"]["action"]])
                # y.append([
                #     example["attribute"]["room"],
                #     example["attribute"]["action"]
                # ])
                udps = []
                for _, data in example["udp"].items():
                    udps.append([])
                    data = list(data.values())
                    if sigma > 0:
                        data = gaussian_filter1d(data,sigma).tolist()
                    udps[-1].append(data)
                udps = np.array(udps).T
                x.append(udps)
        
        x = np.array(x).reshape(-1,24,4)
        return x, r, a
    
    def get_names(self):
        return self.__r_mlb.classes_, self.__a_mlb.classes_
    
    def get_names_from_mlb_data(self, mlbdata, t=0.5):
        try:
            mlbdata = mlbdata.numpy()
            
        except:
            pass
        
        
        
        mlbdata = (mlbdata > t).astype('int')

        return [[self.__mlb.classes_[index] for index, value in enumerate(l) if value == 1] for l in mlbdata.tolist()]

    # def get_rooms_names(self):
    #     return self.__r_mlb.classes_
    
    # def get_actions_names(self):
    #     return self.__a_mlb.classes_

    # def get_rooms_names_from_onehot_data(self, onehot):
    #     sparce = np.argmax(onehot,axis=1)
    #     return self.__r_mlb.classes_[sparce]
    
    # def get_actions_names_from_onehot_data(self, onehot):
    #     sparce = np.argmax(onehot,axis=1)
    #     return self.__a_mlb.classes_[sparce]

    def gen_data(self,tf_examples, gen_num=20):
        X = []
        Yr = []
        Ya = []

        cats = np.array([])
        for (batch, (_, labels_r, labels_a)) in enumerate(tf_examples):
            if batch == 0:
                cats = np.concatenate((labels_r, labels_a),axis=1)
            else:
                cats = np.concatenate((cats,labels_r, labels_a),axis=1)

        cats_u = np.unique(cats,axis=0)
        inputs_per_cat = []
        for (batch, (inputs, labels_r, labels_a)) in enumerate(tf_examples):
            cat = np.concatenate((labels_r, labels_a),axis=1)
            in_per_cat = []
            for cu in cats_u:
                in_per_cat.append(
                    inputs.numpy()[np.where((cat==cu).sum(axis=1)==6)]
                )
            inputs_per_cat.append(in_per_cat)

        inputs_per_cat = np.array(inputs_per_cat)
        inputs_per_cat = inputs_per_cat.reshape(*inputs_per_cat.shape[1:])
        mean = np.mean(inputs_per_cat,axis=(1,2))
        std = np.std(inputs_per_cat,axis=(1,2))
        for m,s,c in zip(mean, std, cats_u):
            gen_d = np.random.normal(m,s,(gen_num,24,4))
            X.append(gen_d)
            Yr.append(
                np.tile(c[:4],(gen_num,1)).tolist()
            )
            Ya.append(
                np.tile(c[4:],(gen_num,1)).tolist()
            )

        X = np.array(X)
        X = X.reshape(X.shape[0]*X.shape[1],24,4)
        Yr = np.array(Yr)
        Yr = Yr.reshape(Yr.shape[0]*Yr.shape[1],4)

        Ya = np.array(Ya)
        Ya = Ya.reshape(Ya.shape[0]*Ya.shape[1],2)

        # random sort
        rand_sort = np.arange(X.shape[0])
        np.random.shuffle(rand_sort)
        X = X[rand_sort].reshape(*X.shape)
        Yr = Yr[rand_sort].reshape(*Yr.shape)
        Ya = Ya[rand_sort].reshape(*Ya.shape)

        new_data = tf.data.Dataset.from_tensor_slices((
            tf.cast(X, tf.float32),
            tf.cast(Yr, tf.int64),
            tf.cast(Ya, tf.int64)
        )).batch(128)

        return new_data