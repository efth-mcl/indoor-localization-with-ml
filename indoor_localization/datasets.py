from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from algomorphism.method.graph import vertexes2adjacency
from algomorphism.method.opt import three_d_identity_matrix
from algomorphism.dataset.graph.base import GraphBaseDataset
from algomorphism.dataset.zeroshot.base import SeenUnseenBase
import tensorflow as tf
import numpy as np
import json



TRAIN_PATH = "train_data.json"

TEST_PATH = "test_data.json"


GAE_dataset ={
    'Labels': ['0', '1', '2', '3', '4', '5'],
    'listgraph': ([
        [0, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5]
    ])
}

SGAE_dataset ={
    'Labels': ['moving_device', 'static0', 'static1', 'static2', 'static3'],
    'route_links': [[
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4]
    ]]
}


class GaeDataset(GraphBaseDataset):
    def __init__(self):
        super(GaeDataset, self).__init__()
        a = vertexes2adjacency(GAE_dataset['listgraph'])
        a = a.reshape(1, a.shape[0], a.shape[1])
        a += np.eye(a.shape[-1])
        atld = self.renormalization(a)
        x = np.eye(a.shape[1]).reshape(1, a.shape[1], a.shape[2])

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(x, tf.float32),
            tf.cast(atld, tf.float32),
            tf.cast(a, tf.float32)
        )).batch(128)


class ZeroShotDataset(object):
    def __init__(self, embs_id, pathroot='.'):
        self.__a_mlb = MultiLabelBinarizer()
        self.__r_lb = LabelBinarizer()
        self.__pathroot = pathroot
        self.__r_emb_dict = np.load('{}/data/gae/gae-node-embs{}.npy'.format(self.__pathroot, embs_id), allow_pickle=True)[()]

    def r_emb_disct_preprosesing(self, seen_labels, unseen_labels):

        tag_embs_seen = []
        for k, v in self.__r_emb_dict.items():
            if k in seen_labels:
                tag_embs_seen.append(v)
        tag_embs_seen = np.array(tag_embs_seen)

        tag_embs_unseen = []
        for k, v in self.__r_emb_dict.items():
            if k in unseen_labels:
                tag_embs_unseen.append(v)
        tag_embs_unseen = np.array(tag_embs_unseen)

        c = np.matmul(tag_embs_seen, np.transpose(tag_embs_seen))
        # c = np.round(c)
        c = np.linalg.inv(c)
        c = np.matmul(c, tag_embs_seen)
        c = np.matmul(np.transpose(tag_embs_seen), c)

        tag_embs_seen_y = np.matmul(tag_embs_seen, c)
        tag_embs_unseen_y = np.matmul(tag_embs_unseen, c)

        r_emb_dict_pr = {}
        for k, v in zip(seen_labels, tag_embs_seen_y):
            r_emb_dict_pr[k] = v
        for k, v in zip(unseen_labels, tag_embs_unseen_y):
            r_emb_dict_pr[k] = v

        return r_emb_dict_pr

    def load_data(self, seen_labels, unseen_labels):
        r_emb_dict_pr = self.r_emb_disct_preprosesing(seen_labels, unseen_labels)

        self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train = self.__load_data(TRAIN_PATH, seen_labels, r_emb_dict_pr)
        self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val = self.__load_data(TEST_PATH, seen_labels, r_emb_dict_pr)
        self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test = self.__load_merge_data(TRAIN_PATH, TEST_PATH,
                                                                                                unseen_labels, r_emb_dict_pr)

        avg = np.mean(self.__x_train)
        std = np.std(self.__x_train)

        self.__x_train = (self.__x_train - avg) / std
        self.__x_val = (self.__x_val - avg) / std
        self.__x_test = (self.__x_test - avg) / std

        self.__a_mlb.fit(self.__a_train)
        self.__a_train = self.__a_mlb.transform(self.__a_train)
        self.__a_test = self.__a_mlb.transform(self.__a_test)
        self.__a_val = self.__a_mlb.transform(self.__a_val)

        self.__r_lb.fit(self.__r_train + self.__r_test)
        self.__r_train = self.__r_lb.transform(self.__r_train)
        self.__r_val = self.__r_lb.transform(self.__r_val)
        self.__r_test = self.__r_lb.transform(self.__r_test)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_train, tf.float32),
            tf.cast(self.__r_emb_train, tf.float32),
            tf.cast(self.__r_train, tf.int32),
            tf.cast(self.__a_train, tf.int32)
        )).batch(128)

        self.val = SeenUnseenBase(
            tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_val, tf.float32),
            tf.cast(self.__r_emb_val, tf.float32),
            tf.cast(self.__r_val, tf.int32),
            tf.cast(self.__a_val, tf.int32)
        )).batch(128),
        tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__x_test, tf.float32),
            tf.cast(self.__r_emb_test, tf.float32),
            tf.cast(self.__r_test, tf.int32),
            tf.cast(self.__a_test, tf.int32)
        )).batch(128))

    def __load_data(self, path, labels, r_emb_dict):
        x, r, a = self.__load_file(path, labels)
        r_emb = []
        for r_label in r:
            r_emb.append(r_emb_dict[r_label])

        r_emb = np.array(r_emb)
        return x, r_emb, r, a

    def __load_merge_data(self, train_path, test_path, labels, r_emb_dict):
        x1, r_emb1, r1, a1 = self.__load_data(train_path, labels, r_emb_dict)
        x2, r_emb2, r2, a2 = self.__load_data(test_path, labels, r_emb_dict)
        x = np.concatenate([x1, x2], axis=0)
        r = r1 + r2
        a = a1 + a2
        r_emb = np.concatenate([r_emb1, r_emb2], axis=0)

        return x, r_emb, r, a

    def __load_file(self, path, labels):
        # read file
        with open('{}/data/udp_dataset/{}'.format(self.__pathroot, path), 'r') as json_file:
            raw_data = json_file.read()

        # parse file
        obj = json.loads(raw_data)
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

        x = np.array(x).reshape(-1, 24, 4)
        return x, r, a

    def get_r_embs_dict(self):
        return self.__r_emb_dict

    def get_r_labels(self):
        return self.__r_lb.classes_

    def get_a_labels(self):
        return self.__a_mlb.classes_

    def get_r_onehot_from_labels(self, labels):
        return self.__r_lb.transform(labels)

    def get_a_onehot_from_labels(self, labels):
        return self.__a_mlb.transform(labels)

    def get_datatset(self):
        return (self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train), (
        self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val), (
               self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test)


class ENNDataset(ZeroShotDataset, GraphBaseDataset):
    def __init__(self, embid='32', pathroot='.', seen_labels=['0', '2'], unseen_labels=['1']):
        super(ENNDataset, self).__init__(embid, pathroot)
        self.load_data(seen_labels, unseen_labels)
        adj = vertexes2adjacency(SGAE_dataset['route_links'][0])
        adj = adj.reshape(1, adj.shape[0], adj.shape[1])
        adj += np.eye(adj.shape[-1])
        atld = self.renormalization(adj)
        xf = np.eye(adj.shape[1]).reshape(1, adj.shape[1], adj.shape[2])
        xpr = self.Pr0(atld, xf)

        (self.__x_train, self.__r_emb_train, self.__r_train, self.__a_train), (
        self.__x_val, self.__r_emb_val, self.__r_val, self.__a_val), (
        self.__x_test, self.__r_emb_test, self.__r_test, self.__a_test) = super(ENNDataset, self).get_datatset()

        Ntrian = self.__x_train.shape[0]
        self.__xpr_train = np.concatenate(Ntrian * [xpr], axis=0)
        self.__atld_train = np.concatenate(Ntrian * [atld], axis=0)
        self.__adj_train = np.concatenate(Ntrian * [adj], axis=0)
        Nval = self.__x_val.shape[0]
        self.__xpr_val = np.concatenate(Nval * [xpr], axis=0)
        self.__atld_val = np.concatenate(Nval * [atld], axis=0)
        self.__adj_val = np.concatenate(Nval * [adj], axis=0)
        Ntest = self.__x_test.shape[0]
        self.__xpr_test = np.concatenate(Ntest * [xpr], axis=0)
        self.__atld_test = np.concatenate(Ntest * [atld], axis=0)
        self.__adj_test = np.concatenate(Ntest * [adj], axis=0)

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__xpr_train, tf.float32),
            tf.cast(self.__atld_train, tf.float32),
            tf.cast(self.__x_train, tf.float32),

            tf.cast(self.__adj_train, tf.float32),

            tf.cast(self.__r_emb_train, tf.float32),
            tf.cast(self.__r_train, tf.float32),
            tf.cast(self.__a_train, tf.float32)
        )).batch(128)

        self.val = SeenUnseenBase(
            tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__xpr_val, tf.float32),
            tf.cast(self.__atld_val, tf.float32),
            tf.cast(self.__x_val, tf.float32),

            tf.cast(self.__adj_val, tf.float32),

            tf.cast(self.__r_emb_val, tf.float32),
            tf.cast(self.__r_val, tf.float32),
            tf.cast(self.__a_val, tf.float32)
        )).batch(128),
        tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__xpr_test, tf.float32),
            tf.cast(self.__atld_test, tf.float32),
            tf.cast(self.__x_test, tf.float32),

            tf.cast(self.__adj_test, tf.float32),

            tf.cast(self.__r_emb_test, tf.float32),
            tf.cast(self.__r_test, tf.float32),
            tf.cast(self.__a_test, tf.float32)
        )).batch(128)
        )

    def get_datatset(self):
        return (self.__x_train, self.__atld_train, self.__x_train, self.__a_train, self.__r_emb_train, self.__r_train,
                self.__a_train
                ), (
               self.__x_val, self.__atld_val, self.__x_val, self.__a_val, self.__r_emb_val, self.__r_val, self.__a_val
               ), (self.__x_test, self.__atld_test, self.__x_test, self.__a_test, self.__r_emb_test, self.__r_test,
                   self.__a_test
                   )

    def Pr0(self, atld, x):
        I0 = three_d_identity_matrix(atld.shape[0])
        I2 = three_d_identity_matrix(atld.shape[2])
        xpr = np.tensordot(I0, atld, [[1], [0]])
        xpr = np.tensordot(xpr, I2, [[3], [1]])
        xpr = np.tensordot(xpr, x, [[1, 4], [0, 1]])
        return xpr
