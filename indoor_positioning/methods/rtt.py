import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def tod_and_spectrum_per_ap_with_pad(df, new_m_ind, dt):
    max_pad = 0
    tods_ap = []
    spectrs_ap = []
    for i in range(len(new_m_ind)-1):
        dfmini = df.loc[new_m_ind[i]+1:new_m_ind[i+1]]

        mint = dfmini["%Timestamp[s]"].min()
        dfmini["%Timestamp[s]"] += -mint
        widx = 0
        while True:
            tod_ap = []
            spectr_ap = []
            data = dfmini[(dfmini["%Timestamp[s]"] <= dt*(widx+1)) & (dfmini["%Timestamp[s]"] > dt*widx)].iloc[
                :, :57+11
            ]
            if len(data) == 0:
                break
            for i in range(1, 13):
                tod_ap.append(data[data["AP_index"] == i]["ToD_factor[m]"].to_numpy())
                spectr_ap.append(data[data["AP_index"] == i].iloc[:, 11:].applymap(lambda x: np.abs(complex(x.replace('i', 'j')))).to_numpy())

                if len(tod_ap[-1]) > max_pad:
                    max_pad = len(tod_ap[-1])

            tods_ap.append(tod_ap)
            spectrs_ap.append(spectr_ap)
            widx += 1

    for i in range(len(tods_ap)):
        tods_ap[i] = pad_sequences(tods_ap[i], padding="post", maxlen=max_pad)
        spectrs_ap[i] = pad_sequences(spectrs_ap[i], padding="post", maxlen=max_pad)
    tods_ap = tf.transpose(np.array(tods_ap), perm=[0,2,1]).numpy()
    spectrs_ap = tf.transpose(np.array(spectrs_ap), perm=[0,2,1,3]).numpy()
    return tods_ap, spectrs_ap


if __name__ == "__main__":
    import doctest
    doctest.testmod()