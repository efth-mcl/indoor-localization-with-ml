import subprocess
import json
import time
import numpy as np
import tensorflow as tf


class Iperf3Out:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def iperf3_udp_throughput(c, B, p, t=30, b_Mbps=32, R=False, zero_copy=False):
    iperf3cmd = 'iperf3 -c {} -B {} -p {} -u -t {} -b {}Mbps'.format(c, B, p, t, b_Mbps)
    if R:
        iperf3cmd += ' -R'
    if zero_copy:
        iperf3cmd += ' -Z'
    iperf3cmd += ' -J'

    iperf3cmd = iperf3cmd.split()
    iperf3cmd_out = subprocess.run(iperf3cmd, stdout=subprocess.PIPE)
    iperf3cmd_out = iperf3cmd_out.stdout.decode('utf-8')

    iperf3out2json = json.loads(iperf3cmd_out)
    return iperf3out2json


def iperf3_udp_obj_custom_avg(udp_obj):
    Mbps = []
    for inter in udp_obj.intervals:
        Mbps.append(
            inter['sum']['bits_per_second'] / (10 ** 6)
        )
    return np.mean(Mbps)


def full_mesurment_to_tf(secs_before_start_mesure=5):
    time.sleep(secs_before_start_mesure)
    # 2c -- 3s
    udp3 = iperf3_udp_throughput('10.42.0.3', '10.42.0.2', 6262, 24, 40, R=True)
    udp3 = Iperf3Out(**udp3)

    time.sleep(5)
    # 2c -- 14s
    udp14 = iperf3_udp_throughput('10.42.0.14', '10.42.0.2', 6262, 24, 40, R=True)
    udp14 = Iperf3Out(**udp14)

    time.sleep(5)
    # 2c -- 12s
    udp12 = iperf3_udp_throughput('10.42.0.12', '10.42.0.2', 6262, 24, 40, R=True)
    udp12 = Iperf3Out(**udp12)

    time.sleep(5)
    # 2c -- 7s
    udp7 = iperf3_udp_throughput('10.42.0.7', '10.42.0.2', 6262, 24, 40, R=True)
    udp7 = Iperf3Out(**udp7)

    #####
    udp3Mbps = [inter['sum']['bits_per_second'] / (10 ** 6) for inter in udp3.intervals]
    udp12Mbps = [inter['sum']['bits_per_second'] / (10 ** 6) for inter in udp12.intervals]
    udp7Mbps = [inter['sum']['bits_per_second'] / (10 ** 6) for inter in udp7.intervals]
    udp14Mbps = [inter['sum']['bits_per_second'] / (10 ** 6) for inter in udp14.intervals]

    udparray = np.array([udp3Mbps, udp7Mbps, udp12Mbps, udp14Mbps]).T

    tensor = tf.cast(udparray, tf.float32)
    return tensor

if __name__ == "__main__":
    import doctest
    doctest.testmod()