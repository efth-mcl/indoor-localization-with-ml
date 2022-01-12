# Core thesis work (Indoor Localization)

The core work have two subnetworks and at the end we give an example of merging. At first, we do experiments with UDP 
throughput on network topologies: line, square and topology by BABEL routing protocol. After that we visualize the 
experimental results. For Indoor Localization task we train at first Graph Auto Encoder (GAE) for deep features 
extraction, and finally we train a stacked Recreant Neural Network (RNN) for Zero-Shot learning  task using the deep 
features (code) by GAE and train an Extended Neural Network for merging and generalize the problem of moving device on 
a Mesh Network. Optionally we do an experimental training on Graph Neural Network for Classification task for types 
of graphs.

## Hardware Configuration

We use 4 Raspberry Pi 3 (RPI) with Raspbian OS all RPIs has an external Wi-Fi adapter type [bale link edo](https://blb)
which are used for communication on ad-hoc network. The internal antenna uses for communication with controller device
where we execute the experiments on centralized network using an Access Point (router). Also, we use for Machine Learning
tasks a GPU to accelerate the computational routines.

### Installation

1. Install Raspbian OS, we use [Raspberry Pi Imager](link) to install OS on SD card, and after that we create an empty 
   file on **boot folder** with name `ssh` with **no** file extension with purpose to connect by controller device
   to RPIs via SSH protocol.
2. Paste the `interfaces` file:
   > `$ sudo cp ./RPIs/RPI{i}/interfaces /path/to/sd/card/manage?/etc/network` \
   >  where `i` is the number of RPI.
3. Wi-Fi adapter drivers:
   > `$ sudo wget http://www.fars-robotics.net/install-wifi -O /usr/bin/install-wifi` \
   > `$ sudo chmod +x /usr/bin/install-wifi` \
   > `$ sudo install-wifi` \
4. BABEL routing protocol:
   > `$ git clone https://github.com/jech/babeld.git` \
   > `$ cd babeld` \
   > `$ make` \
   > `$ sudo make install` \
   > `# check if works` \
   > `$ babeld -V` \
   > `# stop BABEL` \
   > `$ sudo killall babeld` \
   > `# configure BABEL` \
   > Use `./RPIs/RPI{i}/babeld.conf` (where `i` is the number of RPI). \
   > `# sart BABEL` \
   > `$ sudo babeld -h 2 -D -c /path/to/babeld.conf -L path/to/babeld.log -d 2 wlan0`, \
   > where `-h` is the time interval in seconds, `-D` run as demon, `-c` set the configuration file, `-L` set the log 
   > file, `-d` the level of details of log file and `wlan0` the interface (in our case the external adapter). \
   > Optionally use the at start up the BABEL protocol editing the `/etv/rc.local` file adding at the end \
   > `$ sudo babeld ...`


## Routing Experiments
We use [Iperf3](ipefer3_link) tool.

### Routing analysis
In this part we visualize the collection of data for UDP Throughput experiments on a variety of
Transition Power ${10, 20, 31} dBm$ for $60 sec.$ we set the RPI0 as client and the rest as servers.
we save the results in [data/routing](data/routing) folder. We visualize the experiment results on notbook:
>[routing analysis](routinganalisis.ipynb).

### UDP Throughput as Metric
In this part we collect UDP examples for $24 sec.$. We set as mobile device the controller
device as server and static devices, the RPIs as clients. The measurements are serial (one by another). The Computer Network 
topology is a Star with center the controller(Laptop) and RPIs in the edges. The following is the notebook where we use:
> [udp metric collection notebook](collect-udp-metric.ipynb).

## Machine Learning Systems for Indoor Localization

### 2ORNN System
1. We train a GAE for deep feature extraction (aka Graph Embeddings)
   > [train gae notebook](train-gae.ipynb)
2. We train a stacked RNN based on LSTM with 2 outputs one the moving status (moving/stading) and space location
   > [train 2ornn notebook](train-2ornn.ipynb)

### Extended NN System
An Extended Neural Network for dynamic routing where we have one mobile device.
> [train enn notebook](train-extnn.ipynb)

## GCN performance per depth
> [train depth gcn](gcn-depth.ipynb)