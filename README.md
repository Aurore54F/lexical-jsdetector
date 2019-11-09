# lexical-jsdetector

This repository contains Dennis Salzmann's reimplementation of the concept presented by [Rieck et al. with Cujo](https://www.sec.cs.tu-bs.de/pubs/2010-acsac.pdf) (static part only).
Please note that in its current state, the code is a Poc and not a fully-fledged production-ready API.


## Summary
In our reimplementation, we combined a lexical analysis of JavaScript inputs with machine learning algorithms to automatically and accurately detect malicious samples. 

## Setup

jassi installation:
```
cd src
git clone https://github.com/rieck/jassi.git
cd jassi
./bootstrap
./configure
make
cd ../..
```

python3 installations:
```
install python3
install python3-pip
pip3 install -r requirements.txt
```


## Usage

### Learning: Building a Model

To build a model from the folders BENIGN and MALICIOUS, containing JS files, use the option -ti BENIGN MALICIOUS and add their corresponding ground truth with -tc benign malicious. To save the model in the MODEL path, use the option --save_model MODEL:

```
$ python3 src/main.py -ti  BENIGN MALICIOUS -tc benign malicious --save_model MODEL
```


### Classification of JS Samples

The process is similar for the classification process.
To classify JS samples from the folders BENIGN2 and MALICIOUS2, use the option -ai BENIGN2 MALICIOUS2. Add their corresponding ground truth with the option -ac benign malicious. To load an existing model MODEL to be used for the classification process, use the option --load_model MODEL:

```
$ python3 src/main.py -ai  BENIGN2 MALICIOUS2 -ac benign malicious --load_model MODEL
```


## License

This project is licensed under the terms of the AGPL3 license, which you can find in ```LICENSE```.
