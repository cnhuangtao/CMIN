### <center>Context-Aware Multi-Interest Network in Click-Through Rate Prediction</center>

#### Data Preparation
----

Download the public dataset [Amazon](https://nijianmo.github.io/amazon/index.html) to *./raw_data/*.

> python data_prepare.py

#### Model Train
----

> python train.py --is_part_test 0 --eval_per_num 1000 --batch_size 1024 --log_steps 100 --train_epochs 5 --dropout_rate 0.2 --deep_layers 128,64 --data_type book --model_type [model name]

The model blelow had been supported:

- LR
- PNN
- DIN
- DIEN
- DMIN
- MIAN
- CMIN
