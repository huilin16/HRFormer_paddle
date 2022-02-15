# HRFormer_paddle

An paddle implementation of  NeurIPS 2021 paper "HRFormer: High-Resolution Transformer for Dense Prediction".

This project is based on [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

Reproduction results：

* Top-1 acc.：77.3

Reroduction process:

```
python -m paddle.distributed.launch $PDCS_DIR/tools/train.py \
    -c $PDCS_DIR/hrt.yml \
    -o Global.device=gpu \
    -o DataLoader.Train.sampler.batch_size=128 \
    -o DataLoader.Eval.sampler.batch_size=256
```

* train process1: train 300 epoch, top1 acc 77.1
  * [config1](config/hrt.yml)
  * log:
    * [0-50eooch](log/trainer-0_50.log)
    * [50-100eooch](log/trainer-50_100.log)
    * [100-150eooch](log/trainer-100_150.log)
    * [150-200eooch](log/trainer-150_200.log)
    * [200-250eooch](log/trainer-200_250.log)
    * [250-300eooch](log/trainer-250_300.log)
  * [acc plot1](log/train1.png)
  * [model1](model/train1/latest.pdparams)
* train process2: base 77.1 model, train 50 epoch, top1 acc 77.3
  * [config2](config/hrt2.yml)
  * log: [add50epoch](log/trainer-add50.log)
  * [acc plot2](log/train2.png)
  * [model2](model/train2/latest.pdparams)

This is a Beta version, some bugs still exist, these will be fixed in the following versions.
