# 1.Configure

* 1.Install TensorFlow 2.8 + Pillow + tqdm
```Bash
conda create -n tf28 python=3.8
conda activate tf28

conda install cudatoolkit=11.2 -c conda-forge
conda install cudnn=8.1 -c conda-forge
conda install tqdm
conda install pillow

pip install tensorflow==2.8.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```
* 2.Download and perpare the **Pascal Context** dataset
  * Download the raw dataset (zip)
  * Unzip the raw dataset.
  * Use the following script to convert the raw dataset to tfrecord, you need to adjust the paths:
  ```Bash
  python ids/tools/convert_record.py \ 
  --convert_datasets=pascalcontext \
  --pascalcontext_path={Downloaded raw dataset dir} \
  --tfrecord_outputs={Your output path}
  ```
* 3.Download pretrained backbone weights from [CAR/docs/download.md](https://github.com/edwardyehuang/CAR/blob/master/docs/download.md)

# 2.Training
```Bash
python train.py --flagfile={path to your cfg file }
```

# 3.Testing
```Bash
python eval.py --flagfile={path to your cfg file } --checkpoint_dir={path to your checkpoint }
```



The code references the [CAR: Class-aware Regularizations for Semantic Segmentation](https://github.com/edwardyehuang/CAR) code, thanks to the work of CAR.







