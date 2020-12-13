# UNet

## Install
```shell
python setup.py install
pip install .
```

## Using trained model
```python
from unet import UnetPipeline

model = UnetPipeline()
model.load(PATH_TO_MODEL_DIRECTORY)
model.predict(np.array(...))
```

## Train Model
```shell
python train.py \
  --seed int \
  --data_dir path_to_data_folder \
  --save_dir path_to_save_model \
  --encoder_name pretrained_backbone_encoder_name_of_Unet \
  --crop_size image_crop_size \
  --stride image_stride \
  --train_batch_size int \
  --eval_batch_size int \
  --n_epochs int \
  --save_epochs int \
  --lr float \
  --no_cuda boolean \
  --cpu_count int \
  
```
