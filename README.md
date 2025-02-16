# Requirements after installation
- Make sure numpy 1.24.3 is installed afterwards (if not use it to avoid issues)
- Python 3.10.16 is used.

# CUDA Toolkit
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
nano ~/.bashrc
```
add the below lines:
```
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export PATH=/usr/local/cuda-11.8/bin:$PATH
```
```
source ~/.bashrc
sudo nano /etc/ld.so.conf
```
add the below line:
```
/usr/local/cuda-11.8/lib64
```
Ctrl+O<br />
Enter<br />
Ctrl+X
```
sudo ldconfig
```

# CUDNN
The file `test_cudnn.c` can be found in `setup` directory.
```
cd ~/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
cd ~
gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-11.8/include -L/usr/local/cuda-11.8/lib64 -lcudnn
./test_cudnn
```

# TensorRT
```
cd ~
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz TensorRT-8.6.1.6
sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1.6
```

# Conda
```
conda create --name tfnew python=3.10.16
conda activate tfnew
```

# Protobuf
```
nano ~/.bashrc
```

add the below line:
```
export PATH=/usr/local/protoc-3.20.3/bin:$PATH
```

```
source ~/.bashrc
cd ~/projects/custom-object-detection/models/research
protoc object_detection/protos/*.proto --python_out=.
```

# Install Tensorflow
```
conda activate tfnew
python3 -m pip install tensorflow==2.13.1
```

# Install COCO API
```
python3 -m pip install "cython<3.0.0" wheel
python3 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Edit cocoeval.py by using `nano /home/ubuntu/miniconda3/envs/tfnew/lib/python3.10/site-packages/pycocotools/cocoeval.py` (adapt the path, with the path of your conda environment/python/install package pycocotools).

and replace<br />
```self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
```
with
```
self.iouThrs = np.linspace(.5, 0.95, 10, endpoint=True)
self.recThrs = np.linspace(.0, 1.00, 101, endpoint=True)
```

and

```
tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
```

with

```
tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
```


Then Ctrl+O, Enter, Ctrl+X to save and exit nano.

# Install Object Detection API
```
cd ~/projects/custom-object-detection/models/research
cp object_detection/packages/tf2/setup.py .
nano setup.py
```
Modify `tf-models-official>=2.5.1` to `tf-models-official==2.13.1`,<br />
Ctrl+O<br />
Enter<br />
Ctrl+X
```
python3 -m pip install "cython<3.0.0" wheel
python3 -m pip install "pyyaml==5.4.1" --no-build-isolation
python3 -m pip install .
python3 -m pip install tensorflow==2.13.1
python3 -m pip install Pillow==9.5.0
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

# Partition images to train and test sets
```
cd ~/projects/custom-object-detection/scripts/preprocesssing

python partition_dataset.py -x -i /home/ubuntu/projects/custom-object-detection/workspace/training_demo/images -r 0.1
```

# Pre-process
```
cd ~/projects/custom-object-detection/scripts/preprocesssing

python3 generate_tfrecord.py -x /home/ubuntu/projects/custom-object-detection/workspace/training_demo/images/train -l /home/ubuntu/projects/custom-object-detection/workspace/training_demo/annotations/label_map.pbtxt -o /home/ubuntu/projects/custom-object-detection/workspace/training_demo/annotations/train.record

python3 generate_tfrecord.py -x /home/ubuntu/projects/custom-object-detection/workspace/training_demo/images/test -l /home/ubuntu/projects/custom-object-detection/workspace/training_demo/annotations/label_map.pbtxt -o /home/ubuntu/projects/custom-object-detection/workspace/training_demo/annotations/test.record
```

# Train

Unzip `ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz` and rename the folder to `ssd_resnet50_v1`. Place it in `pre-trained-model` folder.<br />
Rename folder `checkpoint` (from `ssd_resnet50_v1`) to `checkpoint0`.
Update the `pipeline.config` (from `ssd_resnet50_v1`).
```
cd ~/projects/custom-object-detection/workspace/training_demo

# One time the below line
cp ~/projects/custom-object-detection/models/research/object_detection/model_main_tf2.py .

python3 model_main_tf2.py --model_dir=pre-trained-models/ssd_resnet50_v1 --pipeline_config_path=pre-trained-models/ssd_resnet50_v1/pipeline.config
```

# Evaluate
```
python3 model_main_tf2.py --model_dir=pre-trained-models/ssd_resnet50_v1 --pipeline_config_path=pre-trained-models/ssd_resnet50_v1/pipeline.config --checkpoint_dir=pre-trained-models/ssd_resnet50_v1
```

# Tensorboard
```
cd ~/projects/custom-object-detection/workspace/training_demo
tensorboard --logdir=pre-trained-models/ssd_resnet50_v1
```

# Export trained model
```
cd ~/projects/custom-object-detection/workspace/training_demo

python ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./pre-trained-models/ssd_resnet50_v1/pipeline.config --trained_checkpoint_dir ./pre-trained-models/ssd_resnet50_v1/ --output_directory ./exported-models/my_model
```

# Inference
```
cd ~/projects/custom-object-detection/workspace/training_demo

python3 ./plot_object_detection_checkpoint.py
```