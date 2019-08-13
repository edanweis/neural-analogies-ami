# neural-analogies-ami
How to setup AWS AMI for neural-analogies

## Set up Theano + Keras

1. Find the theano/keras environment in Conda, and activate the Theano environment
```
conda info --envs
activate theano_p36
```
2. Set Keras `sudo nano ~/keras/keras.json` backend
```
{
"image_dim_ordering":"th",
"epsilon": 1e-07,
"floatx": "float32",
"backend": "theano"
}
```

3. Configure Theano GPU backend
Create, or edit file `sudo nano ~/.theanorc`:

```
[global]
device = cuda
floatX = float32

[gpuarray]
preallocate = -1
```

4. Test Theano

Create and run `theano_test.py`:
```
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
```

If GPU backend isn't working:

1. [Install Nvidia drivers on AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)

2. Check that `[dnn]` paths are defined in `~/.theanorc`


## Install neural analogies

5. Install Python package from the zkneupper/image-analogies GitHub repository

`pip install git+https://github.com/zkneupper/image-analogies`

`cd image-analogies/scripts`

6. Download vgg weights

`wget https://github.com/awentzonline/image-analogies/releases/download/v0.0.5/vgg16_weights.h5`

7. Upload images to current local directory

`scp -i -r mykey.pem ubuntu@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:image-analogies/scripts/out .`

## Run make_image_analogy

8. `make_image_analogy.py $MASK_IMAGE 


## Download output

9. `scp -i -r mykey.pem ubuntu@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:/path/to/folder .`
