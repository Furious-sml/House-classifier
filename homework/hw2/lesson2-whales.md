
## Humpback whale classification


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
from fastai.imports import *
```


```python
from fastai.conv_learner import *
from fastai.transforms import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import pandas as pd
import matplotlib.pyplot as plt
```


```python
PATH = 'data/whale/'
```


```python
ls {PATH}
```

    [0m[01;34mmodels[0m/                [01;34mtest[0m/     [01;34mtmp[0m/    train.csv
    sample_submission.csv  [01;31mtest.zip[0m  [01;34mtrain[0m/  [01;31mtrain.zip[0m


### Exploring the image


```python
def plot_images_for_filenames(filenames, labels, rows=4):
    imgs = [plt.imread(f'{PATH}/train/{filename}') for filename in filenames]
    
    return plot_images(imgs, labels, rows)
    
def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i])
```


```python
np.random.seed(24)
```

## using resnext 50


```python
sz = 224
arch = resnext50
bs = 24
```


```python
def get_1st(path): return glob(f'{path}/*.*')[0]
```

In single-label classification each sample belongs to one class. In the previous example, each image is either a *dog* or a *cat*.

In multi-label classification each sample can belong to one or more clases. In the previous example, the first images belongs to two clases: *haze* and *primary*. The second image belongs to four clases: *agriculture*, *clear*, *primary* and  *water*.

## Read labelled data, creating validation set


```python
label_csv = f'{PATH}train.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
val_idxs = val_idxs[:100]
```


```python
labels_df = pd.read_csv(label_csv)
labels_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image</th>
      <th>Id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00022e1a.jpg</td>
      <td>w_e15442c</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000466c4.jpg</td>
      <td>w_1287fbc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00087b01.jpg</td>
      <td>w_da2efe0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001296d5.jpg</td>
      <td>w_19e5482</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0014cfdf.jpg</td>
      <td>w_f22f3e3</td>
    </tr>
  </tbody>
</table>
</div>




```python
rand_rows = labels_df.sample(frac=1.)[:20]
imgs = list(rand_rows['Image'])
labels = list(rand_rows['Id'])

plot_images_for_filenames(imgs, labels)
```


![png](output_17_0.png)


We use a different set of data augmentations for this dataset - we also allow vertical flips, since we don't expect vertical orientation of satellite images to change our classifications.


```python
def get_data(sz):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    return ImageClassifierData.from_csv(PATH, 'train', label_csv, test_name='test', tfms=tfms,
                    val_idxs=val_idxs)
```


```python
data = get_data(224)
```


```python
x,y = next(iter(data.val_dl))
```


```python
y
```




    
     3201
      614
      890
     2955
     4104
     2620
      527
      431
      241
      173
     2990
        0
      652
     4005
      177
     2669
     2638
     3203
     3942
     2378
     2620
     3010
     2663
      498
      767
     3589
     1203
        0
     3026
     3592
     3476
      544
     3252
     3192
      757
     1329
      301
      956
     1182
     1643
      756
     1680
      294
     2573
      106
       15
      760
     3997
     3320
     2782
      389
     2501
        0
     3673
     2687
     2721
     3481
     2469
     2214
     4219
     1372
     2628
      809
     1724
    [torch.LongTensor of size 64]




```python
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);
```


![png](output_23_0.png)



```python
learn = ConvLearner.pretrained(arch, data, precompute=True)
```


```python
lrf=learn.lr_find()
learn.sched.plot()
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



     86%|████████▌ | 131/153 [00:02<00:00, 46.25it/s, loss=132] 
                                                               


![png](output_25_2.png)



```python
learn.fit(1e-2, 5)
learn.precompute=False
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    [0.      8.41846 7.91034 0.08681]                           
    [1.      7.53795 7.71089 0.08681]                           
    [2.      7.00856 7.57325 0.10069]                           
    [3.      6.54838 7.49331 0.10851]                           
    [4.      6.08575 7.3723  0.11632]                           
    



```python
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    [0.      6.73886 7.34232 0.10851]                           
    [1.      6.5868  7.17467 0.10851]                           
    [2.      6.30265 7.17084 0.11632]                           
    [3.      6.3159  7.13258 0.12413]                           
    [4.      6.10811 7.1048  0.13194]                           
    [5.      5.85838 7.06237 0.13194]                           
    [6.      5.76518 7.05797 0.13194]                           
    



```python
#lrs = np.array([lr/9,lr/3,lr])
```


```python
#learn.unfreeze()
#learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
```


```python
learn.save('humpback_whale')
learn.load('humpback_whale')
```


```python
test_preds,y = learn.TTA(is_test=True)
```

                                                  


```python
preds_avg = np.mean(test_preds, axis=0)
```


```python
#preds = np.argmax(preds_avg, axis=1)
#probs = np.exp(preds_avg)
probs = np.exp(test_preds)
probs.shape
```




    (15610, 4251)




```python
data.test_ds.fnames
```




    ['test/c292553f.jpg',
     'test/c70ca9cc.jpg',
     'test/18ff383f.jpg',
     'test/5802f268.jpg',
     'test/080f5fab.jpg',
     'test/a075c784.jpg',
     'test/ca130a09.jpg',
     'test/a32fc078.jpg',
     'test/3a08fd5e.jpg',
     'test/cd316c62.jpg',
     'test/eb9aa389.jpg',
     'test/59dc6925.jpg',
     'test/4a3dda56.jpg',
     'test/4799d6d7.jpg',
     'test/107fbca3.jpg',
     'test/97efd070.jpg',
     'test/0e987496.jpg',
     'test/85afb6bd.jpg',
     'test/1e2bf374.jpg',
     'test/ea2ca0a2.jpg',
     'test/7a2b8779.jpg',
     'test/4e168c7e.jpg',
     'test/8ebbfbf1.jpg',
     'test/5e2fed01.jpg',
     'test/39e4f5a3.jpg',
     'test/8a16a939.jpg',
     'test/c6d9e63a.jpg',
     'test/1f33c310.jpg',
     'test/961d5312.jpg',
     'test/faa7fad3.jpg',
     'test/397bc85e.jpg',
     'test/7a8de645.jpg',
     'test/bb643a35.jpg',
     'test/b2ed17ce.jpg',
     'test/e080f337.jpg',
     'test/b5a61d42.jpg',
     'test/bdf86b00.jpg',
     'test/88a66661.jpg',
     'test/20d0ffc1.jpg',
     'test/8f047c48.jpg',
     'test/9e912090.jpg',
     'test/944ceaee.jpg',
     'test/0e8e84c2.jpg',
     'test/c0ca04a0.jpg',
     'test/c4362011.jpg',
     'test/0a010220.jpg',
     'test/e4c04116.jpg',
     'test/bde31f6b.jpg',
     'test/163a3a67.jpg',
     'test/2b9586b7.jpg',
     'test/3a2b6223.jpg',
     'test/1a5772ab.jpg',
     'test/d4660728.jpg',
     'test/a84cc358.jpg',
     'test/1e50e6a7.jpg',
     'test/048c312f.jpg',
     'test/85aaec1f.jpg',
     'test/a4cd3f2c.jpg',
     'test/1302cc35.jpg',
     'test/4f63a37b.jpg',
     'test/9bc04aee.jpg',
     'test/6a0a6d66.jpg',
     'test/3cf1a1dc.jpg',
     'test/fad409b6.jpg',
     'test/384434bc.jpg',
     'test/42fca740.jpg',
     'test/a21d7f3e.jpg',
     'test/b1fc7879.jpg',
     'test/739309e5.jpg',
     'test/6608bbae.jpg',
     'test/95ae3c8d.jpg',
     'test/ad0ccc41.jpg',
     'test/632c42de.jpg',
     'test/cbf040e4.jpg',
     'test/17f67826.jpg',
     'test/2291ca15.jpg',
     'test/fcbe8e7c.jpg',
     'test/bf9ef0ea.jpg',
     'test/040c2a16.jpg',
     'test/d50abed7.jpg',
     'test/43cad4ea.jpg',
     'test/e1841979.jpg',
     'test/a25a6f84.jpg',
     'test/fef4d00d.jpg',
     'test/88f46a33.jpg',
     'test/102063ff.jpg',
     'test/fa484acf.jpg',
     'test/ddc2b22a.jpg',
     'test/04d62bd2.jpg',
     'test/711fb1b5.jpg',
     'test/cf848ac9.jpg',
     'test/902bb67a.jpg',
     'test/ab0b71c2.jpg',
     'test/85f75637.jpg',
     'test/8e2d52af.jpg',
     'test/1aca3c32.jpg',
     'test/38ab6a87.jpg',
     'test/2e307a15.jpg',
     'test/5c2473e0.jpg',
     'test/6f79709f.jpg',
     'test/35df51b7.jpg',
     'test/118a681a.jpg',
     'test/fa968b5a.jpg',
     'test/3b1d20bd.jpg',
     'test/1794ede4.jpg',
     'test/7f95b119.jpg',
     'test/5ba3314e.jpg',
     'test/18389e17.jpg',
     'test/090fa854.jpg',
     'test/b130a152.jpg',
     'test/b159ca30.jpg',
     'test/a115eb8f.jpg',
     'test/194e3efc.jpg',
     'test/25df0a68.jpg',
     'test/e111b2cc.jpg',
     'test/a840133b.jpg',
     'test/de7390ac.jpg',
     'test/56d67433.jpg',
     'test/20fc4986.jpg',
     'test/cbdf0fb2.jpg',
     'test/f2895dd2.jpg',
     'test/b755af68.jpg',
     'test/790f77e9.jpg',
     'test/0bf12fb8.jpg',
     'test/156de6b8.jpg',
     'test/011fc871.jpg',
     'test/224479de.jpg',
     'test/dbea348e.jpg',
     'test/51ef73aa.jpg',
     'test/77c48d7f.jpg',
     'test/f13930b9.jpg',
     'test/fc0c2db5.jpg',
     'test/0a5b8b78.jpg',
     'test/6fd1f649.jpg',
     'test/647f8e83.jpg',
     'test/f62c11e5.jpg',
     'test/f5d5af85.jpg',
     'test/675337eb.jpg',
     'test/fd851f16.jpg',
     'test/ea7f29b1.jpg',
     'test/44e9872e.jpg',
     'test/23360fc6.jpg',
     'test/f5321b65.jpg',
     'test/4efe412c.jpg',
     'test/e26fa15f.jpg',
     'test/70a95110.jpg',
     'test/a91820c3.jpg',
     'test/16731037.jpg',
     'test/bb4351d7.jpg',
     'test/d2346ce7.jpg',
     'test/afbe35e6.jpg',
     'test/7af2d6dc.jpg',
     'test/8e1c4a4d.jpg',
     'test/6d0d9824.jpg',
     'test/b84e6f63.jpg',
     'test/b34bf8c1.jpg',
     'test/f33f041d.jpg',
     'test/56805251.jpg',
     'test/fa89bc92.jpg',
     'test/6216ad6e.jpg',
     'test/27cf62f2.jpg',
     'test/86cceae7.jpg',
     'test/09c75a4a.jpg',
     'test/cfd6e3a1.jpg',
     'test/9c3178d9.jpg',
     'test/56580305.jpg',
     'test/a59e740b.jpg',
     'test/d5833a89.jpg',
     'test/13d93db5.jpg',
     'test/d211508a.jpg',
     'test/f08f6cd4.jpg',
     'test/6601b15c.jpg',
     'test/aa92e238.jpg',
     'test/fcbb962e.jpg',
     'test/f872be02.jpg',
     'test/bc973c0c.jpg',
     'test/a50e0a66.jpg',
     'test/6386c3da.jpg',
     'test/be57ec90.jpg',
     'test/2f2544ee.jpg',
     'test/932a920f.jpg',
     'test/bccb1669.jpg',
     'test/14ac80b3.jpg',
     'test/c2214eb6.jpg',
     'test/d4e93588.jpg',
     'test/77cecd91.jpg',
     'test/9e483373.jpg',
     'test/82ddb63c.jpg',
     'test/f0198458.jpg',
     'test/14876ca5.jpg',
     'test/192ae0c6.jpg',
     'test/ddbd799b.jpg',
     'test/dae480cb.jpg',
     'test/dd865a8c.jpg',
     'test/77dcc4f9.jpg',
     'test/e5c7948a.jpg',
     'test/85ad2200.jpg',
     'test/3ae79741.jpg',
     'test/a47eabb2.jpg',
     'test/9c89af60.jpg',
     'test/89df1c8b.jpg',
     'test/93be6443.jpg',
     'test/e6312bb1.jpg',
     'test/e31f309f.jpg',
     'test/6817ac5b.jpg',
     'test/fd3fb596.jpg',
     'test/46c887cf.jpg',
     'test/d6e589dc.jpg',
     'test/b33b80de.jpg',
     'test/96f6b579.jpg',
     'test/33d86897.jpg',
     'test/86511a1f.jpg',
     'test/ff46562c.jpg',
     'test/0b8448c8.jpg',
     'test/bb146299.jpg',
     'test/5acb9cd6.jpg',
     'test/c7d915b5.jpg',
     'test/849cea0e.jpg',
     'test/f93c44d4.jpg',
     'test/88fc1bce.jpg',
     'test/0266c5b9.jpg',
     'test/46acadb4.jpg',
     'test/28d4e63b.jpg',
     'test/4229d5fd.jpg',
     'test/dacc1ec9.jpg',
     'test/cd4544c4.jpg',
     'test/99cd4177.jpg',
     'test/575cd05f.jpg',
     'test/2f950e12.jpg',
     'test/ba927651.jpg',
     'test/66180a8b.jpg',
     'test/f0f9090a.jpg',
     'test/6b6f4054.jpg',
     'test/479efec3.jpg',
     'test/887e7ad4.jpg',
     'test/460bee1e.jpg',
     'test/e3ff4282.jpg',
     'test/34ede78f.jpg',
     'test/eec84d2f.jpg',
     'test/cb84e66b.jpg',
     'test/740d9754.jpg',
     'test/1f190e76.jpg',
     'test/928ee350.jpg',
     'test/64f5fc2b.jpg',
     'test/4c7abb64.jpg',
     'test/80856bdd.jpg',
     'test/c8b522c5.jpg',
     'test/3d06d9fc.jpg',
     'test/625b9d11.jpg',
     'test/bf78928b.jpg',
     'test/f0b9b699.jpg',
     'test/054f9ad2.jpg',
     'test/a3fa4943.jpg',
     'test/83594e9e.jpg',
     'test/c3dd0a96.jpg',
     'test/a8d670e4.jpg',
     'test/00367faf.jpg',
     'test/09d7ac3c.jpg',
     'test/9b4b818d.jpg',
     'test/2cac3ecc.jpg',
     'test/a16727d3.jpg',
     'test/6cb1c550.jpg',
     'test/8a71e5c1.jpg',
     'test/f3ecacfe.jpg',
     'test/24d3a770.jpg',
     'test/27972adb.jpg',
     'test/92a7ba36.jpg',
     'test/2171c8cf.jpg',
     'test/581046e9.jpg',
     'test/e004a474.jpg',
     'test/76320e2b.jpg',
     'test/9f925716.jpg',
     'test/cb039008.jpg',
     'test/c5db62cf.jpg',
     'test/bef44806.jpg',
     'test/67956b0f.jpg',
     'test/bfb47ad9.jpg',
     'test/baa708cf.jpg',
     'test/a25ae909.jpg',
     'test/28d413e8.jpg',
     'test/4d1738ef.jpg',
     'test/25bec01f.jpg',
     'test/c6be7696.jpg',
     'test/81e62bee.jpg',
     'test/2109edab.jpg',
     'test/d386888c.jpg',
     'test/bdcdb761.jpg',
     'test/3fa15942.jpg',
     'test/939e15b5.jpg',
     'test/46840219.jpg',
     'test/1776a23b.jpg',
     'test/60328584.jpg',
     'test/ccc70742.jpg',
     'test/df222b19.jpg',
     'test/32b83c86.jpg',
     'test/91545217.jpg',
     'test/32575533.jpg',
     'test/f9dfbe92.jpg',
     'test/54d35cf0.jpg',
     'test/5b536d03.jpg',
     'test/8b4feea4.jpg',
     'test/c208a718.jpg',
     'test/ef8c89f7.jpg',
     'test/ce7f0f40.jpg',
     'test/d00cf68a.jpg',
     'test/505c0a65.jpg',
     'test/e2ac0e08.jpg',
     'test/db7c2abc.jpg',
     'test/8a511f82.jpg',
     'test/f63572f4.jpg',
     'test/8d8af3f0.jpg',
     'test/918722b3.jpg',
     'test/45204323.jpg',
     'test/8e4295ab.jpg',
     'test/11ffa611.jpg',
     'test/85204ab9.jpg',
     'test/3dc643e4.jpg',
     'test/27fbe0f5.jpg',
     'test/61773fe1.jpg',
     'test/ccbebe58.jpg',
     'test/5c453bf0.jpg',
     'test/a53cdda5.jpg',
     'test/6dbdf165.jpg',
     'test/2f028d0b.jpg',
     'test/049c3da3.jpg',
     'test/e6bf93b9.jpg',
     'test/337300f6.jpg',
     'test/2e7687ac.jpg',
     'test/de14251c.jpg',
     'test/315d4c7f.jpg',
     'test/c0062d80.jpg',
     'test/cda4652b.jpg',
     'test/ee2061b2.jpg',
     'test/0cbd5aca.jpg',
     'test/c3f09687.jpg',
     'test/ae7fb1f6.jpg',
     'test/150cc570.jpg',
     'test/2612bd2a.jpg',
     'test/297991be.jpg',
     'test/20789779.jpg',
     'test/3871f2aa.jpg',
     'test/05f17a1f.jpg',
     'test/f1663434.jpg',
     'test/bfdfae98.jpg',
     'test/265da307.jpg',
     'test/477e5830.jpg',
     'test/2112d690.jpg',
     'test/633d2553.jpg',
     'test/7750e202.jpg',
     'test/e7c92028.jpg',
     'test/b9603045.jpg',
     'test/0d8e2ba4.jpg',
     'test/2392ab73.jpg',
     'test/b924951e.jpg',
     'test/ff355c05.jpg',
     'test/802455cc.jpg',
     'test/f5c3e2d9.jpg',
     'test/b15b6ac7.jpg',
     'test/793b9ee9.jpg',
     'test/820047db.jpg',
     'test/2e6e899f.jpg',
     'test/87a37a22.jpg',
     'test/65199df7.jpg',
     'test/1213ad77.jpg',
     'test/6c390e76.jpg',
     'test/2f99e656.jpg',
     'test/a261a7a2.jpg',
     'test/ed766b22.jpg',
     'test/25b9e679.jpg',
     'test/5d0e7340.jpg',
     'test/699052e8.jpg',
     'test/e7c148f4.jpg',
     'test/830ed619.jpg',
     'test/c99e6f27.jpg',
     'test/95169209.jpg',
     'test/cb88ecc1.jpg',
     'test/6e3233a5.jpg',
     'test/18bf1f7b.jpg',
     'test/2a9851f0.jpg',
     'test/bf46e16f.jpg',
     'test/3c3d4d45.jpg',
     'test/a4804381.jpg',
     'test/05b2063a.jpg',
     'test/8fd44b7e.jpg',
     'test/c77f6739.jpg',
     'test/fa3b5c6f.jpg',
     'test/f894f4b9.jpg',
     'test/c015e347.jpg',
     'test/2f5c9417.jpg',
     'test/40ced212.jpg',
     'test/3021c535.jpg',
     'test/eb896e4a.jpg',
     'test/e3f5fb3c.jpg',
     'test/3b8f97ee.jpg',
     'test/81cf8057.jpg',
     'test/8ad7e824.jpg',
     'test/2eea599e.jpg',
     'test/765401d2.jpg',
     'test/8e0f5e74.jpg',
     'test/c814b4b0.jpg',
     'test/a92cb363.jpg',
     'test/336d13c7.jpg',
     'test/88410523.jpg',
     'test/9633ef04.jpg',
     'test/415c6aa3.jpg',
     'test/6a1bebba.jpg',
     'test/b9f8a5d6.jpg',
     'test/5c6fdf03.jpg',
     'test/09c424bc.jpg',
     'test/25541291.jpg',
     'test/5e976997.jpg',
     'test/5f2a0afe.jpg',
     'test/06f099aa.jpg',
     'test/28ee0086.jpg',
     'test/73382d98.jpg',
     'test/36891fe8.jpg',
     'test/e641645a.jpg',
     'test/837fd5a8.jpg',
     'test/8b5908db.jpg',
     'test/95ea12ca.jpg',
     'test/73c3614b.jpg',
     'test/8a50ae45.jpg',
     'test/ab80fe26.jpg',
     'test/3a3d5f88.jpg',
     'test/13dc95b6.jpg',
     'test/cd853027.jpg',
     'test/4aac146b.jpg',
     'test/7dfba9bc.jpg',
     'test/088c1e2d.jpg',
     'test/0e742427.jpg',
     'test/b0e7d0c9.jpg',
     'test/5786a1db.jpg',
     'test/7634b304.jpg',
     'test/04997d5e.jpg',
     'test/6483a52c.jpg',
     'test/6b046145.jpg',
     'test/236b0924.jpg',
     'test/0892408f.jpg',
     'test/8eaa8773.jpg',
     'test/3d0daa18.jpg',
     'test/8599b5eb.jpg',
     'test/8eee1794.jpg',
     'test/d1af36db.jpg',
     'test/cf0a3123.jpg',
     'test/00b0d279.jpg',
     'test/1c39f085.jpg',
     'test/0fd567d5.jpg',
     'test/05b23512.jpg',
     'test/7f180ff4.jpg',
     'test/86334414.jpg',
     'test/b0473b2b.jpg',
     'test/9a427015.jpg',
     'test/2fcdc2f1.jpg',
     'test/a41588e2.jpg',
     'test/06fad355.jpg',
     'test/035333d0.jpg',
     'test/2cae06a7.jpg',
     'test/6b952759.jpg',
     'test/b9b61fed.jpg',
     'test/84e10b1c.jpg',
     'test/060d7d39.jpg',
     'test/d055d77b.jpg',
     'test/253dfaac.jpg',
     'test/086d9a97.jpg',
     'test/9ca6aba0.jpg',
     'test/44bc1fe0.jpg',
     'test/3fb49113.jpg',
     'test/9e2f523b.jpg',
     'test/4be884b0.jpg',
     'test/446a0eba.jpg',
     'test/48696cc8.jpg',
     'test/7cd03ca7.jpg',
     'test/714d3459.jpg',
     'test/bfc4ee42.jpg',
     'test/8501e16c.jpg',
     'test/f8e2f4bc.jpg',
     'test/3be792ce.jpg',
     'test/211ddd7d.jpg',
     'test/b435b5bb.jpg',
     'test/b4d2790d.jpg',
     'test/ee565a86.jpg',
     'test/eeedaf09.jpg',
     'test/2e63370f.jpg',
     'test/83b9f8d7.jpg',
     'test/7331a99a.jpg',
     'test/3361727c.jpg',
     'test/d382e6ee.jpg',
     'test/ef569f83.jpg',
     'test/ad3d480c.jpg',
     'test/2987c014.jpg',
     'test/6d466faf.jpg',
     'test/ff7f469f.jpg',
     'test/0b651283.jpg',
     'test/d97bb43b.jpg',
     'test/37db08be.jpg',
     'test/361f7c7a.jpg',
     'test/27ed473d.jpg',
     'test/e307b7c2.jpg',
     'test/e8e10398.jpg',
     'test/818408d3.jpg',
     'test/72293c24.jpg',
     'test/cd3ed013.jpg',
     'test/c41236dc.jpg',
     'test/784f44c2.jpg',
     'test/f9833b21.jpg',
     'test/9fc4a952.jpg',
     'test/1d4fee96.jpg',
     'test/57d879db.jpg',
     'test/e6428ba4.jpg',
     'test/dcdbf5b0.jpg',
     'test/d6bed141.jpg',
     'test/70df2643.jpg',
     'test/097267e2.jpg',
     'test/8f3d11dc.jpg',
     'test/b9ba921f.jpg',
     'test/0ec7c569.jpg',
     'test/db0bc720.jpg',
     'test/be9cc3f5.jpg',
     'test/7f16c886.jpg',
     'test/6de4c958.jpg',
     'test/6b510466.jpg',
     'test/2b32d765.jpg',
     'test/0b1666a7.jpg',
     'test/423346d6.jpg',
     'test/17459273.jpg',
     'test/d5f2bc6a.jpg',
     'test/c502c387.jpg',
     'test/af2dd8fa.jpg',
     'test/866a24fe.jpg',
     'test/c959bd40.jpg',
     'test/d13de039.jpg',
     'test/00ce5ca3.jpg',
     'test/c08886b3.jpg',
     'test/c31e3d5e.jpg',
     'test/940d4f6c.jpg',
     'test/75b17cee.jpg',
     'test/c52ef613.jpg',
     'test/ba64ce30.jpg',
     'test/bf1bfb17.jpg',
     'test/81232d97.jpg',
     'test/a62f3d83.jpg',
     'test/f8474488.jpg',
     'test/350f8f5c.jpg',
     'test/23185664.jpg',
     'test/303d40a9.jpg',
     'test/ca01f0d0.jpg',
     'test/5634b746.jpg',
     'test/0845b7cb.jpg',
     'test/718d6cf9.jpg',
     'test/43317cb0.jpg',
     'test/bd70f118.jpg',
     'test/b1559658.jpg',
     'test/070a0cfd.jpg',
     'test/fd803e52.jpg',
     'test/81e36336.jpg',
     'test/4b5e982b.jpg',
     'test/fc947001.jpg',
     'test/141aec1f.jpg',
     'test/86e5331e.jpg',
     'test/f24c3123.jpg',
     'test/1e41cd4e.jpg',
     'test/e62d220b.jpg',
     'test/7deadbb0.jpg',
     'test/9b2bfeeb.jpg',
     'test/c8024b45.jpg',
     'test/c815cdcf.jpg',
     'test/e05581c0.jpg',
     'test/ac1f9783.jpg',
     'test/6c4f87d3.jpg',
     'test/c0a0db35.jpg',
     'test/7404a7c9.jpg',
     'test/ba9ade68.jpg',
     'test/268241a9.jpg',
     'test/0dabea5d.jpg',
     'test/a0aea5c6.jpg',
     'test/5ef27b66.jpg',
     'test/4ab13827.jpg',
     'test/6bad28a3.jpg',
     'test/cdf1f3b6.jpg',
     'test/57adc461.jpg',
     'test/9caef6e0.jpg',
     'test/6d42bdba.jpg',
     'test/5755ea71.jpg',
     'test/43ca5115.jpg',
     'test/ae96e8ee.jpg',
     'test/3c8ea78b.jpg',
     'test/c0a97327.jpg',
     'test/3ab2800e.jpg',
     'test/828f0d1e.jpg',
     'test/6875d9f1.jpg',
     'test/d88e6f69.jpg',
     'test/a7762731.jpg',
     'test/c5563cf3.jpg',
     'test/852c019e.jpg',
     'test/1db42333.jpg',
     'test/181cbba2.jpg',
     'test/3a450687.jpg',
     'test/358494f8.jpg',
     'test/cc6f37df.jpg',
     'test/ef9ce2d1.jpg',
     'test/bfcf2ef8.jpg',
     'test/9eac9191.jpg',
     'test/01d2ff07.jpg',
     'test/a0b24883.jpg',
     'test/3ef82b19.jpg',
     'test/eedd11d0.jpg',
     'test/ae245530.jpg',
     'test/e6cf6bde.jpg',
     'test/34932efd.jpg',
     'test/ddbf9401.jpg',
     'test/2e089190.jpg',
     'test/29598899.jpg',
     'test/837874e2.jpg',
     'test/aa4280b7.jpg',
     'test/049dd98d.jpg',
     'test/27c6b953.jpg',
     'test/765828e4.jpg',
     'test/0619b2a2.jpg',
     'test/2b717c4f.jpg',
     'test/ae9c9c23.jpg',
     'test/4acd8349.jpg',
     'test/172a051f.jpg',
     'test/6bdb957f.jpg',
     'test/bb03e6fc.jpg',
     'test/002d8d81.jpg',
     'test/b59aa327.jpg',
     'test/fd7ffb4d.jpg',
     'test/923a841d.jpg',
     'test/7682ac97.jpg',
     'test/3145e23b.jpg',
     'test/3b54ee91.jpg',
     'test/29018ea5.jpg',
     'test/63a0e789.jpg',
     'test/05588003.jpg',
     'test/7a7d8ce0.jpg',
     'test/e5fe6cf7.jpg',
     'test/fcc1d5f5.jpg',
     'test/a849ecc6.jpg',
     'test/34aecd84.jpg',
     'test/c98dab6b.jpg',
     'test/21f06ad9.jpg',
     'test/7f6cb3d6.jpg',
     'test/7b5ec3c5.jpg',
     'test/de583314.jpg',
     'test/cb256d50.jpg',
     'test/776882fa.jpg',
     'test/977c4b0b.jpg',
     'test/ef6b9274.jpg',
     'test/04eeb6f7.jpg',
     'test/8f28a9c9.jpg',
     'test/e3d1a5ff.jpg',
     'test/4ff9e893.jpg',
     'test/83c652a0.jpg',
     'test/f7b3ce58.jpg',
     'test/416a4c3a.jpg',
     'test/e3ac32ef.jpg',
     'test/90710ce0.jpg',
     'test/36aaa526.jpg',
     'test/2da96475.jpg',
     'test/e418ffa0.jpg',
     'test/dbe7cce5.jpg',
     'test/353207f4.jpg',
     'test/67fef658.jpg',
     'test/32ec782d.jpg',
     'test/d2e1c35d.jpg',
     'test/7bcf5f6e.jpg',
     'test/6eac6744.jpg',
     'test/61669569.jpg',
     'test/83d96634.jpg',
     'test/6163b2b3.jpg',
     'test/2c210a4b.jpg',
     'test/51848f6b.jpg',
     'test/65944b86.jpg',
     'test/a880a4ce.jpg',
     'test/cbd5ac86.jpg',
     'test/41bcea05.jpg',
     'test/73b850c0.jpg',
     'test/f62433a1.jpg',
     'test/a3f15bda.jpg',
     'test/81e546ab.jpg',
     'test/46fe1c68.jpg',
     'test/62d88065.jpg',
     'test/9634ee9b.jpg',
     'test/83af3e82.jpg',
     'test/d8818def.jpg',
     'test/6763b6e5.jpg',
     'test/7ae531c5.jpg',
     'test/7ec5794c.jpg',
     'test/91747759.jpg',
     'test/37c9e7f5.jpg',
     'test/9c56af3b.jpg',
     'test/49f836c2.jpg',
     'test/e771f46f.jpg',
     'test/5cc2b7c2.jpg',
     'test/f54c4b6c.jpg',
     'test/01eb30a7.jpg',
     'test/fa356f10.jpg',
     'test/bec7d327.jpg',
     'test/e29a68b4.jpg',
     'test/6bd2b2df.jpg',
     'test/3372f651.jpg',
     'test/1440bb27.jpg',
     'test/ecc7d8ed.jpg',
     'test/850ad7fe.jpg',
     'test/25ea6e4d.jpg',
     'test/66a84f9f.jpg',
     'test/fb408afd.jpg',
     'test/23eeecf5.jpg',
     'test/f62c9c41.jpg',
     'test/e5ac86fd.jpg',
     'test/1ff4e67c.jpg',
     'test/fa1502db.jpg',
     'test/4db0ef84.jpg',
     'test/a190d9ca.jpg',
     'test/c8a160ba.jpg',
     'test/405c2115.jpg',
     'test/6a798e76.jpg',
     'test/a72ebfd0.jpg',
     'test/32006070.jpg',
     'test/0d4875fb.jpg',
     'test/638eb87f.jpg',
     'test/9075c129.jpg',
     'test/62a25948.jpg',
     'test/054a6881.jpg',
     'test/41e6c5fa.jpg',
     'test/2bfe94aa.jpg',
     'test/742b423f.jpg',
     'test/124151ea.jpg',
     'test/076a9746.jpg',
     'test/8a47b752.jpg',
     'test/d8de25e2.jpg',
     'test/d20fd809.jpg',
     'test/852a33cd.jpg',
     'test/c8c86b7c.jpg',
     'test/c55a35a7.jpg',
     'test/179554ed.jpg',
     'test/94c15ca2.jpg',
     'test/f4958f60.jpg',
     'test/0ba30afe.jpg',
     'test/40a81485.jpg',
     'test/850dcee4.jpg',
     'test/16d1df4c.jpg',
     'test/5c1e3835.jpg',
     'test/38b2d7e4.jpg',
     'test/85c2bdf9.jpg',
     'test/30b76a67.jpg',
     'test/0d74f3d8.jpg',
     'test/aa61e224.jpg',
     'test/9f5c828f.jpg',
     'test/eb9db7c4.jpg',
     'test/d03cb19b.jpg',
     'test/c96b736a.jpg',
     'test/65169597.jpg',
     'test/74cb5238.jpg',
     'test/f2d2e4f4.jpg',
     'test/2f417e43.jpg',
     'test/40cb7565.jpg',
     'test/7c2d067d.jpg',
     'test/cb9a1ab9.jpg',
     'test/d697a910.jpg',
     'test/c8e7929f.jpg',
     'test/821d0431.jpg',
     'test/b6729120.jpg',
     'test/e94dbb59.jpg',
     'test/c6828f57.jpg',
     'test/d5678e81.jpg',
     'test/867a2389.jpg',
     'test/d8249316.jpg',
     'test/596f842e.jpg',
     'test/e815c73f.jpg',
     'test/0ed10f63.jpg',
     'test/fd2485b0.jpg',
     'test/74e7f57e.jpg',
     'test/839b9f3e.jpg',
     'test/59ff7ef4.jpg',
     'test/a534486c.jpg',
     'test/9016ae05.jpg',
     'test/d22be6cf.jpg',
     'test/23f91d76.jpg',
     'test/69f48bc2.jpg',
     'test/6c83a9b9.jpg',
     'test/8f7f7a88.jpg',
     'test/4b4e7fa4.jpg',
     'test/0b567249.jpg',
     'test/f1c12865.jpg',
     'test/657330ef.jpg',
     'test/1dbb7d37.jpg',
     'test/4acbca71.jpg',
     'test/5ce0e1db.jpg',
     'test/2299dc28.jpg',
     'test/4f99f434.jpg',
     'test/dbbcc754.jpg',
     'test/23511b80.jpg',
     'test/9159eda5.jpg',
     'test/2ea2ec83.jpg',
     'test/91c93b91.jpg',
     'test/6fd7a3ee.jpg',
     'test/3eb1d33e.jpg',
     'test/c546cde8.jpg',
     'test/53c78016.jpg',
     'test/1b457332.jpg',
     'test/03092073.jpg',
     'test/6c0a8d96.jpg',
     'test/1f3d04ca.jpg',
     'test/fe6884f6.jpg',
     'test/c4b485c1.jpg',
     'test/989c6b29.jpg',
     'test/e5fe3dfc.jpg',
     'test/07175ead.jpg',
     'test/31353e3b.jpg',
     'test/c5c95cb9.jpg',
     'test/9c498220.jpg',
     'test/6bc3103d.jpg',
     'test/669ddfb1.jpg',
     'test/6b94ba56.jpg',
     'test/c38175f5.jpg',
     'test/19dfe3c3.jpg',
     'test/6f352dfe.jpg',
     'test/4d8a196c.jpg',
     'test/63abb5b2.jpg',
     'test/6f971c31.jpg',
     'test/6301f132.jpg',
     'test/aed6fbb5.jpg',
     'test/e52e3b77.jpg',
     'test/a29c7619.jpg',
     'test/ed8d0bf7.jpg',
     'test/07e3accb.jpg',
     'test/d5a5bd3c.jpg',
     'test/2196316d.jpg',
     'test/7cf83313.jpg',
     'test/4e7633f0.jpg',
     'test/1f70b1b1.jpg',
     'test/71c283fd.jpg',
     'test/382ad58d.jpg',
     'test/f7130528.jpg',
     'test/4f158050.jpg',
     'test/68596681.jpg',
     'test/3ff0a42f.jpg',
     'test/1e7fdf35.jpg',
     'test/657e7af4.jpg',
     'test/027a214c.jpg',
     'test/380ffab8.jpg',
     'test/bdfcedec.jpg',
     'test/0feb37c0.jpg',
     'test/555cd9f7.jpg',
     'test/c0fa1c25.jpg',
     'test/059dff9f.jpg',
     'test/7da8c193.jpg',
     'test/eddc57c4.jpg',
     'test/7143e927.jpg',
     'test/a0d985c6.jpg',
     'test/340a2b68.jpg',
     'test/6c13f3de.jpg',
     'test/9279f56a.jpg',
     'test/0b1fa76a.jpg',
     'test/fea0f59f.jpg',
     'test/094304bf.jpg',
     'test/da525ab0.jpg',
     'test/82f53be8.jpg',
     'test/99354fc1.jpg',
     'test/7164e51e.jpg',
     'test/52cc163b.jpg',
     'test/d6d1132f.jpg',
     'test/dd5b75c1.jpg',
     'test/81556ead.jpg',
     'test/a1e20d8d.jpg',
     'test/1fd420b2.jpg',
     'test/f5d69e73.jpg',
     'test/a23d15c1.jpg',
     'test/3ff4bda5.jpg',
     'test/779815fd.jpg',
     'test/296e1766.jpg',
     'test/562b8718.jpg',
     'test/a9a970b7.jpg',
     'test/f014284c.jpg',
     'test/d2d22d0c.jpg',
     'test/04d1b1e2.jpg',
     'test/ca84686c.jpg',
     'test/7728a76d.jpg',
     'test/5d3c8238.jpg',
     'test/038c5e79.jpg',
     'test/c6b57b09.jpg',
     'test/336c5c85.jpg',
     'test/423741c3.jpg',
     'test/d5d7da9a.jpg',
     'test/8640ae4e.jpg',
     'test/575f4fbf.jpg',
     'test/3593f9a9.jpg',
     'test/6914d83d.jpg',
     'test/db945c64.jpg',
     'test/569046b3.jpg',
     'test/0c925c96.jpg',
     'test/4dd21cce.jpg',
     'test/8585640f.jpg',
     'test/fa750ea7.jpg',
     'test/635acd66.jpg',
     'test/b6869d69.jpg',
     'test/2ea53ec3.jpg',
     'test/eea5632a.jpg',
     'test/4d201b24.jpg',
     'test/a04bf3d6.jpg',
     'test/be5fa26b.jpg',
     'test/13f3b883.jpg',
     'test/9aa844d4.jpg',
     'test/7926fa0b.jpg',
     'test/d10e4ab4.jpg',
     'test/91aa1b8e.jpg',
     'test/69d6cf2d.jpg',
     'test/58aa7309.jpg',
     'test/8c3a258e.jpg',
     'test/bf087d80.jpg',
     'test/6402e0c8.jpg',
     'test/1d99196e.jpg',
     'test/dfddd2b8.jpg',
     'test/b7841491.jpg',
     'test/033263d1.jpg',
     'test/d28fbb3b.jpg',
     'test/f2dd4db0.jpg',
     'test/5b8e3407.jpg',
     'test/8ddff553.jpg',
     'test/33e9da87.jpg',
     'test/890685ff.jpg',
     'test/ea106a8e.jpg',
     'test/ea240e23.jpg',
     'test/087000f4.jpg',
     'test/65e552a7.jpg',
     'test/0ea2bb36.jpg',
     'test/44855cc3.jpg',
     'test/e614b474.jpg',
     'test/8f7226d9.jpg',
     'test/cb10d527.jpg',
     'test/6f766d10.jpg',
     'test/5ec836df.jpg',
     'test/c4d81880.jpg',
     'test/a75469aa.jpg',
     'test/6563a727.jpg',
     'test/0974d632.jpg',
     'test/4ef154bf.jpg',
     'test/60cf6bae.jpg',
     'test/e0d823e6.jpg',
     'test/c7fdd7a0.jpg',
     'test/1dddf68b.jpg',
     'test/d22fda7f.jpg',
     'test/cdba9e97.jpg',
     'test/fc9c9ba8.jpg',
     'test/46260685.jpg',
     'test/71ab2a21.jpg',
     'test/4cb35c27.jpg',
     'test/c5316f5d.jpg',
     'test/820be618.jpg',
     'test/cb87dd81.jpg',
     'test/67211915.jpg',
     'test/d002a759.jpg',
     'test/6762b456.jpg',
     'test/c064ec31.jpg',
     'test/b918924a.jpg',
     'test/3fa19e5a.jpg',
     'test/5c6ac6d1.jpg',
     'test/3247679c.jpg',
     'test/8fb6197c.jpg',
     'test/cd91febe.jpg',
     'test/8555bb32.jpg',
     'test/395e5c1f.jpg',
     'test/ba2e94e6.jpg',
     'test/88b28be9.jpg',
     'test/01e63369.jpg',
     'test/3cd3c8ac.jpg',
     'test/cd72c9c4.jpg',
     'test/73020575.jpg',
     'test/e10906cf.jpg',
     'test/b85cfbeb.jpg',
     'test/1e3469da.jpg',
     'test/56896120.jpg',
     'test/d07da5cc.jpg',
     'test/86a04133.jpg',
     'test/eff1747c.jpg',
     'test/5d185e9a.jpg',
     'test/4aab4f4c.jpg',
     'test/ab136b0e.jpg',
     'test/7cec1556.jpg',
     'test/b48c5580.jpg',
     'test/93573825.jpg',
     'test/326b70e9.jpg',
     'test/82296fad.jpg',
     'test/ad01763b.jpg',
     'test/eedfb25a.jpg',
     'test/931489d0.jpg',
     'test/0442910d.jpg',
     'test/e8f30203.jpg',
     'test/26ec294e.jpg',
     'test/be4e4a66.jpg',
     'test/dcc3557a.jpg',
     'test/3450fc1c.jpg',
     'test/7d9394b3.jpg',
     'test/b0a6d9e9.jpg',
     'test/90a43c34.jpg',
     'test/fe60db66.jpg',
     'test/f5c2a89e.jpg',
     'test/ecb4154d.jpg',
     'test/176bf518.jpg',
     ...]




```python
#top5probs = probs.argmax(axis=1)
data.classes
```




    ['new_whale',
     'w_0013924',
     'w_001ebbc',
     'w_002222a',
     'w_002b682',
     'w_002dc11',
     'w_0087fdd',
     'w_008c602',
     'w_009dc00',
     'w_00b621b',
     'w_00c4901',
     'w_00cb685',
     'w_00d8453',
     'w_00fbb4e',
     'w_0103030',
     'w_010a1fa',
     'w_011d4b5',
     'w_0122d85',
     'w_01319fa',
     'w_0134192',
     'w_013bbcf',
     'w_014250a',
     'w_014a645',
     'w_0156f27',
     'w_015c991',
     'w_015e3cf',
     'w_01687a8',
     'w_0175a35',
     'w_018bc64',
     'w_01a4234',
     'w_01a51a6',
     'w_01a99a5',
     'w_01ab6dc',
     'w_01b2250',
     'w_01c2cb0',
     'w_01cbcbf',
     'w_01d6ca0',
     'w_01e1223',
     'w_01f211f',
     'w_01f8a43',
     'w_01f9086',
     'w_024358d',
     'w_0245a27',
     'w_0265cb6',
     'w_026fdf8',
     'w_028ca0d',
     'w_029013f',
     'w_02a768d',
     'w_02b775b',
     'w_02bb4cf',
     'w_02c2248',
     'w_02c9470',
     'w_02cf46c',
     'w_02d5fad',
     'w_02d7dc8',
     'w_02e5407',
     'w_02facde',
     'w_02fce90',
     'w_030294d',
     'w_0308405',
     'w_0324b97',
     'w_032d44d',
     'w_0337aa5',
     'w_034a3fd',
     'w_0378699',
     'w_037955e',
     'w_03a2ed7',
     'w_03b5e9a',
     'w_03c6d18',
     'w_03c84ef',
     'w_03dc41c',
     'w_03ed3de',
     'w_03f060f',
     'w_03fcd5d',
     'w_0408054',
     'w_045d9fc',
     'w_0466071',
     'w_046634b',
     'w_0467840',
     'w_046a210',
     'w_0471bdf',
     'w_0475042',
     'w_048f7a9',
     'w_04c1951',
     'w_04c841c',
     'w_04feee0',
     'w_050bdac',
     'w_050d056',
     'w_0526c04',
     'w_05396d8',
     'w_05567a9',
     'w_056be75',
     'w_057c418',
     'w_058547a',
     'w_059ac60',
     'w_059df09',
     'w_059e347',
     'w_05b2ddd',
     'w_05e6ba7',
     'w_05ec84e',
     'w_05ecba5',
     'w_0600ecf',
     'w_060bdf2',
     'w_060d2c8',
     'w_0626e4d',
     'w_063d82f',
     'w_064ab78',
     'w_0652d70',
     'w_0654dd9',
     'w_0674604',
     'w_068f5bd',
     'w_06972d2',
     'w_06a6351',
     'w_06b6f60',
     'w_06c470d',
     'w_06dbe6b',
     'w_06e47e3',
     'w_06f0fd3',
     'w_06f85b2',
     'w_06fd726',
     'w_06ff732',
     'w_07274b2',
     'w_0729511',
     'w_0729d71',
     'w_073b15e',
     'w_073f071',
     'w_0740d28',
     'w_0753c29',
     'w_075aa6e',
     'w_07616fd',
     'w_076c122',
     'w_076daec',
     'w_0771d4b',
     'w_078b0e7',
     'w_0793503',
     'w_07a425f',
     'w_07abdff',
     'w_07e92ee',
     'w_07fea3d',
     'w_0819271',
     'w_081dd6e',
     'w_0824736',
     'w_0827d51',
     'w_084d01c',
     'w_0853262',
     'w_0869575',
     'w_0899118',
     'w_08d1ccd',
     'w_08ddb50',
     'w_08f1502',
     'w_09558d4',
     'w_095f58d',
     'w_096bc48',
     'w_0970c7f',
     'w_0981144',
     'w_0988bbb',
     'w_099ab25',
     'w_099c712',
     'w_09be3ad',
     'w_09c1e0b',
     'w_09d48d1',
     'w_09d654f',
     'w_09d7946',
     'w_09dd18c',
     'w_09e0cbf',
     'w_09f825c',
     'w_09f9fd3',
     'w_0a0cf7d',
     'w_0a2af22',
     'w_0a565c5',
     'w_0a58a06',
     'w_0a71e87',
     'w_0a91f24',
     'w_0a97a25',
     'w_0aae8c1',
     'w_0ac6a0a',
     'w_0ac9006',
     'w_0acce53',
     'w_0ad6137',
     'w_0ae998f',
     'w_0b04c08',
     'w_0b0b7cc',
     'w_0b0d88d',
     'w_0b18e41',
     'w_0b3b659',
     'w_0b3c02c',
     'w_0b3f313',
     'w_0b4429c',
     'w_0b4bd89',
     'w_0b775c1',
     'w_0b7cc25',
     'w_0b7e949',
     'w_0ba62fd',
     'w_0bbb3de',
     'w_0bc1db0',
     'w_0bc712b',
     'w_0bd3671',
     'w_0be0d81',
     'w_0be82ee',
     'w_0beef28',
     'w_0bfb109',
     'w_0c3295a',
     'w_0c42dba',
     'w_0c45f5e',
     'w_0c6d50d',
     'w_0c6dbbe',
     'w_0c70bc3',
     'w_0c883eb',
     'w_0c8967d',
     'w_0c8a724',
     'w_0c93f94',
     'w_0caa554',
     'w_0cb6294',
     'w_0cc4a2b',
     'w_0cd401c',
     'w_0ce3ccc',
     'w_0d049cf',
     'w_0d0bc48',
     'w_0d0ecfb',
     'w_0d2dc7e',
     'w_0d39a68',
     'w_0d48a7d',
     'w_0d60fdd',
     'w_0d733a5',
     'w_0d85e59',
     'w_0d8fb3f',
     'w_0da589d',
     'w_0da6f67',
     'w_0dac526',
     'w_0dbfc31',
     'w_0dc176f',
     'w_0de84f0',
     'w_0dee306',
     'w_0df2bde',
     'w_0e0e856',
     'w_0e10dbe',
     'w_0e25cf2',
     'w_0e30df6',
     'w_0e32aa2',
     'w_0e40867',
     'w_0e4ef50',
     'w_0e4f53c',
     'w_0e737d0',
     'w_0e7cb1c',
     'w_0e81e60',
     'w_0e96943',
     'w_0e9f6d9',
     'w_0ea2545',
     'w_0ea659e',
     'w_0ead9d7',
     'w_0eb2886',
     'w_0ebd514',
     'w_0ecff13',
     'w_0ee4d6d',
     'w_0f00e71',
     'w_0f16be3',
     'w_0f20cbc',
     'w_0f2f6e6',
     'w_0f41afe',
     'w_0f49f09',
     'w_0f54cdf',
     'w_0f73c3c',
     'w_0f84bf6',
     'w_0f89e72',
     'w_0f8b515',
     'w_0f9166e',
     'w_0f92544',
     'w_0f96780',
     'w_0fa8587',
     'w_0faef4d',
     'w_0fc4835',
     'w_0fce9a3',
     'w_0fe48f3',
     'w_0fea5a3',
     'w_0fecdf7',
     'w_0ffc383',
     'w_1000f90',
     'w_1014df7',
     'w_1029f4e',
     'w_102ab59',
     'w_103488f',
     'w_104cc93',
     'w_108670b',
     'w_108eb8b',
     'w_1090920',
     'w_109fd25',
     'w_10a366f',
     'w_10ace22',
     'w_10bea88',
     'w_11138db',
     'w_1127c35',
     'w_1161030',
     'w_116a51f',
     'w_118d911',
     'w_11adaae',
     'w_11b993a',
     'w_11c2836',
     'w_11c504f',
     'w_11ca856',
     'w_11d8e5a',
     'w_11f6df1',
     'w_120927e',
     'w_1235317',
     'w_123a47f',
     'w_125095f',
     'w_125ba17',
     'w_126653d',
     'w_1272a31',
     'w_1274a11',
     'w_127f4c6',
     'w_1287fbc',
     'w_12c3d3d',
     'w_12cdfbd',
     'w_12d9132',
     'w_12f2352',
     'w_130508d',
     'w_1306632',
     'w_1310342',
     'w_13249f1',
     'w_133ecf4',
     'w_13550b4',
     'w_136337c',
     'w_136653f',
     'w_136ab04',
     'w_1392cde',
     'w_13c6b6b',
     'w_13f8407',
     'w_142402e',
     'w_142dce1',
     'w_1431f4b',
     'w_143b201',
     'w_147b62b',
     'w_1489751',
     'w_14964c1',
     'w_14970b4',
     'w_14ba846',
     'w_14c8f15',
     'w_153645e',
     'w_1539fee',
     'w_15427c3',
     'w_156c0db',
     'w_157ba16',
     'w_158eeb9',
     'w_15928c6',
     'w_1596a47',
     'w_159f36b',
     'w_15b9665',
     'w_15c39f4',
     'w_15d1235',
     'w_15d7ecf',
     'w_15db29f',
     'w_15eae33',
     'w_15ebbda',
     'w_15f29b7',
     'w_1606afc',
     'w_1609b19',
     'w_1632307',
     'w_1638016',
     'w_164cfff',
     'w_1652da1',
     'w_1698c29',
     'w_16ad10e',
     'w_16d1b32',
     'w_16d2ae2',
     'w_16e3c6b',
     'w_16ea0b2',
     'w_16f61aa',
     'w_1703ee5',
     'w_170bacc',
     'w_17136dc',
     'w_1717a13',
     'w_17370ba',
     'w_17377c9',
     'w_17411dd',
     'w_1743d93',
     'w_1746c88',
     'w_175536a',
     'w_1772ed2',
     'w_17964ef',
     'w_179ffe3',
     'w_17a0832',
     'w_17a2610',
     'w_17a3581',
     'w_17b33ae',
     'w_17cb04d',
     'w_17d5eb9',
     'w_17dc953',
     'w_17e592c',
     'w_17e8554',
     'w_17ee910',
     'w_1854334',
     'w_186bcab',
     'w_1875b28',
     'w_187e235',
     'w_1894a1c',
     'w_18a1bf2',
     'w_18a854b',
     'w_18bd7b3',
     'w_18df014',
     'w_18eee6e',
     'w_18fbec1',
     'w_1904e8c',
     'w_1905c66',
     'w_1908442',
     'w_1911cbb',
     'w_1917275',
     'w_19212f0',
     'w_1924883',
     'w_193b7e3',
     'w_1948625',
     'w_1957ab8',
     'w_1969a9d',
     'w_19a5685',
     'w_19aeb9c',
     'w_19c005a',
     'w_19c29aa',
     'w_19ca2c6',
     'w_19dc50f',
     'w_19e5482',
     'w_19e5d10',
     'w_19e8a2d',
     'w_19eb1b8',
     'w_19f0b15',
     'w_19f7a8b',
     'w_1a229eb',
     'w_1a29ba5',
     'w_1a2b4f2',
     'w_1a4ae2c',
     'w_1a4f16e',
     'w_1a536b2',
     'w_1a5beb9',
     'w_1a5e7a2',
     'w_1a62e1f',
     'w_1a70685',
     'w_1a9a38e',
     'w_1a9f141',
     'w_1aa0526',
     'w_1aa512b',
     'w_1ac4c38',
     'w_1add6ae',
     'w_1addcc2',
     'w_1ae1386',
     'w_1aea445',
     'w_1af5d59',
     'w_1b1c32f',
     'w_1b1c4f1',
     'w_1b224c1',
     'w_1b332ba',
     'w_1b5989e',
     'w_1b67a2e',
     'w_1b6d171',
     'w_1b6d34d',
     'w_1beadba',
     'w_1c22d7e',
     'w_1c2fb13',
     'w_1c30ba6',
     'w_1c32062',
     'w_1c3cecd',
     'w_1c3e5da',
     'w_1c432e7',
     'w_1c69443',
     'w_1c6d5f0',
     'w_1cb2134',
     'w_1cca86d',
     'w_1cd9331',
     'w_1d0540c',
     'w_1d05772',
     'w_1d0e29a',
     'w_1d17e8c',
     'w_1d2fbc1',
     'w_1d37fb6',
     'w_1d38877',
     'w_1d4f970',
     'w_1d53d9c',
     'w_1d84278',
     'w_1d96560',
     'w_1d970c9',
     'w_1da7080',
     'w_1dc66b7',
     'w_1dc6d62',
     'w_1dd8e68',
     'w_1ddb63f',
     'w_1deadd7',
     'w_1dff010',
     'w_1e1a051',
     'w_1e31d24',
     'w_1e3ce01',
     'w_1e45a47',
     'w_1e4c0ec',
     'w_1e4e7a4',
     'w_1e5a146',
     'w_1e6559e',
     'w_1e674e6',
     'w_1e68ef5',
     'w_1e75406',
     'w_1e7bb93',
     'w_1e9d5c7',
     'w_1eaebf2',
     'w_1eafe46',
     'w_1ec0481',
     'w_1ec267f',
     'w_1ecfe96',
     'w_1ed4dde',
     'w_1efd2a9',
     'w_1eff98a',
     'w_1f00cb7',
     'w_1f09cdd',
     'w_1f0da94',
     'w_1f10750',
     'w_1f2a32a',
     'w_1f606cf',
     'w_1f6e1db',
     'w_1f95205',
     'w_1fa0ee5',
     'w_1facad2',
     'w_1fb42f1',
     'w_1fc14e9',
     'w_1fc874a',
     'w_1fd0d0e',
     'w_1febbf3',
     'w_200a203',
     'w_2018f6c',
     'w_2039053',
     'w_203e9d5',
     'w_20548a7',
     'w_206742f',
     'w_206e903',
     'w_2071a4c',
     'w_20848b7',
     'w_2085f2e',
     'w_208de0e',
     'w_20ba84b',
     'w_20c671c',
     'w_20da4cf',
     'w_20e863f',
     'w_20eb13c',
     'w_20f86e1',
     'w_2111212',
     'w_211aa88',
     'w_212ad82',
     'w_212b985',
     'w_214c09a',
     'w_2168906',
     'w_2168ae2',
     'w_216f61f',
     'w_2173953',
     'w_21745f8',
     'w_217e78a',
     'w_2184d07',
     'w_218b25e',
     'w_2196938',
     'w_219e33c',
     'w_21a1bdb',
     'w_21b3754',
     'w_21dfc18',
     'w_21e178f',
     'w_2216a46',
     'w_2222ef0',
     'w_222dcb7',
     'w_223cebd',
     'w_224000c',
     'w_2250fa9',
     'w_2270691',
     'w_227ed79',
     'w_2282bb8',
     'w_22b09d0',
     'w_22b8a16',
     'w_22bb9b3',
     'w_22bcbd6',
     'w_22d812d',
     'w_22da5b6',
     'w_22f0a7d',
     'w_22fa4ad',
     'w_230a0de',
     'w_231a0c8',
     'w_232e5bd',
     'w_2341d0a',
     'w_23526cb',
     'w_2356a6d',
     'w_235b156',
     'w_2386a5a',
     'w_238a4ad',
     'w_238bbbf',
     'w_2392b4c',
     'w_23b01a6',
     'w_23b9dcb',
     'w_23cd105',
     'w_23d3818',
     'w_23d3b1c',
     'w_23dce10',
     'w_23e4e61',
     'w_23e5a4c',
     'w_24212f5',
     'w_242a05d',
     'w_243e33e',
     'w_244bd92',
     'w_2466659',
     'w_24823e2',
     'w_2482b4a',
     'w_2486665',
     'w_248af0b',
     'w_24a2923',
     'w_24ac53d',
     'w_24ad46c',
     'w_24bdae3',
     'w_24e2654',
     'w_24fda6e',
     'w_25067c8',
     'w_251ee17',
     'w_2527790',
     'w_2554558',
     'w_257856f',
     'w_25808d2',
     'w_2581f2e',
     'w_25871da',
     'w_25a38b8',
     'w_25d7f93',
     'w_25db560',
     'w_25e2bc9',
     'w_25ec80a',
     'w_25fdcfb',
     'w_263e243',
     'w_2641733',
     'w_2658649',
     'w_2663598',
     'w_2669d75',
     'w_2687f1b',
     'w_269b090',
     'w_26bd720',
     'w_26dd948',
     'w_26edeb8',
     'w_26f9f95',
     'w_2707bcd',
     'w_2709cfc',
     'w_272259b',
     'w_2725793',
     'w_2730966',
     'w_2739d94',
     'w_2757f01',
     'w_27633c3',
     'w_27736a0',
     'w_279b255',
     'w_27b9e86',
     'w_27cf4e2',
     'w_27da7aa',
     'w_27fc5d3',
     'w_2811cea',
     'w_2832e90',
     'w_2850471',
     'w_2855124',
     'w_285da1a',
     'w_2863d51',
     'w_286ec5f',
     'w_2870b33',
     'w_2881995',
     'w_288e40b',
     'w_28c8f60',
     'w_28cd742',
     'w_28ce17c',
     'w_28fa29e',
     'w_2901987',
     'w_2901dbf',
     'w_290f82b',
     'w_293b5e4',
     'w_2941dd2',
     'w_2952678',
     'w_2957331',
     'w_295b96c',
     'w_2963114',
     'w_29670e2',
     'w_296749b',
     'w_29c286a',
     'w_29c9595',
     'w_29cc48b',
     'w_29cf24b',
     'w_29d2cec',
     'w_29eb3b2',
     'w_29f00ae',
     'w_29fc08a',
     'w_29fc831',
     'w_2a04ceb',
     'w_2a18a44',
     'w_2a1d45c',
     'w_2a32d3d',
     'w_2a51a27',
     'w_2a6d006',
     'w_2a939eb',
     'w_2a9727c',
     'w_2ac069f',
     'w_2ac6dd5',
     'w_2ac83b0',
     'w_2ac8b4d',
     'w_2af3059',
     'w_2b0028d',
     'w_2b0fc31',
     'w_2b1e2f5',
     'w_2b28681',
     'w_2b3fe8e',
     'w_2b443f8',
     'w_2b485a8',
     'w_2b4da16',
     'w_2b6cfa6',
     'w_2b8e175',
     'w_2b930da',
     'w_2b939eb',
     'w_2b969fe',
     'w_2bb8c37',
     'w_2bba8c8',
     'w_2bdd2df',
     'w_2be1c9c',
     'w_2bea4c4',
     'w_2c0af5e',
     'w_2c1634b',
     'w_2c19c4a',
     'w_2c1dafa',
     'w_2c3163b',
     'w_2c3f440',
     'w_2c46a73',
     'w_2c51d33',
     'w_2c55303',
     'w_2c68b75',
     'w_2c6fe6f',
     'w_2c717fb',
     'w_2c75202',
     'w_2c83e22',
     'w_2cadad2',
     'w_2cb8996',
     'w_2cca6a1',
     'w_2cde7c0',
     'w_2cdf820',
     'w_2ce5ce2',
     'w_2ce9f0c',
     'w_2ceab05',
     'w_2d09595',
     'w_2d1eb90',
     'w_2d25d1f',
     'w_2d29ddd',
     'w_2d2d7fe',
     'w_2d2de1d',
     'w_2d57dce',
     'w_2d963bb',
     'w_2d99a0c',
     'w_2dae424',
     'w_2db01d5',
     'w_2dbb0fe',
     'w_2dc2ef2',
     'w_2dcbf82',
     'w_2dce318',
     'w_2dd15d3',
     'w_2dea395',
     'w_2e10179',
     'w_2e27b77',
     'w_2e2ba59',
     'w_2e328e6',
     'w_2e42cc0',
     'w_2e4df76',
     'w_2e4fecc',
     'w_2e872af',
     'w_2e977e2',
     'w_2ea9744',
     'w_2eabd5a',
     'w_2eb180f',
     'w_2ef900c',
     'w_2f07b15',
     'w_2f1e886',
     'w_2f283f3',
     'w_2f54c3c',
     'w_2f6a962',
     'w_2f6ad07',
     'w_2f6e76c',
     'w_2f7753e',
     'w_2f81b85',
     'w_2f89dfe',
     'w_2f8aed7',
     'w_2f97ce6',
     'w_2fcb559',
     'w_2fd21ec',
     'w_2fd73d9',
     'w_2fd79c1',
     'w_2fdaa7f',
     'w_2fdee7b',
     'w_2fdf63b',
     'w_2fe43c7',
     'w_2ffed9c',
     'w_3023ee5',
     'w_3026ce2',
     'w_3027b8f',
     'w_302af0a',
     'w_302c025',
     'w_303518a',
     'w_3039e7a',
     'w_3050553',
     'w_305ed07',
     'w_3061bdd',
     'w_3063bf4',
     'w_30660bc',
     'w_307065e',
     'w_3076d8c',
     'w_308bd26',
     'w_3090f78',
     'w_309a2b3',
     'w_30cf2ca',
     'w_30d8376',
     'w_30dc5ce',
     'w_30f095d',
     'w_3136deb',
     'w_313b78a',
     'w_31446aa',
     'w_314d77b',
     'w_3166a4d',
     'w_3172910',
     'w_318e1d2',
     'w_3197568',
     'w_31980d6',
     'w_31a93e7',
     'w_31b020c',
     'w_31d3999',
     'w_31fb9f0',
     'w_32037e2',
     'w_3222bdb',
     'w_32401f6',
     'w_32475a1',
     'w_32602d9',
     'w_326e389',
     'w_3278f8c',
     'w_32915fe',
     'w_3297a52',
     'w_329e594',
     'w_32a920b',
     'w_32d99b4',
     'w_330286e',
     'w_33032d1',
     'w_330f897',
     'w_331df94',
     'w_3320e76',
     'w_3326396',
     'w_3331656',
     'w_3349c9d',
     'w_3350425',
     'w_3355365',
     'w_3380eb0',
     'w_338b130',
     'w_338d945',
     'w_33909b8',
     'w_33973bf',
     'w_339c8ae',
     'w_33b9360',
     'w_33c3ce1',
     'w_33c8db8',
     'w_33e0287',
     'w_33e7def',
     'w_33e89be',
     'w_33f5747',
     'w_33fc58d',
     'w_3411b9f',
     'w_3449e4f',
     'w_3455b23',
     'w_3461d6d',
     'w_347b648',
     'w_34801a4',
     'w_34942b2',
     'w_349fcbc',
     'w_34a0eab',
     'w_34c4927',
     'w_34c8690',
     'w_34fd328',
     'w_35063ed',
     'w_351a1e1',
     'w_35453e0',
     'w_3549f7a',
     'w_3565288',
     'w_3572b44',
     'w_3572e7e',
     'w_3580734',
     'w_35b01a5',
     'w_35c5d43',
     'w_35c8057',
     'w_35d4d83',
     'w_35eb420',
     'w_35f0fd9',
     'w_35ffe80',
     'w_3608443',
     'w_361e290',
     'w_3621c49',
     'w_3650949',
     'w_3664a22',
     'w_3674103',
     'w_367b996',
     'w_3694c7d',
     'w_3698eaf',
     'w_36a4e8b',
     'w_36a853c',
     'w_36ac97f',
     'w_36b6904',
     'w_36d665c',
     'w_36eb0b1',
     'w_37223f9',
     'w_372909c',
     'w_372ae75',
     'w_37307c4',
     'w_373593e',
     'w_37372db',
     'w_37374fa',
     'w_373c114',
     'w_3745f59',
     'w_37523b2',
     'w_376a413',
     'w_379785d',
     'w_379ba08',
     'w_37a7f78',
     'w_37bd99a',
     'w_37c23fa',
     'w_37dd956',
     'w_37f17ad',
     'w_38158d6',
     'w_381c6aa',
     'w_384e9ca',
     'w_38770cd',
     'w_3881300',
     'w_38954da',
     'w_389c788',
     'w_38a3f72',
     'w_38b1edd',
     'w_38bcd3e',
     'w_38de842',
     'w_38e088a',
     'w_38e4aae',
     'w_38ef4a3',
     'w_38efc2d',
     'w_38efce7',
     'w_38f39ed',
     'w_39043c9',
     'w_392bee3',
     'w_3947b78',
     'w_395d578',
     'w_395f5e7',
     'w_39776da',
     'w_397cb24',
     'w_397e3db',
     'w_398aa7f',
     'w_39a31a8',
     'w_39b22d9',
     'w_39c7462',
     'w_39cfc71',
     'w_39d2684',
     'w_39f2064',
     'w_3a47dba',
     'w_3a78626',
     'w_3a7d86d',
     'w_3a94311',
     'w_3a9ee71',
     'w_3aa1da4',
     'w_3aa2073',
     'w_3ac13d2',
     'w_3add848',
     'w_3ae3603',
     'w_3af4e73',
     'w_3b016f0',
     'w_3b02089',
     'w_3b0894d',
     'w_3b3b9b2',
     'w_3b483d3',
     'w_3b5403b',
     'w_3b89846',
     'w_3b8b9c7',
     'w_3b90f9b',
     'w_3b99025',
     'w_3bb210a',
     'w_3bc8a47',
     'w_3bf56d4',
     'w_3c0bbac',
     'w_3c0e79e',
     'w_3c27f42',
     'w_3c2d938',
     'w_3c2ec7a',
     'w_3c304db',
     'w_3c3267c',
     'w_3c3c632',
     'w_3c4062e',
     'w_3c7b2e6',
     'w_3c8eab1',
     'w_3c9f80b',
     'w_3ca387d',
     'w_3cada3d',
     'w_3cbccb7',
     'w_3cc4e90',
     'w_3cd6996',
     'w_3cdf114',
     'w_3ce788a',
     'w_3cf3853',
     'w_3cfeb1a',
     'w_3d0bc7a',
     'w_3d12652',
     'w_3d187b5',
     'w_3d1c2ef',
     'w_3d2724b',
     'w_3d3c0f9',
     'w_3d45268',
     'w_3d4900b',
     'w_3d66298',
     ...]




```python
#top5probs
#probs[top5probs].shape
ds=pd.DataFrame(probs)
ds.columns = data.classes
```


```python
ds.insert(0,'id',[out[5:] for out in data.test_ds.fnames])
```


```python
ds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>new_whale</th>
      <th>w_0013924</th>
      <th>w_001ebbc</th>
      <th>w_002222a</th>
      <th>w_002b682</th>
      <th>w_002dc11</th>
      <th>w_0087fdd</th>
      <th>w_008c602</th>
      <th>w_009dc00</th>
      <th>...</th>
      <th>w_ff70408</th>
      <th>w_ff7630a</th>
      <th>w_ff94ad6</th>
      <th>w_ffa7427</th>
      <th>w_ffa78a5</th>
      <th>w_ffb4e3d</th>
      <th>w_ffbd74c</th>
      <th>w_ffcd98e</th>
      <th>w_ffda8b2</th>
      <th>w_ffdab7a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c292553f</td>
      <td>0.022390</td>
      <td>0.000110</td>
      <td>0.000028</td>
      <td>0.000186</td>
      <td>0.000382</td>
      <td>0.000055</td>
      <td>0.000584</td>
      <td>0.000089</td>
      <td>0.000057</td>
      <td>...</td>
      <td>0.000120</td>
      <td>0.000200</td>
      <td>0.000021</td>
      <td>0.000160</td>
      <td>0.000027</td>
      <td>0.000091</td>
      <td>0.000125</td>
      <td>0.000118</td>
      <td>0.000105</td>
      <td>0.000096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c70ca9cc</td>
      <td>0.117190</td>
      <td>0.000122</td>
      <td>0.000028</td>
      <td>0.000306</td>
      <td>0.000268</td>
      <td>0.000048</td>
      <td>0.000045</td>
      <td>0.000271</td>
      <td>0.000674</td>
      <td>...</td>
      <td>0.000040</td>
      <td>0.000311</td>
      <td>0.000113</td>
      <td>0.000899</td>
      <td>0.000078</td>
      <td>0.000211</td>
      <td>0.000148</td>
      <td>0.000447</td>
      <td>0.000045</td>
      <td>0.000103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18ff383f</td>
      <td>0.102055</td>
      <td>0.000206</td>
      <td>0.000255</td>
      <td>0.000112</td>
      <td>0.000295</td>
      <td>0.000066</td>
      <td>0.000098</td>
      <td>0.000143</td>
      <td>0.000133</td>
      <td>...</td>
      <td>0.000187</td>
      <td>0.000153</td>
      <td>0.000184</td>
      <td>0.000281</td>
      <td>0.000051</td>
      <td>0.000133</td>
      <td>0.000081</td>
      <td>0.000095</td>
      <td>0.000026</td>
      <td>0.000071</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5802f268</td>
      <td>0.006519</td>
      <td>0.000101</td>
      <td>0.000097</td>
      <td>0.000031</td>
      <td>0.000019</td>
      <td>0.000507</td>
      <td>0.000103</td>
      <td>0.000032</td>
      <td>0.000007</td>
      <td>...</td>
      <td>0.000028</td>
      <td>0.000205</td>
      <td>0.000034</td>
      <td>0.000048</td>
      <td>0.000207</td>
      <td>0.000021</td>
      <td>0.000157</td>
      <td>0.000121</td>
      <td>0.000052</td>
      <td>0.000027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>080f5fab</td>
      <td>0.014350</td>
      <td>0.000402</td>
      <td>0.000331</td>
      <td>0.000043</td>
      <td>0.000222</td>
      <td>0.000056</td>
      <td>0.000204</td>
      <td>0.000200</td>
      <td>0.000020</td>
      <td>...</td>
      <td>0.000046</td>
      <td>0.000162</td>
      <td>0.000249</td>
      <td>0.000071</td>
      <td>0.000201</td>
      <td>0.000042</td>
      <td>0.000094</td>
      <td>0.000123</td>
      <td>0.000019</td>
      <td>0.000155</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4252 columns</p>
</div>




```python
ds.to_csv('whale_sub.gz',compression='gzip',index=False)
```


```python
FileLink('whale_sub.gz')
```




<a href='whale_sub.gz' target='_blank'>whale_sub.gz</a><br>



### End
