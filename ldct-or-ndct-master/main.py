#import glob
#ldct_files = glob.glob("/data/CT_data/images/ldct_1e5/*.flt")
#ndct_files = glob.glob("/data/CT_data/images/ndct/*.flt")
#print(ldct_files[:10])
#print(ndct_files[:10])
import os

# Directory with our training ldct pictures
#train_ldct_dir = os.path.join("/data/CT_data/images/ldct_1e5")

train_ldct_dir = os.path.join("CT_data_png/ldct_1e5/train")
print(train_ldct_dir)
# Directory with our training ndct pictures
#train_ndct_dir = os.path.join("/data/CT_data/images/ndct")

train_ndct_dir = os.path.join("CT_data_png/ldct_1e5/train")
print(train_ndct_dir)

train_ldct_names = os.listdir(train_ldct_dir)
print(train_ldct_names[:10])

train_ndct_names = os.listdir(train_ndct_dir)
print(train_ndct_names[:10])


print('total training ldct images:', len(os.listdir(train_ldct_dir)))
print('total training ndct images:', len(os.listdir(train_ndct_dir)))
#---------
#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

from IPython import get_ipython
ipython = get_ipython()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
#---------

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_ldct_pix = [os.path.join(train_ldct_dir, fname) 
                for fname in train_ldct_names[pic_index-8:pic_index]]
next_ndct_pix = [os.path.join(train_ndct_dir, fname) 
                for fname in train_ndct_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_ldct_pix+next_ndct_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
