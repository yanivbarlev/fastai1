#!/usr/bin/env python
# coding: utf-8

# In[1]:


key = '79a6a1db946c48b58b3174564c454791'


# In[2]:


def search_images_bing2(key, term, min_sz=128,cnt=150):
    client = api('https://api.cognitive.microsoft.com', auth(key))
    return L(client.images.search(query=term, count=cnt, min_height=min_sz, min_width=min_sz).value)


# In[3]:


import shutil


# In[4]:


from fastai.vision.widgets import *


# In[5]:


from fastbook import *


# In[6]:


results = search_images_bing(key, 'grizzly bear')


# In[7]:


get_ipython().run_line_magic('pinfo2', 'search_images_bing')


# In[8]:


ims = results.attrgot('content_url')


# In[9]:


men_types = 'beautiful','normal','ugly'


# In[10]:


path = Path('men')


# In[11]:


#shutil.rmtree('men')


# In[ ]:





# In[12]:


if not path.exists():
    path.mkdir()
    for o in men_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing2(key,f'{o} men',cnt=500)
        download_images(dest,urls=results.attrgot('content_url'))
        #delete bad images
        print(o)
        fns = get_image_files(Path(path/o))
        failed = verify_images(fns)
        print(failed)
        failed.map(Path.unlink)
        


# In[23]:


#getting help
get_ipython().run_line_magic('pinfo2', 'search_images_bing')


# In[13]:


men = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[14]:


dls = men.dataloaders(path)


# In[15]:


dls.valid.show_batch(max_n=4,nrows=1)


# In[16]:


men = men.new(item_tfms = RandomResizedCrop(224,min_scale=0.5), batch_tfms = aug_transforms())


# In[17]:


dls= men.dataloaders(path)


# In[18]:


learn = cnn_learner(dls,resnet18,metrics=error_rate)


# In[2]:


learn.fine_tune(4)

