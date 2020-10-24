# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:27:07 2020

@author: Seungbeom Ha
"""
# Setup envionment

import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *


# Grizzly Bear picture download
# Generate key for Bing image search Azure
def main():
    key = os.environ.get('AZURE_SEARCH_KEY', 'b6008d982b22455287c832284cd58b22')
    print('Key for Bing image search generated')
    
    
    # #Search images of grizzly bear from Bing image search
    # results = search_images_bing(key, 'grizzly bear')
    # ims = results.attrgot('content_url')
    # print('Images Gathered')
    
    # # Let's look at one of the images:
    # ims = ['http://2.bp.blogspot.com/-NjMTuklENdE/UHzVv_8dIxI/AAAAAAAAA-U/tNBsQDn8kFI/s1600/Grizzly+Bear+Pic.jpg']
    # dest = 'C:/Users/tmdqj/OneDrive - The University of Texas at Austin/Machine Learning/FastAi Machine Learning Code/grizzle.jpg'
    # download_url(ims[0], dest)
    # im = Image.open(dest)
    # im.to_thumb(128, 128)
    
    
    
    # Download all bear related picture from Bing
    # and store it in the local files
    bear_types = 'grizzly', 'black', 'teddy'
    path = Path('bears')
    
    if not path.exists():
        path.mkdir()
        for o in bear_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_bing(key, f'{o} bear')
            download_images(dest, urls=results.attrgot('content_url'))
    print("Three path created")
    fns = get_image_files(path)
    fns
    print(len(fns))
    
    #Check if there are any corrupted files
    failed = verify_images(fns)
    failed.map(Path.unlink);
        
    
    
    # Transform downloaded data into Machine learning input object with FastAi
    # DataLoaders class
    
    #Make DataBlock object
    bears = DataBlock(
        blocks = (ImageBlock, CategoryBlock),
        get_items = get_image_files,
        splitter = RandomSplitter(valid_pct= 0.2, seed = 42),
        get_y = parent_label,
        item_tfms=Resize(128)) #Resize each images into same sizes so that we can feed it to our model
    
    #Tell FastAi the actual source of data, a.k.a, path where the images can be found
    dls = bears.dataloaders(path)
    dls.valid.show_batch(max_n=4, nrows=1)
    
    #display(HTML(dls.valid.show_batch(max_n=4, nrows=1)))
    #Squish or stretch the images
    bears = bears.new(item_tfms=Resize(100, ResizeMethod.Squish))
    dls = bears.dataloaders(path)
    dls.valid.show_batch(max_n=4, nrows=1)
    
    #pad the images with the black frame
    bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
    dls = bears.dataloaders(path)
    dls.valid.show_batch(max_n=4, nrows=1)
    
    # crop into same images into different portion
    bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
    dls = bears.dataloaders(path)
    dls.train.show_batch(max_n=4, nrows=1, unique=True)
    
    
    
    #DATA Augmentation
    bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
    dls = bears.dataloaders(path)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)


    # Training The Model and Using it to Clean your Data
    
    # Modify dataset with RandomREsizedCrop and Data Augmentation(aug_trnasform)
    bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
    dls = bears.dataloaders(path, num_workers = 0)
    
    learn=cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    
    # Let's see if our model made any mistake with confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    
    print(learn)
    
    # Export the ML model(learn) as pkl file
    learn.export()
    path=Path()
    path.ls(file_exts='.pkl')
    
    learn_inf = load_learner(path/'export.pkl')
    print(learn_inf.predict('grizzle.jpg'))
    
    

if __name__ == "__main__":
    main()

