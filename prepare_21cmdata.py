import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os
import re
import numpy as np
import random
from PIL import Image
from torch import dtype
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


def prepare21cm(
    env, datapath, size=[64]
):
    # get the filenames in the datapath
    filenames=[]
    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))

    #check the dimension of data
    datacache=np.load(filenames[0])
    if datacache.ndim!=3:
        print('Please check the dimension of 21cm cube, should be 3D')
        exit
    slices=np.shape(datacache)[0]

    #load the 21cm datacubes
    total=0
    for filename in filenames:
        datacache=np.load(filename).astype(np.float32)
        #in many cases we dont neet to do the normalization here, the 21cm signal is always less than 255, thus PIL can handle this
        #datacache=datacache/(3*np.std(datacache))
        for i in range(slices):
            key = f"{size[0]}-{str(total).zfill(6)}".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, datacache[i])
            total += 1

    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

# this function is specially designed for my dataset 
def prepareconditional21cm(
    env, size=[64,512],direc = '/scratch/dkn16/dataset_lc512/data-tbandrho'
):
    
    # get the filenames in the datapath. Here we can directly write the code
    filenames=[]
    # first we walk all the filenames
    for root, dirs, files in os.walk(direc):
        for i in range(len(files)):
            filenames.append(os.path.join(root,files[i]))
            #print(filenames[i])

    total_1=int(len(filenames)*0.8)
    random.seed(2)
    random.shuffle(filenames)


    print(total_1)
    print(filenames[0])
    total=0
    pattern = re.compile(r'\d+.\d+')


    #check the dimension of data
    datacache=np.load(filenames[0])
    if datacache.ndim!=4:
        print('Please check the dimension of 21cm cube, should be 3D plus one channel dimension, in total 4')
        return 0
    #tb_cache=datacache[0,::32,:0+size[0],0:0+size[1]]/30.
    #print(tb_cache.shape)
    slices=2
    width=np.shape(datacache)[2]
    length=np.shape(datacache)[3]
    #check if we use the right 'size' parameter
    if width<size[0] or length!=size[1]:
        print('Dataset resolution does not match the size parameter. Please check the resolution of your dataset.')
        return 0

    #load the 21cm datacubes
    
    for i in range(args.num):
        datacube=np.load(filenames[i]).astype(np.float32)
        #datacube = datacube[0:2,...].copy()
        datacube=datacube.transpose(1,0,2,3)
        
        tb_cache=datacube[::128,0:2,0:0+size[0],0:0+size[1]]
        tb_cache[:,0,...]= tb_cache[:,0,...]/30.
        #tb_cache[:,1,...]= (tb_cache[:,1,...])/13.
        #tb_cache[:,1,...]= (tb_cache[:,1,...])/np.mean(datacube[:,2,...],axis=(0,1))
        tb_cache[:,1,...]=(tb_cache[:,1,...]+1.)/2.-1
        #rho_cache=datacube[1,::32,:0+size[0],0:0+size[1]]/30.
        result = pattern.findall(filenames[i])
        print(result)
        #return 0
        paramcache =  np.array([float(result[2]),np.log10(float(result[1]))])
        paramcache=(paramcache-np.array([4.,1.]))/np.array([2.,1.398]).astype(np.float64)
        #print(paramcache)
        #in many cases we dont need to do the normalization here, the 21cm signal is always less than 255, thus PIL can handle this
        #datacache=datacache/(3*np.std(datacache))
        for j in range(slices):
            key = f"{size[0]}-{str(total).zfill(6)}_tb".encode("utf-8")
            #key_rho = f"{size[0]}-{str(total).zfill(6)}_rho".encode("utf-8")
            key_label = f"{size[0]}-{str(total).zfill(6)}_label".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, tb_cache[j].copy())
            #with env.begin(write=True) as txn:
            #    txn.put(key_rho, rho_cache[i].copy())
            with env.begin(write=True) as txn:
                txn.put(key_label,paramcache.copy())
            total += 1
        datacube=np.load(filenames[i]).astype(np.float32)
        datacube=datacube.transpose(2,0,1,3)
        tb_cache=datacube[::128,0:2,:0+size[0],0:0+size[1]]
        tb_cache[:,0,...]= tb_cache[:,0,...]/30.
        #tb_cache[:,1,...]= (tb_cache[:,1,...])/13.
        #print(tb_cache[:,1,...].shape)
        #tb_cache[:,1,...]= (tb_cache[:,1,...])/np.mean(datacube[:,2,...],axis=(0,1))
        tb_cache[:,1,...]=(tb_cache[:,1,...]+1.)/2.-1
        #rho_cache=datacube[1,::32,:0+size[0],0:0+size[1]]/30.
        for j in range(slices):
            key = f"{size[0]}-{str(total).zfill(6)}_tb".encode("utf-8")
            #key_rho = f"{size[0]}-{str(total).zfill(6)}_rho".encode("utf-8")
            key_label = f"{size[0]}-{str(total).zfill(6)}_label".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, tb_cache[j].copy())
            #with env.begin(write=True) as txn:
            #    txn.put(key_rho, rho_cache[i].copy())
            with env.begin(write=True) as txn:
                txn.put(key_label,paramcache.copy())
            total += 1
    print(total)
    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="64,64",
        help="resolutions of images for the dataset",
    )
    parser.add_argument("--num", type=int,default=80, help="number of images in lmdb dataset")
    #deleted the multiprocessing argument, because i'm too stupid to use multiprocessing
    #deleted the resampling argument, not needed anymore
    parser.add_argument("path", type=str, help="path to the image dataset")
    #a parameter for making conditional dataset
    parser.add_argument("--cond", type=bool, help="path to the image dataset",default=False)

    args = parser.parse_args()

    #resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    #resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = args.path

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        if args.cond:
            prepareconditional21cm(env,  size=sizes)
        else:
            prepare21cm(env, imgset,size=sizes)
