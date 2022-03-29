from io import BytesIO
import numpy as np
import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(6)}'.encode('utf-8')
            # binary files in database
            #now change to numpy array
            img_str = txn.get(key)
            img_np=np.fromstring(img_str,dtype=np.float32).reshape((self.resolution,self.resolution))

        #first read the binary file
        #buffer = BytesIO(img_bytes)
        #then open the binary file, get an image object
        #img = Image.open(buffer)
        #apply some transform, which is useless in our case.
        img = self.transform(img_np)

        return img
