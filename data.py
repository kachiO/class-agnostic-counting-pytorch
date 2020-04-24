from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.ndimage import gaussian_filter, affine_transform
from PIL import Image
from torchvision.transforms import Normalize, RandomAffine, RandomHorizontalFlip, CenterCrop, ToTensor, Compose
import numpy as np, torch, json
from pathlib import Path
from utils import reorganize_annotations_by_filename

IMAGENET_VID_DIMS = dict(
    image = (255, 255, 3),
    patch = (63, 63, 3),
    output = (64, 64, 1))

DATA_MAP = {'a': 'ILSVRC2015_VID_train_0000',
            'b': 'ILSVRC2015_VID_train_0001',
            'c': 'ILSVRC2015_VID_train_0002',
            'd': 'ILSVRC2015_VID_train_0003',
            'e': 'val'}

DATA_ROOT = './dataset/Imagenet_Video_2015/ILSVRC2015/crop_255_exemplar_63_context_0.1/'

IMAGENET_STATS = dict(mean = (0.485, 0.456, 0.406),
                      std = (0.229, 0.224, 0.225))

generic_transforms = [ToTensor(), Normalize(**IMAGENET_STATS, inplace=True), ]
patch_transforms = [RandomAffine(degrees=(8, 20), translate=[0.1, 0.1], scale=[0.95, 1.05]),
                    RandomHorizontalFlip(0.5),
                    CenterCrop(IMAGENET_VID_DIMS['patch'][:2]),
                    ]
patch_tfms = Compose(patch_transforms + generic_transforms)
generic_tfms = Compose(generic_transforms)

class ImagenetVidDatatset(Dataset):
    def __init__(self, data_root=DATA_ROOT, data_meta_dir='../imagenet_vid/', mode='train',
                 dims=IMAGENET_VID_DIMS, p_match=0.5, patch_augment=True, imagenet_norm=True):
        self.root = data_root
        self.dims = dims
        self.p_match = p_match
        self.patch_augment = patch_augment
        assert mode in ['train', 'valid']
        self.mode = mode
        self.data = np.load(Path(data_meta_dir) / f'{self.mode}.npy', allow_pickle=True)

        positive = np.zeros(self.dims['output'])
        positive[self.dims['output'][0] // 2, self.dims['output'][1] // 2, 0] = 1
        positive[:, :, 0] = 100 * gaussian_filter(positive[:, :, 0], sigma=(2, 2), mode='constant')
        self.output = np.concatenate((positive, np.zeros(self.dims['output'])), -1)
        self.tfms = generic_tfms
        self.patch_tfms = patch_tfms if self.patch_augment else generic_tfms
        self.imagenet_norm = imagenet_norm
        self.data_inds = {ii: list(range(len(dd))) for ii, dd in enumerate(self.data)}

    def __len__(self):
        return len(self.data)

    def len_total(self):
        return int(np.sum([len(d) for d in self.data]))

    def __getitem__(self, index):
        if self.mode == 'train':
            input_obj = np.random.choice(self.data[index])
        elif self.mode == 'valid':
            sub_index = self.data_inds[index][0]
            self.data_inds[index] = self.data_inds[index][1:]
            input_obj = self.data[index][sub_index]

        _video_dir, object_id, frames = Path(input_obj[0]), input_obj[1], input_obj[2]
        video_dir = Path(self.root) / DATA_MAP[str(_video_dir.parent)] / _video_dir.name

        if np.random.rand() < self.p_match:
            # positive pair
            patch_obj = input_obj
            _patch_dir, patch_object_id, patch_frames = Path(input_obj[0]), input_obj[1], input_obj[2]
            
            # choose two frames at most 100 frames apart
            start_frame = np.random.randint(max(1, len(frames) - 100))
            frame_in, frame_ex = np.random.choice(frames[start_frame : start_frame + 100], 2)
            output_map = self.output[:, :, 0]
            match = True
        else:
            # negative pair
            new_index = np.random.choice(list(set(np.arange(len(self.data))) - set([index])))
            patch_obj = np.random.choice(self.data[new_index])
            _patch_dir, patch_object_id, patch_frames = Path(patch_obj[0]), patch_obj[1], patch_obj[2]
                                                         
            frame_in = np.random.choice(frames)
            frame_ex = np.random.choice(patch_frames)
            output_map = self.output[:, :, 1]
            match = False

        input_fn = video_dir / f'{frame_in:06}.{object_id:02}.x.jpg'
        patch_dir = Path(self.root) / DATA_MAP[str(_patch_dir.parent)] / _patch_dir.name
        patch_fn = patch_dir / f"{frame_ex:06}.{patch_object_id:02}.{'x' if self.patch_augment else 'z'}.jpg"

        img_input = Image.open(input_fn)
        img_patch = Image.open(patch_fn)

        if self.imagenet_norm:
            img_input = self.tfms(img_input)
            img_patch = self.patch_tfms(img_patch)

        output = {'search_img': img_input,
                  'patch_img': img_patch,
                  'output_map': torch.as_tensor(output_map, dtype=torch.float)[None],
                  'match': torch.as_tensor(match, dtype=torch.float)
                 }

        return output


def collate_fn(batch):
    out_batch = dict()
    search_imgs = torch.stack([el['search_img'] for el in batch])
    patch_imgs = torch.stack([el['patch_img'] for el in batch])
    out_batch['images'] = (search_imgs, patch_imgs)
    out_batch['match'] = torch.stack([el['match'] for el in batch])
    out_batch['targets'] = torch.stack([el['output_map'] for el in batch]) 
        
    return out_batch


class ValidSamplerSubset(Sampler):
    """
    Samples subset ('num_samples') from each class
    """
    def __init__(self, data_source, num_samples=100):
        self.data_source = data_source
        self.num_samples = num_samples
        self._reps = torch.arange(len(self.data_source)).repeat_interleave(num_samples)
    def __len__(self): 
        return len(self._reps)
    
    def __iter__(self):
        return iter(self._reps)
    

class ValidSampler(Sampler):
    """
    Samples all images from each class
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self._inds = np.concatenate([len(d) * [i] for i, d in enumerate(self.data_source.data)]).tolist()
    def __len__(self): 
        return len(self._inds)
    
    def __iter__(self):
        return iter(self._inds)
    
DATA_DIR = './dataset/iSAID/Processed/Patches/patch_W800_patch_H800_overlap200/train/'
ANNOT_FILE = 'instancesonly_filtered_train.json'

class DOTADataset(Dataset):
    def __init__(self, data_dir=None, annots_file=None, transform=False):
        self.data_dir = DATA_DIR if data_dir is None else data_dir
        self.annots = Path(DATA_DIR) / ANNOT_FILE if annots_file is None else annots_file
        self.transform = transform
        
        if self.transform:
            self.tfms = tfms.Compose([tfms.ToTensor(), tfms.Normalize(**IMAGENET_STATS)])
            
        with open(self.annots, 'r') as f:
            self.file = json.load(f)
        
        self.annotations_by_filename = reorganize_annotations_by_filename(self.file)

    def __len__(self):
        return len(self.file['images'])
    
    def _getdata(self, index):
        fn = self.file['images'][index]['file_name']
        annotations = self.annotations_by_filename[fn]
        
        if not len(annotations):
            return self._getdata(np.random.choice(len(self)))
        
        return fn, annotations
            
    def __getitem__(self, index):
        self.fn, self.annotations = self._getdata(index)
        
        image = Image.open(Path(self.data_dir) / 'images'/ self.fn)    
        cat_ids = {a['category_id']: a['category_name'] for a in self.annotations}
        
        # select patch
        a_ind = np.random.choice(self.annotations)
        x, y, w, h = a_ind['bbox']
        context  = (w + h ) * 0.1
        bbox = [max(x - context, 0), max(y - context, 0), min(x + w + context, 800), min(y + h + context, 800)]
        patch = image.crop(bbox).resize((63, 63))
        patch_cat = a_ind['category_name']
        
        if self.transform:
            patch = self.tfms(patch)
            image = self.tfms(image)
            
        centers = [((2 * a['bbox'][1] + a['bbox'][3]) // 2, (2 * a['bbox'][0] + a['bbox'][2]) // 2) 
                        for a in self.annotations if a['category_name'] == a_ind['category_name']]

        centers = torch.tensor(centers, dtype=torch.long)
        dot_annot = torch.zeros((800, 800), dtype=torch.float)

        for coord in centers: 
            dot_annot[coord[0], coord[1]] = 1.
            
        dot_annot_blob = 100 * gaussian_filter(dot_annot.numpy(), sigma=(4, 4), mode='constant')
        
        output = {'image': image, 
                  'patch': patch, 
                  'gt': torch.tensor(dot_annot_blob, dtype=torch.float),
                  'patch_category': patch_cat}
    
        return output