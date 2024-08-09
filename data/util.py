'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image
import json
import random
import numpy as np
from copy import deepcopy

def load_json(json_path: str):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            try:
                json_file = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f".json does not exist.\nPath: {json_path}")
    
    return json_file

class MixupModule():
    def __init__(self, dataset, num_class) -> None:
        self.dataset = dataset
        self.num_class = num_class

    # Mixup functions
    def type1(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        if self.dataset.origin2gene[base_sample]:
            generated_sample = random.choice(self.dataset.origin2gene[base_sample])
            return [self.dataset.data[index], os.path.join(self.dataset.preproc_root, generated_sample)], [class_idx, class_idx]
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]
        
    def type2(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_conflict_attr = random.choice(list(self.dataset.itg_tag_stats[class_idx]['bias_conflict_tags']))
        if self.dataset.original_class_bias_stats[class_idx][bias_conflict_attr]:
            c_pos_sample = random.choice(self.dataset.original_class_bias_stats[class_idx][bias_conflict_attr])
            return [self.dataset.data[index], os.path.join(self.dataset.root, c_pos_sample)], [class_idx, class_idx]
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]

    def type3(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_attr = list(self.dataset.itg_tag_stats[class_idx]['bias_tags'])[0]
        if self.dataset.original_class_bias_stats[class_idx][bias_attr]:
            pos_sample = random.choice(self.dataset.original_class_bias_stats[class_idx][bias_attr])
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]
        
        if self.dataset.origin2gene[pos_sample]:
            generated_sample = random.choice(self.dataset.origin2gene[pos_sample])
            return [self.dataset.data[index], os.path.join(self.dataset.preproc_root, generated_sample)], [class_idx, class_idx]
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]
        
    def type4(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_conflict_attr1 = random.choice(list(self.dataset.itg_tag_stats[class_idx]['bias_conflict_tags']))
        bias_conflict_attr2 = random.choice(list(self.dataset.itg_tag_stats[class_idx]['bias_conflict_tags']))
        generated_sample1 = random.choice(self.dataset.generated_class_bias_stats[class_idx][bias_conflict_attr1])
        generated_sample2 = random.choice(self.dataset.generated_class_bias_stats[class_idx][bias_conflict_attr2])

        return [os.path.join(self.dataset.preproc_root, generated_sample1), os.path.join(self.dataset.preproc_root, generated_sample2)], [class_idx, class_idx]

    def type5(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_attr = list(self.dataset.itg_tag_stats[class_idx]['bias_tags'])[0]

        class_candidates = [str(i) for i in range(self.num_class)]
        class_candidates.remove(class_idx)
        c_neg_class = random.choice(class_candidates)

        c_neg_sample_candidates = self.dataset.original_class_bias_stats[c_neg_class][bias_attr]
        if c_neg_sample_candidates:
            c_neg_sample = random.choice(c_neg_sample_candidates)
            return [self.dataset.data[index], os.path.join(self.dataset.root, c_neg_sample)], [class_idx, c_neg_class]
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]
        
    def type6(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_attr = list(self.dataset.itg_tag_stats[class_idx]['bias_tags'])[0]

        class_candidates = [str(i) for i in range(self.num_class)]
        class_candidates.remove(class_idx)
        c_neg_class = random.choice(class_candidates)

        c_neg_generated_candidates = self.dataset.generated_class_bias_stats[c_neg_class][bias_attr]
        if c_neg_generated_candidates:
            neg_sample = random.choice(c_neg_generated_candidates)
            return [self.dataset.data[index], os.path.join(self.dataset.preproc_root, neg_sample)], [class_idx, c_neg_class]
        else:
            return [self.dataset.data[index], self.dataset.data[index]], [class_idx, class_idx]
        
    def type7(self, index):
        base_sample = self.dataset.data[index].replace(self.dataset.root+'/', '')
        class_idx = str(base_sample.split('_')[-2])
        bias_attr = list(self.dataset.itg_tag_stats[class_idx]['bias_tags'])[0]

        class_candidates = [str(i) for i in range(self.num_class)]
        class_candidates.remove(class_idx)
        c_neg_class1 = random.choice(class_candidates)
        c_neg_class2 = random.choice(class_candidates)

        c_neg_generated_candidates1 = self.dataset.generated_class_bias_stats[c_neg_class1][bias_attr]
        if c_neg_generated_candidates1:
            c_neg_sample1 = os.path.join(self.dataset.preproc_root, random.choice(c_neg_generated_candidates1))
            c_neg_sample1_label = c_neg_class1
        else:
            c_neg_sample1 = self.dataset.data[index]
            c_neg_sample1_label = class_idx

        c_neg_generated_candidates2 = self.dataset.generated_class_bias_stats[c_neg_class2][bias_attr]
        if c_neg_generated_candidates2:
            c_neg_sample2 = os.path.join(self.dataset.preproc_root, random.choice(c_neg_generated_candidates2))
            c_neg_sample2_label = c_neg_class2
        else:
            c_neg_sample2 = self.dataset.data[index]
            c_neg_sample2_label = class_idx

        return [c_neg_sample1, c_neg_sample2], [c_neg_sample1_label, c_neg_sample2_label]
    
    def mixup(self, index, p, lam, mix_ratio=0.5):
        if p < mix_ratio:
            mixup_functions = [self.type1, self.type2, self.type3, self.type4]
        else:
            mixup_functions = [self.type5, self.type6, self.type7]

        mixup_function = random.choice(mixup_functions)
        samples, labels = mixup_function(index)

        image0 = Image.open(samples[0]).convert('RGB')
        image1 = Image.open(samples[1]).convert('RGB')

        if self.dataset.transform is not None:
            image0 = self.dataset.transform(image0)
            image1 = self.dataset.transform(image1)

        y0 = torch.nn.functional.one_hot(torch.tensor(int(labels[0])), num_classes=self.num_class)
        y1 = torch.nn.functional.one_hot(torch.tensor(int(labels[1])), num_classes=self.num_class)

        # # mixed_input = l * x + (1 - l) * x2
        # mixed_x = torch.tensor(lam, dtype=torch.float32).to(image0.device) * image0 + torch.tensor(1-lam, dtype=torch.float32).to(image1.device) * image1
        # mixed_y = torch.tensor(lam, dtype=torch.float32).to(y0.device) * y0 + torch.tensor(1-lam, dtype=torch.float32).to(y1.device) * y1

        mixed_x = lam * image0 + (1 - lam) * image1
        mixed_y = lam * y0 + (1 - lam) * y1
        
        return mixed_x, mixed_y


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item
    
class CMNISTDataset(Dataset):
    def __init__(self,
                 args,
                 root, 
                 split,
                 origin_only,
                 transform=None, 
                 image_path_list=None,
                 preproc_root=None,
                 ):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.preproc_root = preproc_root
        self.mixup = args.mixup
        
        if split == 'train' and args.ours and not origin_only:
            self.img2attr = {}
            original_class_bias_stats_path = os.path.join(preproc_root, 'original_class_bias_stats.json')
            self.original_class_bias_stats = load_json(original_class_bias_stats_path)
            
            generated_class_bias_stats_path = os.path.join(preproc_root, 'generated_class_bias_stats.json')
            self.generated_class_bias_stats = load_json(generated_class_bias_stats_path) # generated_class_bias_stats[class][bias_attr]
            
            itg_tag_stats_path = os.path.join(preproc_root, 'tag_stats.json')
            self.itg_tag_stats = load_json(itg_tag_stats_path)
            
            origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
            self.origin2gene = load_json(origin2gene_path)
            
            self.class_biases = [self.itg_tag_stats[str(class_idx)]['bias_tags'] for class_idx in range(10)]
            self.class_biases.append('none')
            self.class_biases = list(set([bias for sublist in self.class_biases for bias in sublist]))
            
            for class_idx in self.original_class_bias_stats:
                for bias_attr in self.original_class_bias_stats[class_idx]:
                    for sample_path in self.original_class_bias_stats[class_idx][bias_attr]:
                        self.img2attr[sample_path] = bias_attr

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
    
            if not origin_only:
                if args.half_generated:
                    # return origin + generated(matched 1:1) -> generated ratio ~= biased ratio
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]: continue
                        tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                        self.generated_data.append(tmp_gene_data)
                    self.data += self.generated_data

                elif args.only_no_tags:
                    # return only non-bias-tag samples
                    self.no_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    self.data = self.no_tags

                elif args.only_tags:
                    # retrun only bias-tag samples(origin)
                    self.yes_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if self.origin2gene[image_key]:
                            self.yes_tags.append(data)
                    self.data = self.yes_tags

                elif args.only_no_tags_balanced:
                    # return non-bias-tags sample + generated sample(# non-bias-tags * # classes)
                    self.no_tags = []
                    self.generated_data = []
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.tmp_generated_data = self.generated_align + self.generated_conflict
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    for _ in range(len(self.no_tags) * 10): # num_class 10
                        choosen_gene_data = random.choice(self.tmp_generated_data)
                        self.generated_data.append(choosen_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif args.no_tags_gene:
                    # return non-bias-tags sample + generated sample(# bias-tags)
                    self.no_tags = []
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                        else:
                            tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                            self.generated_data.append(tmp_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif not args.half_generated and args.include_generated:
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.generated_data = self.generated_align + self.generated_conflict
                    self.data += self.generated_data
            
        elif split=='valid':
            self.data = glob(os.path.join(root, split, '*'))
        elif split=='test':
            self.data = glob(os.path.join(root, '../test', '*', '*'))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]

class bFFHQDataset(Dataset):   
    def __init__(self,
                 args,
                 root, 
                 split,
                 origin_only,
                 transform=None, 
                 image_path_list=None,
                 preproc_root=None,
                 ):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.preproc_root = preproc_root
        self.mixup = args.mixup
        
        if split == 'train' and args.ours and not origin_only:
            self.img2attr = {}
            original_class_bias_stats_path = os.path.join(preproc_root, 'original_class_bias_stats.json')
            self.original_class_bias_stats = load_json(original_class_bias_stats_path)
            
            generated_class_bias_stats_path = os.path.join(preproc_root, 'generated_class_bias_stats.json')
            self.generated_class_bias_stats = load_json(generated_class_bias_stats_path) # generated_class_bias_stats[class][bias_attr]
            
            itg_tag_stats_path = os.path.join(preproc_root, 'tag_stats.json')
            self.itg_tag_stats = load_json(itg_tag_stats_path)
            
            origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
            self.origin2gene = load_json(origin2gene_path)
            
            self.class_biases = [self.itg_tag_stats[str(class_idx)]['bias_tags'] for class_idx in range(2)]
            self.class_biases.append('none')
            self.class_biases = list(set([bias for sublist in self.class_biases for bias in sublist]))
            
            for class_idx in self.original_class_bias_stats:
                for bias_attr in self.original_class_bias_stats[class_idx]:
                    for sample_path in self.original_class_bias_stats[class_idx][bias_attr]:
                        self.img2attr[sample_path] = bias_attr
                        
        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            
            if not origin_only:
                if args.half_generated:
                    # return origin + generated(matched 1:1) -> generated ratio ~= biased ratio
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]: continue
                        tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                        self.generated_data.append(tmp_gene_data)
                    self.data += self.generated_data

                elif args.only_no_tags:
                    # return only non-bias-tag samples
                    self.no_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    self.data = self.no_tags

                elif args.only_tags:
                    # retrun only bias-tag samples(origin)
                    self.yes_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if self.origin2gene[image_key]:
                            self.yes_tags.append(data)
                    self.data = self.yes_tags

                elif args.only_no_tags_balanced:
                    # return non-bias-tags sample + generated sample(# non-bias-tags * # classes)
                    self.no_tags = []
                    self.generated_data = []
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.tmp_generated_data = self.generated_align + self.generated_conflict
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    for _ in range(len(self.no_tags) * 2): # num_class 2
                        choosen_gene_data = random.choice(self.tmp_generated_data)
                        self.generated_data.append(choosen_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif args.no_tags_gene:
                    # return non-bias-tags sample + generated sample(# bias-tags)
                    self.no_tags = []
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                        else:
                            tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                            self.generated_data.append(tmp_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif not args.half_generated and args.include_generated:
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.generated_data = self.generated_align + self.generated_conflict
                    self.data += self.generated_data

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, '*'))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, '*'))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]

class BARDataset(Dataset):
    def __init__(self,
                 args,
                 root, 
                 split,
                 origin_only,
                 transform=None, 
                 image_path_list=None,
                 preproc_root=None,
                 ):
        super(BARDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.preproc_root = preproc_root
        self.mixup = args.mixup
        
        if split == 'train' and args.ours and not origin_only:
            self.img2attr = {}
            original_class_bias_stats_path = os.path.join(preproc_root, 'original_class_bias_stats.json')
            self.original_class_bias_stats = load_json(original_class_bias_stats_path)
            
            generated_class_bias_stats_path = os.path.join(preproc_root, 'generated_class_bias_stats.json')
            self.generated_class_bias_stats = load_json(generated_class_bias_stats_path) # generated_class_bias_stats[class][bias_attr]
            
            itg_tag_stats_path = os.path.join(preproc_root, 'tag_stats.json')
            self.itg_tag_stats = load_json(itg_tag_stats_path)
            
            origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
            self.origin2gene = load_json(origin2gene_path)
            
            self.class_biases = [self.itg_tag_stats[str(class_idx)]['bias_tags'] for class_idx in range(6)]
            self.class_biases.append('none')
            self.class_biases = list(set([bias for sublist in self.class_biases for bias in sublist]))
            
            for class_idx in self.original_class_bias_stats:
                for bias_attr in self.original_class_bias_stats[class_idx]:
                    for sample_path in self.original_class_bias_stats[class_idx][bias_attr]:
                        self.img2attr[sample_path] = bias_attr

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            
            if not origin_only:
                if args.half_generated:
                    # return origin + generated(matched 1:1) -> generated ratio ~= biased ratio
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]: continue
                        tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                        self.generated_data.append(tmp_gene_data)
                    self.data += self.generated_data

                elif args.only_no_tags:
                    # return only non-bias-tag samples
                    self.no_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    self.data = self.no_tags

                elif args.only_tags:
                    # retrun only bias-tag samples(origin)
                    self.yes_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if self.origin2gene[image_key]:
                            self.yes_tags.append(data)
                    self.data = self.yes_tags

                elif args.only_no_tags_balanced:
                    # return non-bias-tags sample + generated sample(# non-bias-tags * # classes)
                    self.no_tags = []
                    self.generated_data = []
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.tmp_generated_data = self.generated_align + self.generated_conflict
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    for _ in range(len(self.no_tags) * 6): # num_class 6
                        choosen_gene_data = random.choice(self.tmp_generated_data)
                        self.generated_data.append(choosen_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif args.no_tags_gene:
                    # return non-bias-tags sample + generated sample(# bias-tags)
                    self.no_tags = []
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                        else:
                            tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                            self.generated_data.append(tmp_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif not args.half_generated and args.include_generated:
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.generated_data = self.generated_align + self.generated_conflict
                    self.data += self.generated_data
                
        elif split=='valid':
            self.data = glob(os.path.join(root, '../valid', '*', '*'))
        elif split=='test':
            self.data = glob(os.path.join(root, '../test', '*', '*'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image_path = self.data[index]

        if 'conflict' in image_path:
            attr[1] = (attr[0] + 1) % 6
        elif 'align' in image_path:
            attr[1] = attr[0]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]
       
class DogCatDataset(Dataset):
    def __init__(self,
                 args,
                 root, 
                 split,
                 origin_only,
                 transform=None, 
                 image_path_list=None,
                 preproc_root=None,
                 ):
        super(DogCatDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.preproc_root = preproc_root
        self.mixup = args.mixup
        
        if split == 'train' and args.ours and not origin_only:
            self.img2attr = {}
            original_class_bias_stats_path = os.path.join(preproc_root, 'original_class_bias_stats.json')
            self.original_class_bias_stats = load_json(original_class_bias_stats_path)
            
            generated_class_bias_stats_path = os.path.join(preproc_root, 'generated_class_bias_stats.json')
            self.generated_class_bias_stats = load_json(generated_class_bias_stats_path) # generated_class_bias_stats[class][bias_attr]
            
            itg_tag_stats_path = os.path.join(preproc_root, 'tag_stats.json')
            self.itg_tag_stats = load_json(itg_tag_stats_path)
            
            origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
            self.origin2gene = load_json(origin2gene_path)
            
            self.class_biases = [self.itg_tag_stats[str(class_idx)]['bias_tags'] for class_idx in range(2)]
            self.class_biases.append('none')
            self.class_biases = list(set([bias for sublist in self.class_biases for bias in sublist]))
            
            for class_idx in self.original_class_bias_stats:
                for bias_attr in self.original_class_bias_stats[class_idx]:
                    for sample_path in self.original_class_bias_stats[class_idx][bias_attr]:
                        self.img2attr[sample_path] = bias_attr

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            if not origin_only:
                if args.half_generated:
                    # return origin + generated(matched 1:1) -> generated ratio ~= biased ratio
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]: continue
                        tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                        self.generated_data.append(tmp_gene_data)
                    self.data += self.generated_data

                elif args.only_no_tags:
                    # return only non-bias-tag samples
                    self.no_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    self.data = self.no_tags

                elif args.only_tags:
                    # retrun only bias-tag samples(origin)
                    self.yes_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if self.origin2gene[image_key]:
                            self.yes_tags.append(data)
                    self.data = self.yes_tags

                elif args.only_no_tags_balanced:
                    # return non-bias-tags sample + generated sample(# non-bias-tags * # classes)
                    self.no_tags = []
                    self.generated_data = []
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.tmp_generated_data = self.generated_align + self.generated_conflict
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    for _ in range(len(self.no_tags) * 2): # num_class 2
                        choosen_gene_data = random.choice(self.tmp_generated_data)
                        self.generated_data.append(choosen_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif args.no_tags_gene:
                    # return non-bias-tags sample + generated sample(# bias-tags)
                    self.no_tags = []
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                        else:
                            tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                            self.generated_data.append(tmp_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif not args.half_generated and args.include_generated:
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.generated_data = self.generated_align + self.generated_conflict
                    self.data += self.generated_data
                
        elif split == "valid":
            self.data = glob(os.path.join(root, split, '*'))
        elif split == "test":
            self.data = glob(os.path.join(root, "../test", '*', '*'))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]

class CIFAR10CDataset(Dataset):
    def __init__(self,
                 args,
                 root, 
                 split,
                 origin_only,
                 transform=None, 
                 image_path_list=None,
                 preproc_root=None,
                 ):
        super(CIFAR10CDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        self.preproc_root = preproc_root
        self.mixup = args.mixup
        
        if split == 'train' and args.ours and not origin_only:
            self.img2attr = {}
            original_class_bias_stats_path = os.path.join(preproc_root, 'original_class_bias_stats.json')
            self.original_class_bias_stats = load_json(original_class_bias_stats_path)
            
            generated_class_bias_stats_path = os.path.join(preproc_root, 'generated_class_bias_stats.json')
            self.generated_class_bias_stats = load_json(generated_class_bias_stats_path) # generated_class_bias_stats[class][bias_attr]
            
            itg_tag_stats_path = os.path.join(preproc_root, 'tag_stats.json')
            self.itg_tag_stats = load_json(itg_tag_stats_path)
            
            origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
            self.origin2gene = load_json(origin2gene_path)
            
            self.class_biases = [self.itg_tag_stats[str(class_idx)]['bias_tags'] for class_idx in range(10)]
            self.class_biases.append('none')
            self.class_biases = list(set([bias for sublist in self.class_biases for bias in sublist]))
            
            for class_idx in self.original_class_bias_stats:
                for bias_attr in self.original_class_bias_stats[class_idx]:
                    for sample_path in self.original_class_bias_stats[class_idx][bias_attr]:
                        self.img2attr[sample_path] = bias_attr

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            
            if not origin_only:
                if args.half_generated:
                    # return origin + generated(matched 1:1) -> generated ratio ~= biased ratio
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]: continue
                        tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                        self.generated_data.append(tmp_gene_data)
                    self.data += self.generated_data

                elif args.only_no_tags:
                    # return only non-bias-tag samples
                    self.no_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    self.data = self.no_tags

                elif args.only_tags:
                    # retrun only bias-tag samples(origin)
                    self.yes_tags = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if self.origin2gene[image_key]:
                            self.yes_tags.append(data)
                    self.data = self.yes_tags

                elif args.only_no_tags_balanced:
                    # return non-bias-tags sample + generated sample(# non-bias-tags * # classes)
                    self.no_tags = []
                    self.generated_data = []
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.tmp_generated_data = self.generated_align + self.generated_conflict
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                    for _ in range(len(self.no_tags) * 10): # num_class 10
                        choosen_gene_data = random.choice(self.tmp_generated_data)
                        self.generated_data.append(choosen_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif args.no_tags_gene:
                    # return non-bias-tags sample + generated sample(# bias-tags)
                    self.no_tags = []
                    self.generated_data = []
                    for data in self.data:
                        image_key = data.replace(root+'/', '')
                        if not self.origin2gene[image_key]:
                            self.no_tags.append(data)
                        else:
                            tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
                            self.generated_data.append(tmp_gene_data)
                    self.data = self.no_tags + self.generated_data

                elif not args.half_generated and args.include_generated:
                    self.generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                    self.generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                    self.generated_data = self.generated_align + self.generated_conflict
                    self.data += self.generated_data
                
        elif split=='valid':
            self.data = glob(os.path.join(root, split, '*', '*'))
        elif split=='test':
            self.data = glob(os.path.join(root, '../test', '*', '*'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]


transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    "dogs_and_cats": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
        },
    "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),])
        },
    }


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
    "dogs_and_cats": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        },
    "cifar10c": {
        "train": T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        },
    }

def get_dataset(args,
                dataset, 
                data_dir, 
                dataset_split, 
                transform_split, 
                percent,
                origin_only,
                use_preprocess=None, 
                image_path_list=None,
                preproc_dir='none'):

    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]

    dataset_split = "valid" if (dataset_split == "eval") else dataset_split

    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        preproc_root = preproc_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(args=args,
                                root=root, 
                                split=dataset_split,
                                origin_only=origin_only,
                                transform=transform,
                                image_path_list=image_path_list,
                                preproc_root=preproc_root)
        
    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        preproc_root = preproc_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(args=args,
                               root=root, 
                               split=dataset_split, 
                               origin_only=origin_only,
                               transform=transform, 
                               image_path_list=image_path_list,
                               preproc_root=preproc_root)
        
    elif dataset == "bar":
        root = data_dir + f"/bar/{percent}"
        preproc_root = preproc_dir + f"/bar/{percent}"
        dataset = BARDataset(args=args,
                             root=root, 
                             split=dataset_split, 
                             origin_only=origin_only,
                             transform=transform, 
                             image_path_list=image_path_list,
                             preproc_root=preproc_root)
        
    elif dataset == "dogs_and_cats":
        root = data_dir + f"/dogs_and_cats/{percent}"
        preproc_root = preproc_dir + f"/dogs_and_cats/{percent}"
        dataset = DogCatDataset(args=args,
                                root=root, 
                                split=dataset_split, 
                                origin_only=origin_only,
                                transform=transform, 
                                image_path_list=image_path_list,
                                preproc_root=preproc_root)
        
    elif dataset == "cifar10c":
        root = data_dir + f"/cifar10c/{percent}"
        preproc_root = preproc_dir + f"/cifar10c/{percent}"
        dataset = CIFAR10CDataset(args=args,
                                  root=root, 
                                  split=dataset_split, 
                                  origin_only=origin_only,
                                  transform=transform, 
                                  image_path_list=image_path_list,
                                  preproc_root=preproc_root)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset