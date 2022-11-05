from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Flatten:
    
    def __call__(self, pic):
        
        return torch.flatten(pic)

class Onehot_transform:
    def __init__(self, num_class) -> None:
        self.num_class = num_class
    
    def __call__(self, target):
        return F.one_hot(torch.Tensor([target])[0].long(), num_classes=self.num_class)


class ImageTransform:
    def __init__(self, crop_size, padding_size, rand_noise, train, flatten, dataset, normalize, from_tensor):
        self.rand_noise = rand_noise
        
        transform_list = []
        if from_tensor:
            transform_list.append(transforms.ToPILImage())
        if train:
            transform_list += [transforms.RandomCrop(crop_size, padding=padding_size),
                                transforms.RandomHorizontalFlip()]
        
        transform_list += [transforms.ToTensor()]
        if normalize:
            if dataset=='cifar10':
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            elif dataset=='cifar100':
                mean = [0.507, 0.487, 0.441]
                std = [0.267, 0.256, 0.276]
            transform_list += [transforms.Normalize(mean=mean, std=std)]

        if flatten:
            transform_list.append(Flatten())
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, pic):
        pic = self.transform(pic)
        if self.rand_noise:
            pic += (torch.rand_like(pic)/255.0)
        
        return pic

class ImageClientDataset(Dataset):
    def __init__(self, x, y, weight, transform) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.weight = weight
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.transform(self.x[index])
        if type(self.weight)!=type(None):
            return x, self.y[index], self.weight[index]
        else:
            return x, self.y[index]