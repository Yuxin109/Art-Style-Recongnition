import os
import glob
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, img_path, transform=None, num_classes=None):
        super(BaseDataset, self).__init__()
        self.img_path = img_path
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, idx):
        img_path = self.img_path[idx][0]
        label = self.img_path[idx][1]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_path)

transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def dataload(image_path):

    category = ["Cubism", "Expressionism", "New_Realism", "Northern_Renaissance"]
    Dataset_path_label = []

    for idx, classes in enumerate(category):
        path = os.path.join(image_path, classes)

        if os.path.isdir(path):
            img_paths = glob.glob(os.path.join(path, "*.jpg"))
            Dataset_path_label += [[t, idx] for t in img_paths]

    targets = [item[1] for item in Dataset_path_label]
    train_paths, test_paths = train_test_split(
        Dataset_path_label,
        test_size=0.2,
        stratify=targets
    )

    print("train dataset count : {}".format(len(train_paths)))
    print("test dataset count : {}".format(len(test_paths)))

    train_dataset = BaseDataset(train_paths, transform=transform_train, num_classes=4)
    test_dataset = BaseDataset(test_paths, transform=transform_test, num_classes=4)

    return train_dataset, test_dataset