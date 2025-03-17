from torch.utils.data import Dataset
import os
from PIL import Image

class DR_HR_Dataset(Dataset):
    def __init__(self, dataframe, images_folder, transform=None):
        self.dataframe = dataframe
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.images_folder, f"{img_name}.jpeg")
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['binary_label'] if 'binary_label' in self.dataframe.columns else self.dataframe.iloc[idx]['severity_label']

        if self.transform:
            image = self.transform(image)
        return image, label