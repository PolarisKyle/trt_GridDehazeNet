from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])



class TestData(Dataset):
    def __init__(self,test_img_path):
        super(TestData,self).__init__()
        
        inp_files = sorted(os.listdir(test_img_path))
        
        self.inp_filenames = [os.path.join(test_img_path, x) for x in inp_files if is_image_file(x)]
        
        self.inp_size = len(self.inp_filenames)
    
    def __len__(self):
        return self.inp_size
        

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)
        
        transform_raw = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        test_image = transform_raw(inp)
        return test_image,filename


  
