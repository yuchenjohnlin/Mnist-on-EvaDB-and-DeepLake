import opendatasets as od
import os
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
import deeplake 
import numpy as np

od.download("https://www.kaggle.com/datasets/scolianni/mnistasjpg/data")
clear_output()
print(os.listdir())


# Create dataset in deeplake format
deep_lake_path = "hub://organization_name/dataset_name"

ds = deeplake.empty('./mnist_jpg', overwrite=True) # Create the dataset locally

# Find the class_names and list of files that need to be uploaded
dataset_folder = './mnistasjpg'
cwd = os.getcwd()
print("Current working directory:", cwd)

# Find the subfolders, but filter additional files like DS_Store that are added on Mac machines.
set_names = [item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))]
print(f"set names : '{set_names}'")

trainset_folder = os.path.join(dataset_folder,'trainingSet/trainingSet')
class_names = [item for item in os.listdir(trainset_folder) if os.path.isdir(os.path.join(trainset_folder, item))]
print(class_names)

files_list = []

# Get only files in trainingSet
for dirpath, dirnames, filenames in os.walk(dataset_folder):
  if dirpath.startswith(os.path.join(dataset_folder, 'trainingSet')) and os.path.basename(dirpath) in ['3','1','2','4','5','6','7','8','9']: #,'testSet', 'testSample']:
    for filename in filenames:
      files_list.append(os.path.join(dirpath, filename))
#print(files_list)
print(len(files_list))

print("Load dataset to DeepLake ... ")
with ds:
    # Create the tensors with names of your choice.
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg', exist_ok=True)
    ds.create_tensor('labels', htype = 'class_label', class_names = class_names, exist_ok=True)

load_data2_start = time.time()
with ds:
    # Iterate through the files and append to Deep Lake dataset
    for file in files_list:
        label_text = os.path.basename(os.path.dirname(file))
        label_num = class_names.index(label_text)

        #Append data to the tensors
        ds.append({'images': deeplake.read(file), 'labels': np.uint32(label_num)})

  # Warning !!! :  Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions.
load_data2_end = time.time()

data_load_time = load_data2_end - load_data2_start
print(data_load_time)

# --------------------------------- Model --------------------------------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

tform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 100

# Since torchvision transforms expect PIL images, we use the 'pil' decode_method for the 'images' tensor. This is much faster than running ToPILImage inside the transform
#hope this works
mnist_loader = ds.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
model = mnist(pretrained=True)
#test_model(model, test_loader, device)

#print("Type of model is : ", type(model))
runall_start = time.time()
test_model(model, mnist_loader, device)
runall_end = time.time()

runall_time = runall_end - runall_start
print(f"Run all data time : {runall_time} seconds")

pred_num = 6
query_start = time.time()

get_label6 = get_pred_model(model, mnist_loader, device, pred_num)
# print(type(get_label6[0]))
# for i in range(0,len(get_label6)-1) :
#   plt.imshow(get_label6[i].permute(1, 2, 0).cpu().numpy())  # Assuming channels-last format
#   plt.show()

query_end = time.time()
query_time = query_end - query_start
print(f"Query data time : {query_time} seconds")




class MLP(nn.Module):
  def __init__(self, input_dims, n_hiddens, n_class):
    super(MLP, self).__init__()
    assert isinstance(input_dims, int), 'Please provide int for input_dims'
    self.input_dims = input_dims
    current_dims = input_dims
    layers = OrderedDict()

    if isinstance(n_hiddens, int):
        n_hiddens = [n_hiddens]
    else:
        n_hiddens = list(n_hiddens)
    for i, n_hidden in enumerate(n_hiddens):
        layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
        layers['relu{}'.format(i+1)] = nn.ReLU()
        layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
        current_dims = n_hidden
    layers['out'] = nn.Linear(current_dims, n_class)

    self.model= nn.Sequential(layers)
    print(self.model)

  def forward(self, input):
    input = input.view(input.size(0), -1)
    assert input.size(1) == self.input_dims
    return self.model.forward(input)

def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
  model = MLP(input_dims, n_hiddens, n_class)
  if pretrained is not None:
      m = model_zoo.load_url(model_urls['mnist'],
                             map_location=torch.device(device)) # this is the line where EvaDB was different from the github code
      state_dict = m.state_dict() if isinstance(m, nn.Module) else m
      assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
      model.load_state_dict(state_dict)
  return model


def test_model(model, data_loader, device):
  model.eval()

  start_time = time.time()
  total = 0
  correct = 0
  with torch.no_grad(): # this deactivates the gradient computation which is not needed for model evaluation and inference, turning this off can make things run faster , this is mostky for backpropagation while training
    for i, data in enumerate(data_loader): # here we use the data in the data_loader , but because it is not transformed so I don;t know if it will work
      inputs = data['images']
      labels = torch.squeeze(data['labels'])

      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs.float())

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      #print("Predicted : ", predicted)
      correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total

      print('Finished Testing')
      print('Testing accuracy: %.1f %%' %(accuracy))

def get_pred_model(model, data_loader, device, pred_num):
  model.eval()

  result = []
  with torch.no_grad(): # this deactivates the gradient computation which is not needed for model evaluation and inference, turning this off can make things run faster , this is mostky for backpropagation while training
    for i, data in enumerate(data_loader): # here we use the data in the data_loader , but because it is not transformed so I don;t know if it will work
      inputs = data['images']
      labels = torch.squeeze(data['labels'])

      inputs = inputs.to(device)
      labels = labels.to(device)

      # chatgpt says that this only does forward pass
      outputs = model(inputs.float())
      _, predicted = torch.max(outputs.data, 1)
      for j in range(inputs.size(0)):
        if predicted[j].item()==pred_num:
          result.append(inputs[j])

  return result
