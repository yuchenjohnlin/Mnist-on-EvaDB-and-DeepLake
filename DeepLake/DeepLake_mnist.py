import opendatasets as od
import os
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
import deeplake 
import numpy as np
import torch
from torchvision import transforms, models
from mnist_model import mnist, test_model, get_pred_model
import time
# import requests


def wait_user():
  print("Press any key to continue...")
  input()
  print("Continuing execution...")
  return


# Download mnist jpg data set 
od.download("https://www.kaggle.com/datasets/scolianni/mnistasjpg/data")
clear_output()
print(os.listdir())

# Create dataset in deeplake format
print("Create DeepLake Dataset : ")

# set up authentication
# user_token = input("Please enter your deeplake token: ")

# user_token = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMDg2NTcwMSwiZXhwIjoxNzMyNDg4MDgxfQ.eyJpZCI6Inl1Y2hlbmpvaG5saW4ifQ.WgiN1CSwZHgJjyezhu6k-RvGJm34al62Qps9fK6ETEWMkfbMebax_sZMOBj76puq6hm1lpeeEzk2T2YF7-x2B'
# os.environ['ACTIVELOOP_TOKEN'] = user_token

# deeplake_path = 'hub://yuchenjohnlin/mnistjpg'
# ds = deeplake.empty(deeplake_path, overwrite=True) # Create the dataset in activeloop account

# the new one can upload the data set to my account
ds = deeplake.empty('./mnist_jpg', overwrite=True) # Create the dataset locally

# Find the class_names and list of files that need to be uploaded
dataset_folder = './mnistasjpg'
cwd = os.getcwd()
print("Current working directory:", cwd)

# Find the subfolders, but filter additional files like DS_Store that are added on Mac machines.
set_names = [item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))]

'''
trainSet       : image of each digit and has a total of 42000 images
testSet        : 28000 images 
trainingSample : 60 image samples for digit -> 600 uniform samples from trainSet
testSample     : 350 image samples from the testSet
'''
# should use os.walk here
for set in set_names:
  file_count = 0
  set_path_name = os.path.join(dataset_folder,set)
  print(f"Set name : '{set}'")
  for path in os.listdir(set_path_name):
    file_name = os.path.join(set_path_name, path)
    if os.path.isfile(file_name):
      file_count += 1
  print('File count:', file_count)

# use the trainingSet to test the accuracy of the model 
# use testSet and the other sets to test how well the database performs
trainset_folder = os.path.join(dataset_folder,'trainingSet/trainingSet')

# get the class name when getting the images instead of doing it seperately
# class_names = [item for item in os.listdir(trainset_folder) if os.path.isdir(os.path.join(trainset_folder, item))]
# print(class_names)

files_list = []
class_names = []

'''
# Get only files in trainingSet
for dirpath, dirnames, filenames in os.walk(dataset_folder):
  if dirpath.startswith(os.path.join(dataset_folder, 'trainingSet')) and os.path.basename(dirpath) in ['0','1','2','3','4','5','6','7','8','9']: #,'testSet', 'testSample']:
    print(dirpath)
    for filename in filenames:
      files_list.append(os.path.join(dirpath, filename))
print(files_list)
# print(files_list)
'''

print(len(files_list))
# This is the new version because it was wrong in the previous edition 
# but I think it was because I didn't include the 0 label 

# Get files in trainingSample/trainingSample
files_list = []
for num in range(0,10):
  class_names.append(str(num))
  for dirpath, dirnames, filenames in os.walk(os.path.join(dataset_folder,"trainingSample/trainingSample") ):
    if os.path.basename(dirpath) == str(num): #,'testSet', 'testSample']:
      print("Dir path : ", dirpath)
      for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))
# print(files_list)
print(len(files_list))
print(class_names)


# wait_user()

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
print("Load data time : ",data_load_time)
ds.summary()

# --------------------------------- Model --------------------------------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


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
# can print out images using the last parameter 
# it will print out the image and tell the actual and predicted label 
# have to cloas the popped out image to continue to the next one so it is better to keep it close

# test_model(model, mnist_loader, device, True)
test_model(model, mnist_loader, device, False)
runall_end = time.time()

runall_time = runall_end - runall_start
print(f"Run all data time : {runall_time} seconds")


'''
  Above runs through the whole data set 
  Below is doing filter on the data
  which is to try query data
'''



############################################################
#               Using User Defined Function                #
############################################################

pred_num = 6
query_start = time.time()

get_label6 = get_pred_model(model, mnist_loader, device, pred_num)
print(type(get_label6[0]))
print(get_label6[0].shape)

show_image = input("Want to show image ? y/n")
if show_image == 'y':
  for i in range(0,len(get_label6)-1) :
    print(get_label6[i].shape)
    plt.imshow(get_label6[i].permute(1, 2, 0).cpu().numpy())  # Assuming channels-last format
    plt.show()

query_end = time.time()
query_time = query_end - query_start
print(f"Query data time : {query_time} seconds")





