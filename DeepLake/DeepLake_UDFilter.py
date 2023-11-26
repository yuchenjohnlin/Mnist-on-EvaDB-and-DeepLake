import opendatasets as od
import os
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
import deeplake 
import numpy as np
import torch
from torchvision import transforms, models
from mnist_model import mnist, test_model, get_pred_model, get_prediction
import time
import torchvision.transforms.functional as F
import cProfile
import pstats
import random

def wait_user():
  print("Press any key to continue...")
  input()
  print("Continuing execution...")
  return

timing_file = 'Timing_DeepLake_UDQuery_7500.txt'
with open(timing_file, 'w') as output_file:
    output_file.write("Using testSet from mnistjpg\n")


# Create a cProfile object
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

# Run function to Profile
####################################################################################


# Download mnist jpg data set 
od.download("https://www.kaggle.com/datasets/scolianni/mnistasjpg/data")
clear_output()
# print(os.listdir())

# Create dataset in deeplake format
print("Create DeepLake Dataset : ")

# set up authentication
# user_token = input("Please enter your deeplake token: ")

deeplake_path = './mnist_jpg'
ds = deeplake.empty(deeplake_path, overwrite=True) # Create the dataset in activeloop account

# Find the class_names and list of files that need to be uploaded in the local dataset
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

# use the trainingSet to test the accuracy of the model 
# use testSet and the other sets to test how well the database performs

load_data_start = time.time()

# Get files in testSet/testSet
files_list = []

for dirpath, dirnames, filenames in os.walk(os.path.join(dataset_folder,"testSet/testSet") ):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))
# print(files_list)
print(len(files_list))

num_images = 7500
files_list = random.sample(files_list, num_images)

print("Load dataset to DeepLake ... ")
# ds.delete_tensor("images",large_ok=True)
# ds.delete_tensor("labels",large_ok=True)
with ds:
    # Create the tensors with names of your choice.
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg', exist_ok=True)
    # ds.create_tensor('labels', htype = 'class_label', class_names = class_names, exist_ok=True)

with ds:
    # Iterate through the files and append to Deep Lake dataset
    for file in files_list:
        #Append data to the tensors
        ds.append({'images': deeplake.read(file)})

  # Warning !!! :  Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions.

load_data_end = time.time()

data_load_time = load_data_end - load_data_start
# print("Load data time : ",data_load_time)
ds.summary()
with open(timing_file, 'a') as output_file:
    output_file.write(f"Load {ds['images'].shape[0]} images")
    output_file.write("Load Image : " + str(data_load_time) + " s\n")


############################################################
#                          Model                           #
############################################################

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
mnist_loader = ds.pytorch(num_workers = 0, transform = {'images': tform}, batch_size = batch_size, decode_method = {'images': 'pil'})
model = mnist(pretrained=True)
#test_model(model, test_loader, device)

#print("Type of model is : ", type(model))

# can print out images using the last parameter 
# it will print out the image and tell the actual and predicted label 
# have to cloas the popped out image to continue to the next one so it is better to keep it close

# test_model(model, mnist_loader, device, True)
# test_model(model, mnist_loader, device, False)


############################################################
#              User defined filter to do query             #
############################################################
# add predicted labels using the model since we don't have the labels
# test model only tests the accuracy of the model we want to get the predictions

query_start = time.time()

import torchvision.transforms.functional as F
@deeplake.compute
def filter_labels(sample_in, labels_list,model,device):
    # get_pred_model()
    # print(type(sample_in.images[0]))
    # print(sample_in)
    # print(sample_in.images)
    # print(sample_in.images.data())
    # print(sample_in.images.data()['value'].shape)

    pil_image = F.to_pil_image(sample_in.images.data()['value'])
    im_tensor = tform(pil_image)
    # print(type(im_tensor))
    # print(im_tensor)

    # this can print the (28,28,1) image directly
    # plt.imshow(sample_in.images.data()['value'])  # Assuming channels-last format
    # plt.show()

    model.eval()
    inputs = im_tensor.to(device)
    outputs = model(inputs.float())

    _, predicted = torch.max(outputs.data, 1)
    # print("label : ",sample_in.labels.data()['text'])
    # print("Predict : ",predicted.item())
    # print(predicted.item() in labels_list)
    return predicted.item() in labels_list

label_list = [6]
model = mnist(pretrained=True)
ds_filtered = ds.filter(filter_labels(label_list,model,'cpu'), scheduler = 'threaded', num_workers = 0)

query_end = time.time()
query_time = query_end - query_start


print("Query time : ",query_time)
with open(timing_file, 'a') as output_file:
    output_file.write("Execute qeury : SELECT * WHERE labels = 6\n")
    output_file.write(f"Query {ds_filtered['images'].shape[0]} images")
    output_file.write("Query : " + str(query_time) + " s\n")

ds_filtered.summary()

####################################################################################
# End function to profile

# Stop profiling
profiler.disable()

# Print the profiling results
# profiler.print_stats()
profiler.dump_stats('DeepLake_mnist_UDQuery.prof')

# Get stats
# stats = pstats.Stats('DeepLake_mnist.prof')
# stats.sort_stats('cumulative')  # Sort by cumulative time
# stats.print_stats()

# Redirect the standard output to a file
with open("DeepLake_mnist_UDQuery_profile.txt", "w") as f:
    ps = pstats.Stats("DeepLake_mnist_UDQuery.prof", stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()







