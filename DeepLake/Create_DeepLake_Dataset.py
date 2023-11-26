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

timing_file = 'Timing_DeepLake_Tquery_testSet.txt'
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

user_token = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMDg2NTcwMSwiZXhwIjoxNzMyNDg4MDgxfQ.eyJpZCI6Inl1Y2hlbmpvaG5saW4ifQ.WgiN1CSwZHgJjyezhu6k-RvGJm34al62Qps9fK6ETEWMkfbMebax_sZMOBj76puq6hm1lpeeEzk2T2YF7-x2B'
os.environ['ACTIVELOOP_TOKEN'] = user_token

# deeplake_path = 'hub://yuchenjohnlin/mnistjpg_testSample'
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
#            Create label using prediction                 #
############################################################
# add predicted labels using the model since we don't have the labels
# test model only tests the accuracy of the model we want to get the predictions

load_label_start = time.time()

# append data one by one 
print("Get prediction")
prediction = get_prediction(model,mnist_loader,device)
print("Load labels to dataset")

with ds:
    ds.create_tensor('labels', htype = 'class_label',  exist_ok=True)
    
    # After creating an empty tensor, the length of the dataset is 0
    # Therefore, we iterate over ds.max_view, which is the padded version of the dataset
    for i, sample in enumerate(ds.max_view):
        pil_image = F.to_pil_image(sample.images.data()['value'])
        im_tensor = tform(pil_image)
        model.eval()
        inputs = im_tensor.to(device)
        outputs = model(inputs.float())

        _, predicted = torch.max(outputs.data, 1)
        ds.labels.append(np.uint32(predicted.item()))

load_label_end = time.time()
label_load_time = load_label_end - load_label_start
print("Load label time : ",data_load_time)
ds.summary()
with open(timing_file, 'a') as output_file:
    output_file.write("Load Label : " + str(label_load_time) + " s\n")

filter_start = time.time()

filtered_ds = ds.query("""
    SELECT *
    WHERE labels = 6
""")

filter_end = time.time()
filter_time = filter_end - filter_start

with open(timing_file, 'a') as output_file:
    output_file.write("Execute qeury : SELECT * WHERE labels = 6\n")
    output_file.write(f"Query {filtered_ds['images'].shape[0]} images")

# print("Filter time : ",filter_time)
with open(timing_file, 'a') as output_file:
    output_file.write("Filter : " + str(filter_time) + " s\n")

query_time = filter_end - load_label_start
# print("Query time : ",query_time)
with open(timing_file, 'a') as output_file:
    output_file.write("Query : " + str(query_time) + " s\n")


filtered_ds.summary()

####################################################################################
# End function to profile

# Stop profiling
profiler.disable()

# Print the profiling results
# profiler.print_stats()
profiler.dump_stats('DeepLake_mnist_TQuery.prof')

# Get stats
# stats = pstats.Stats('DeepLake_mnist.prof')
# stats.sort_stats('cumulative')  # Sort by cumulative time
# stats.print_stats()

# Redirect the standard output to a file
with open("DeepLake_mnist_TQuery_profile.txt", "w") as f:
    ps = pstats.Stats("DeepLake_mnist_TQuery.prof", stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()







