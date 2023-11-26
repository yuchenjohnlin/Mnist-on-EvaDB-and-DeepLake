import evadb 
import os 
import opendatasets as od
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
import random

def wait_user():
  print("Press any key to continue...")
  input()
  print("Continuing execution...")
  return

timing_file = 'Timing_EvaDB_Query_7500.txt'
with open(timing_file, 'w') as output_file:
  output_file.write("Using testSet from mnistjpg\n")


# Create a cProfile object
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

# Run function to Profile
####################################################################################

#dataset_folder = '/mnist_evadb'

#os.makedirs(dataset_folder)
#print(f"Folder '{dataset_folder}' create successfully")
#os.chdir(dataset_folder)
od.download("https://www.kaggle.com/datasets/scolianni/mnistasjpg/data")
#print("Mnist jpg dataset downloaded successfully to '{dataset_folder}'")
print("Mnist jpg dataset downloaded successfully '")
print(os.listdir())

dataset_folder = './mnistasjpg'
cwd = os.getcwd()
print(f"Dataset in folder : {dataset_folder}")
print("Current working directory:", cwd)

# Get files in trainingSample/trainingSample
files_list = []

for dirpath, dirnames, filenames in os.walk(os.path.join(dataset_folder,"testSet/testSet") ):
    for filename in filenames:
      files_list.append(os.path.join(dirpath, filename))
# print(files_list)
print(f"# of jpg files : '{len(files_list)}'")

num_images = 15000
files_list = random.sample(files_list, num_images)

# --- run model on evadb ---
print("Loading data into EvaDB ... might take 30 minutes ...")
cursor = evadb.connect().cursor()

cursor.query("DROP TABLE IF EXISTS MNIST_image").df()

load_data_start = time.time()

# if the file is already in the table it will show an error
for file in files_list:
  input = cursor.query(f"LOAD IMAGE '{file}' INTO MNIST_image").df()
# this query doesn't return anything so the input doesn;t contaain anything
# maybe have to use query  SELECT * FROM MNIST_image 

load_data_end = time.time()
load_data_time = load_data_end - load_data_start
# print(f"Load data_time : {load_data_time} seconds")

with open(timing_file, 'a') as output_file:
    output_file.write(f"Load {len(files_list)} images")
    output_file.write("Load Image : " + str(load_data_time) + " s\n")
    
print(input.info())
print(type(input))
print(input[0])


####################################################################################
# End function to profile

# Stop profiling
profiler.disable()

# Print the profiling results
# profiler.print_stats()
profiler.dump_stats('EvaDB_dataload.prof')

# Redirect the standard output to a file
with open("EvaDB_dataload.txt", "w") as f:
    ps = pstats.Stats("EvaDB_dataload.prof", stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()



# Run function to Profile
####################################################################################

# Start profiling
profiler.enable()

############################################
#            --- run query ---             #
############################################
print("Running query ... might take 30 minutes ...")

query_time_start = time.time()

query = cursor.query("""
    SELECT data, MnistImageClassifier(data).label
    FROM MNIST_image
    WHERE MnistImageClassifier(data).label = '6'
""")

response = query.df()

query_time_end = time.time()
query_time = query_time_end - query_time_start
print(f"Query data time : {query_time} seconds")
with open(timing_file, 'a') as output_file:
  output_file.write("SELECT data, MnistImageClassifier(data).label  FROM MNIST_image  WHERE MnistImageClassifier(data).label = '6'")
  output_file.write(f"Query {response['mnist_image.data'].shape[0]} images")
  output_file.write("Query : " + str(query_time) + " s\n")
    

response.describe()
print(response.info())
print(type(response))
print(response['mnist_image.data'].shape)
print(response['mnist_image.data'][0].shape)

# prints the filtered image 
'''
import matplotlib.pyplot as plt
import numpy as np

show_image = input("Want to show image ? y/n")
if show_image == 'y':
  
  for i in range(len(response['mnist_image.data'])):
    plt.title(f"label: {response['mnistimageclassifier.label'][i]}")
    plt.imshow(response['mnist_image.data'][i])  
    plt.show()
'''

  # this prints out some of the samples in the filtered data
  # create figure (fig), and array of axes (ax)
  # fig, ax = plt.subplots(nrows=1, ncols=5, figsize=[6,8])
  # for axi in ax.flat:
  #     idx = np.random.randint(len(response))
  #     img = response['mnist_image.data'].iloc[idx]
  #     label = response['mnistimageclassifier.label'].iloc[idx]
  #     axi.imshow(img)

  #     axi.set_title(f'label: {label}')

  # plt.show()

####################################################################################
# End function to profile

# Stop profiling
profiler.disable()

# Print the profiling results
# profiler.print_stats()
profiler.dump_stats('EvaDB_query.prof')

# Redirect the standard output to a file
with open("EvaDB_query.txt", "w") as f:
    ps = pstats.Stats("EvaDB_query.prof", stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()