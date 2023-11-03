import evadb 
import os 
import opendatasets as od
import time

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
print("Current working directory:", cwd)


trainset_folder = os.path.join(dataset_folder,'trainingSet/trainingSet')
class_names = [item for item in os.listdir(trainset_folder) if os.path.isdir(os.path.join(trainset_folder, item))]
print(f"class names : '{class_names}' ")


# Get only files in trainingSet
files_list = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
  if dirpath.startswith(os.path.join(dataset_folder, 'trainingSet')) and os.path.basename(dirpath) in ['3','1','2','4','5','6','7','8','9']: #,'testSet', 'testSample']:
    for filename in filenames:
      files_list.append(os.path.join(dirpath, filename))
#print(files_list)
print(f"# of jpg files : '{len(files_list)}'")

# --- run model on evadb ---
print("Loading data into EvaDB ... might take 30 minutes ...")
cursor = evadb.connect().cursor()

cursor.query("DROP TABLE IF EXISTS MNIST_image").df()

load_data_start = time.time()

# if the file is already in the table it will show an error
for file in files_list:
  cursor.query(f"LOAD IMAGE '{file}' INTO MNIST_image").df()

load_data_end = time.time()
load_data_time = load_data_end - load_data_start
print(f"Load data_time : {load_data_time} seconds")

# --- run query ---
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

