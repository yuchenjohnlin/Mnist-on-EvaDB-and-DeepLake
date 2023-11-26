import deeplake 
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

# set up authentication
# user_token = input("Please enter your deeplake token: ")
# 
user_token = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMDg2NTcwMSwiZXhwIjoxNzMyNDg4MDgxfQ.eyJpZCI6Inl1Y2hlbmpvaG5saW4ifQ.WgiN1CSwZHgJjyezhu6k-RvGJm34al62Qps9fK6ETEWMkfbMebax_sZMOBj76puq6hm1lpeeEzk2T2YF7-x2B'
os.environ['ACTIVELOOP_TOKEN'] = user_token

deeplake_path = 'hub://yuchenjohnlin/mnistjpg_testSample'
ds = deeplake.load(deeplake_path)
ds.summary()

############################################################
#                       Using Tquery                       #
############################################################


# the FROM clause doesn't actually do anything, ok it does it has to be specified correctly
# becuase we are actually using the ds dataset which is already decided
filtered_ds = ds.query("""
    SELECT *
    WHERE labels = 6
""")
print(type(ds))
print(type(filtered_ds))

filtered_ds.summary()
print(type(ds['images']))

show_image = input("Want to show image ? y/n")
if show_image == 'y':
  for sample in filtered_ds:
      plt.imshow(sample.images.data()['value'])  # Assuming channels-last format
      plt.show()






