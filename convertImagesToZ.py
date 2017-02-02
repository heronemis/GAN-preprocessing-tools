import PIL
from PIL import Image
from PIL import ImageOps

import numpy as np
import glob

src = 'faces/'
outputFolder = '/processedImages/'
basewidth = 10
counter = 0

batch_size=64
batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

data = glob.glob(src + "*.jpeg")
np.random.shuffle(data)



for filename in data:
    img = Image.open(filename)
    # img = img.convert('L')  # convert image to greyscale
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS) # resizes the image to 10x10
    img = img.convert('L') # convert image to black and white
    name = 'processedImages/'+ str(counter) + '.jpeg'
    # img = ImageOps.invert(img)
    img.save(name)

    pix = np.array(img, np.float32)
    pix = (pix - 128) / 128 #Scales the pixels to be between -1 and 1
    pix = pix.flatten() #flattens the image to a single array of lenght of 100


    batch_z[counter] = pix # Adds the image to the z-array
    counter +=1
    print("Counter: " + str(counter))
    if(counter >= batch_size):
        break






