import os

path =  os.getcwd()
filenames = os.listdir(path)
i = 0;

for filename in filenames:
    print filename
    os.rename(filename, filename.replace("Canarinho Figurinhas", "canarinho").lower())

