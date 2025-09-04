from PIL import Image
import os
import pandas as pd

# Chemin vers le dossier contenant les images
folder_path = "../IMG/"

# Lister tous les fichiers du dossier
files = os.listdir(folder_path)
#print(files)

df_columns = {
        'image': 'string',
        'format': 'string',
        'height': 'int64',
        'width': 'int64',
        'mode': 'string'
}

data_list = []

for file in files:
    img = Image.open(os.path.join(folder_path, file))
    #print(file)

    new_data = {
        'image': file,
        'format': img.format,
        'height': img.height,
        'width': img.width,
        'mode': img.mode
    }
    data_list.append(new_data)
    img.close()

df = pd.DataFrame(data_list).astype(df_columns)

#print(df.tail(5))
#print(df.shape)
#print(df.dtypes)
#print(df.info())
#print(df['height']!=600)
#print(df['width']!=400)
print(df.loc[df['height']!=600,:])
print(df.loc[df['width']!=400,:])
#im.show()