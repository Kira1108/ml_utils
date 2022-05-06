# Less Tedious Code

> It is quite boring to write ml code, especially preprocessing code.   
> This repo intends to make everything easier.   
> There is nothing you can learn by writing the logics yourself.

*Installation*
```bash
pip install git+https://github.com/Kira1108/ml_utils.git
```

*Example*
```python
from ml_utils.cv import prepare_cv_dataset
split_sizes = [0.8,0.2]
splits = ['training', 'testing']
source_path = "./PetImages"
target_path = "./cats-v-dogs"
prepare_cv_dataset(source_path, target_path, splits, split_sizes)
```

You will see the following message(classifition rolder build automatically)
```
********** Summary **********
Category: Cat, files: 12500
Category: Dog, files: 12500
******************************
Spliting Dataset....
working on category:  Cat
Shuffling dataset....
666.jpg is zero length, so ignoring.
Shuffling dataset Done
working on category:  Dog
Shuffling dataset....
11702.jpg is zero length, so ignoring.
Shuffling dataset Done
Splitting Dataset Down
Target directory tree created
Copying files....
Copying file for category: Cat
Split set: training
Split set: testing
Copying file for category: Dog
Split set: training
Split set: testing
Copying files Done.
Finish prepare ml image classification dataset
```

*And you can then continue your ml work*
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "./cats-v-dogs/testing/"

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))
```
