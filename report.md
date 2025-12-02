# COMP 4630 Assignment 2 Report

Team Name: Deshawn & Friends

Team Members: Clency Tabe, Tobias Wondwoseen, Glenn Yeap

## Abstract
For both our models, we downsampled the train set to size of 112 x 112 and data augmentation. 

- Starting with our best performance model, we began with pixel rescaling (0–1 normalization, which improved training stability) followed by three stages of residual blocks with increasing filter sizes (64 → 128 → 256). Each residual block contains two Conv2D layers with BatchNormalization, followed by a skip connection which adds the original input to the processed features, enabling stable training for the deep networks. We also used Max-pooling between blocks to reduce spatial dimensions while retaining critical features. The network ends with a GlobalAveragePooling2D layer rather than a Flatten for better generalization and reduced parameters, lowering overfitting, then Dropout (0.5) for regularization, and lastly a softmax classifier for 53-card categorization.

- For our size-optimized model, we used SeparableConv2D layers with fewer filters in each block (32 → 64 → 128) and fewer blocks (2 → 2 → 2). We also used a smaller kernel size (3x3) and fewer filters in the first layer (32). Additionally, we applied a smaller dropout rate (0.4) to reduce overfitting. This model has significantly fewer parameters (68K vs 3.49M) and is faster to train, while still maintaining good accuracy (90.57% vs 96.60%).

## Model summaries
#### Best performance model

The best performance model focuses on maximizing accuracy.

**Overview:**

- **Architecture**: A deep residual network with standard Conv2D layers for feature extraction
- **Structure**: Input (112×112×3) -> Rescaling (255 -> [0,1]) → Initial Conv (64 filters) with batch normalization → three stages of convolutional layers (with increasing filter sizes (64 → 128 → 256)) and each convolutional layer has 2 residual blocks each with increasing filter sizes (64 → 128 → 256) → Global Average Pooling → Dense output layer (53 classes)
- **Depth**: Three main blocks with two residual blocks each
- **Regularization**: Batch normalization, dropout (0.5)
- **Key features**: Residual blocks, batch normalization, global average pooling, dropout (0.5)
- **Parameters**: ~3.49M

```
Total params: 3,491,765
Trainable params: 3,487,285
Non-trainable params: 4,480

Test accuracy: 0.9660
```

#### Size efficient model

The size efficient model focuses on minimizing parameters while maintaining decent accuracy.

**Overview:**

- **Architecture**: Similar structure to the best performance model but uses separable convolutions
- **Structure**: Input (112×112×3) → Initial SeparableConv (32 filters) → Efficient residual blocks with 32, 64, and 128 filters → Global Average Pooling → Dense output layer (53 classes)
- **Depth**: Three main blocks with one efficient residual block each
- **Regularization**: Batch normalization, reduced dropout (0.4)
- **Parameters**: ~68K (significantly reduced from the best performance model)

Size-optimized model:
```
Total params: 68,464
Trainable params: 67,120
Non-trainable params: 1,344

Test accuracy: 0.9057
```
## Experiment Log
## Dataset Preparation
- Started with 224×224 images but experimented with downsampling to reduce computational complexity
- Tested 112×112 images and found this to be a good balance of detail preservation and efficiency
- Further tested 56×56 images but found they lost too much detail: "At 56×56, you start to lose the difference between hearts and diamonds, and clubs and spades"

## Data Augmentation Experiments
- Experimented with multiple data augmentation techniques:
  - Horizontal flipping
  - Saturation adjustment (custom function with range 0.5-1.5)
  - Contrast variation (0.6)
  - Zoom variation (0.15)
  - Initially tried but later removed: brightness variation, rotation, and translation
  
## Model Architecture 
1. #### **Best Performance Model**:
   - Used standard residual blocks with Conv2D layers
   - Implemented deeper structure with more filters (up to 256)
   - Applied aggressive dropout (0.5) to prevent overfitting

2. #### **Size Optimized Model**:
   - Replaced Conv2D with SeparableConv2D for parameter efficiency
   - Reduced filter counts (maximum 128 vs 256)
   - Simplified the architecture by using fewer residual blocks
   - Applied less aggressive dropout (0.4)

### Training Strategy
- Implemented callback for learning rate scheduling with ReduceLROnPlateau to 1e-6
- Used early stopping with patience=5 and best weights restoration to prevent overfitting
- Both models were trained with categorical cross-entropy loss and Adam optimizer
- Best performance model: 50 epochs max, more if training didn't experience overfitting or early stopping
- Size optimized model: 100 epochs max (allowing more time for the smaller model to converge), early stopping if overfitting.

## Reflections
What challenges did you encounter? Were there any "ah-hah" moments or surprising results? Let me know!

We never really ran into any serious issues with the actual model design and implementation, but getting TensorFlow to work with CUDA was not fun. It required particular versions of Python and CUDA that are relatively old. We actually ended up implementing this project in both PyTorch and TensorFlow. By default, PyTorch doesn't have a framework like Keras, so we needed to write our own training loop. In the end, we stuck with TensorFlow because the code was a lot shorter and felt like it gave us less room for mistakes. If we were to start this project again, we would begin with PyTorch from the beginning since it's a lot more flexible and more widely used today. We just didn't want to take the risk of using it after only having experience with it for a short amount of time.

In terms of implementation, the only real issue we had was training time. Colab was not working well for us, so we decided to just run everything locally with our own GPUs (3060Ti, 4060, 2070 Super), which are pretty mid-range GPUs by today's standards. Glenn's had 16GB GDDR6 VRAM, but it still took most of our attempts 20+ minutes for training 25-35 epochs.
Initially, it took much longer until we realized that we could cut the resolution of the images and they would still retain the important details such that important features would remain identifiable/percivable. This improved training time, but it still meant that we had to be intentional about the changes and choices we made since it would take at least 10 minutes (if we stopped training halfway through) to see if our choices made an improvement or not. 

Our "ah-hah" moment was definitely implementing Separable Convolutions for our smaller model. Until this point, anything we did to try and reduce the parameter count greatly reduced the model's accuracy, and we were really surprised it worked since it's effectively almost 90 percent smaller than our main most performant model. Another one of our moments was adding zooming as part of our augmentations. For a while, we thought it would be better to recognize the number, but then it helped more with recognizing the color, symbols, and number of symbols.

## Appendices
### References
What resources did you use? It's okay if these are other notebooks training models for the same dataset, just cite your sources. If using generative AI, please include some of the prompts that you used.

- claud 3.7 for debugging errors,
- gemini 2.0 pro with grounding for ideas that are based on google searches
- stack overflow for bug fixes
- stack exchange for ideas and bug fixes
- youtube for how to setup tensor flow with cuda 
- kagglehub samples for our attempted pytorch version 

### Loading code
How do I load your models? E.g.

**IMPORTANT PLEASE USE DRIVE LINK FOR MAIN MODEL**
https://drive.google.com/file/d/1mHTn3Xb0fIyYIK4I7nOyEud0_dfoHtMp/view?usp=sharing
unfortuntly it was to large to commit to github (greater than 25mb)

```python
best = tf.keras.models.load_model("best_performance_model.keras")
small = tf.keras.models.load_model("size_optimized_model.keras")
```

**If you are using colab please replace the first cell with the following**

```python
import os
import shutil
import kagglehub
import tensorflow as tf
from google.colab import drive

# Mount your google drive to /content/drive
drive.mount("/content/drive")

# MyDrive is the root of your Google drive, so you can make this wherever you want
path_to_cards = "/content/drive/MyDrive/CardDataset"

if not os.path.exists(path_to_cards + "/cards.csv"):
    # Download and move it, for some reason the path parameter of dataset_download isn't working
    path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
    shutil.move(path, path_to_cards)
```

Please make sure that your model loads and runs properly in the Colab environment.
