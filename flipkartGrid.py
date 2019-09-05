
# coding: utf-8

# In[1]:


import os
import imghdr
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import keras
import glob
import gc
import pickle 
from PIL import Image
from matplotlib.pyplot import imshow
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.initializers import glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input,Flatten,Add,Lambda,AveragePooling2D,Conv2D,ZeroPadding2D,Dense,Dropout,Flatten,BatchNormalization,Activation,MaxPooling2D,Reshape
from keras.models import  Model,Sequential
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    cc = gc.collect()
    return image, image_data


# In[1]:



def load_data():
    path = "../input/train1/traindata/trainData/*.png"
    files = glob.glob(path)
    files = sorted(files)
    X_train={}
    for i,file in enumerate(files): 
        image, image_data = preprocess_image(file, model_image_size = (512,512))
        X_train[i] = image_data[0,:,:]
#         print(image_data[0,:,:].shape)
        if i%100==0:
            print(i)
   
    return X_train


# In[2]:


y=pd.read_csv('../input/training/training.csv',delimiter=',')
Y_train= y.sort_values('image_name',ascending=True)
Y_train = (Y_train[['x1','y1','x2','y2']])


# In[6]:


# def load_train_data(path,Y):
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0,
#         zoom_range=0,
#         horizontal_flip=False,
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1)
    
#     generator = datagen.flow_from_directory(
#         directory=path,
#         target_size=(512,512),
#         color_mode="rgb",
#         batch_size=32,
#         class_mode=None,
#         shuffle=True,
#         seed=42,
#         )
#     return generator


# In[7]:


X_train = load_data()


# In[ ]:


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


# In[ ]:


anchors = read_anchors("../input/anchors-new/anchor.txt")
print(anchors)


# In[ ]:



# print(Y_train)


# In[ ]:


def filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 1)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    ### END CODE HERE ###
#     print(boxes.shape)
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask =(box_class_scores >= threshold )
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores,filtering_mask,name='boolean_mask')
    boxes = tf.boolean_mask(boxes,filtering_mask,name='boolean_mask')
    classes = tf.boolean_mask(box_classes,filtering_mask,name='boolean_mask')
    ### END CODE HERE ###
    
    return scores, boxes, classes


# In[ ]:


def boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    
    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


# In[ ]:


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


# In[ ]:


def non_max_suppression(scores, boxes, classes, max_boxes = 10,iou_threshold = 0.5):
 
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold)
    ### END CODE HERE ###
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.gather(scores,nms_indices)
    boxes = tf.gather(boxes,nms_indices)
    classes = tf.gather(classes,nms_indices)
    ### END CODE HERE ###
    
    return scores, boxes, classes


# In[ ]:



def eval( box_confidence, box_xy, box_wh, box_class_probs , image_shape = (640.,480.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box Xcoordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE ### 
    
    # Retrieve outputs of the YOLO model (≈1 line)
   

#  Convert boxes to be ready for filtering functions 
    boxes = boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6)
    
#     # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

#     # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes , iou_threshold )
    
    ### END CODE HERE ###
    
    return scores,boxes


# In[ ]:


def convert(X, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    
    
    num_anchors = len(anchors)
#     print(num_anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
#     print(anchors_tensor.shape)
    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(X)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(X))
    
    X = K.reshape(X, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
  
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(X))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_confidence = K.sigmoid(X[..., 4:5])
    box_xy = K.sigmoid(X[..., :2])
    box_wh = K.exp(X[..., 2:4])
    box_class_probs = K.softmax(X[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_confidence, box_xy, box_wh, box_class_probs


# In[ ]:


def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)


# In[ ]:


def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)


# In[ ]:


def my_model(input_shape = (512, 512, 3)):
    X_input = Input(input_shape)
    
    model=Sequential()
    
    model.add(keras.layers.ZeroPadding2D(padding=2,input_shape=input_shape))
    model.add(Conv2D(16,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv1', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv1'))
    model.add(Activation('relu'))
    model.add(keras.layers.ZeroPadding2D(padding=2))
    model.add(MaxPooling2D((3, 3), strides=2))
    
    
    model.add(keras.layers.ZeroPadding2D(padding=2))
    model.add(Conv2D(64, kernel_size=5, strides = (1, 1),padding="valid",name = 'conv2', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    
    model.add(keras.layers.ZeroPadding2D(padding=2))
    model.add(Conv2D(128, kernel_size=5, strides = (1, 1),padding="valid",name = 'conv3', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv3'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(256,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv4', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv4'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), strides=(2, 2)))
    
    model.add(keras.layers.ZeroPadding2D(padding=2))
    model.add(Conv2D(512,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv5', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv5'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(1024,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv6', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv6'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), strides=(1,1)))
   
    model.add(Conv2D(512,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv7', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv7'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), strides=(2, 2)))
    
    model.add(Conv2D(128,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv8', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv8'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(Conv2D(256,kernel_size=5, strides = (1, 1),padding="valid",name = 'conv9', kernel_initializer = keras.initializers.glorot_uniform(seed=0)))
    model.add(BatchNormalization(axis = 3, name = 'bn_conv9'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((5, 5), strides=(2, 2)))
    
     
    Z = model.output
#     Y=Z
#     print(Y.shape)
    X = convert(Z,anchors,1) 
#     print(X.shape)
    box_confidence, box_xy, box_wh, box_class_probs = X
    score,boxes = eval(box_confidence, box_xy, box_wh, box_class_probs)
    
#     print(boxes)
    model.add(Lambda(lambda x: boxes))
    
    return model


# In[ ]:




ckpt = ModelCheckpoint('weights.{epoch:02d}.hdf5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=5, min_lr=0.001)


# In[ ]:


model=my_model()
model.compile(optimizer='adam', loss=iou_coef_loss, metrics=['accuracy'])
model.summary()
model.fit(X1,Y_train,epochs=10,batch_size=100,callbacks=[ckpt,reduce_lr])


# In[ ]:


# model.fit(X1,Y_train,epochs=2,batch_size=10,validation_split=0.1,callbacks=[reduce_lr,ckpt])


# In[ ]:


def load_testdata():
    path = "../input/flipkartgrid/testdata/testdata/*.png"
    files = glob.glob(path)
    files = sorted(files)
    X_train={}
    for i,file in enumerate(files): 
        image, image_data = preprocess_image(file, model_image_size = (512,512))
        X_train[i] = image_data[0,:,:]
#         print(image_data[0,:,:].shape)
        if i%100==0:
            print(i)
   
    return X_train


# In[ ]:


X_test = load_testdata()


# In[ ]:


del X
X2 = list(X_test.values())
X3 = np.asarray(X2,dtype=np.float32)
del X1
del X_train
import gc
cc = gc.collect()
del X2
del X_test


# In[ ]:


print(X3.shape)


# In[ ]:


Y = model.predict(X3,batch_size=10)


# In[ ]:


np.savetxt("foo.csv", Y, delimiter=",")
y=pd.read_csv('foo.csv',delimiter=',')

