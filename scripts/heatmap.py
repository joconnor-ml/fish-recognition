import glob
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids
import matplotlib.pyplot as plt
    
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)
model.compile(optimizer=sgd, loss='mse')
s = "n02512053"
ids = synset_to_dfs_ids(s)

for f in glob.glob("../input/test_stg1/*.jpg"):
    print("Generating heatmap for {}".format(f))
    im = preprocess_image_batch([f], color_mode="rgb")
    out = model.predict(im)
    heatmap = out[0,ids].sum(axis=0)
    plt.imsave(f.replace("jpg","png"), heatmap)

