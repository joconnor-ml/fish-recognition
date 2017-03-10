import vgg16bn
import dl_utils

print("Building model")
vgg = vgg16bn.Vgg16BN(include_top=False, size=(360, 640))
print("Reading data")
trn = dl_utils.get_data('/home/alpha/fish/train/', (360,640))
print("Computing CNN features")
features, labels, classes = vgg.predict(trn)
print("Saving")
dl_utils.save_array("cnn_features.dat", features)
dl_utils.save_array("labels.dat", labels)
dl_utils.save_array("classes.dat", classes)
