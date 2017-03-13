import os
import json
import pandas as pd
import glob

train_dir = "/home/alpha/fish/train/"


def read_boxes():
    boxes = []
    for f in glob.glob("/home/alpha/annotations/*.json"):
        print(f)
        j = json.load(open(f))
        for element in j:
            try:
                box = element["annotations"][0]
            except:
                box = {}
            box["class"] = element["class"]
            box["filename"] = element["filename"].replace("../data/", "gs://fish_bucket/input/")
            boxes.append(box)

    return pd.DataFrame(boxes).set_index("filename").drop("class", axis=1)


def plot_with_box(path, height=0, width=0, x=0, y=0, color="red"):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img=mpimg.imread(path)
    imgplot = plt.imshow(img)
    rect = plt.Rectangle((x, y), width, height, color=color, fill=False, lw=3)
    plt.gca().add_patch(rect)
    plt.show()


if __name__=="__main__":
    df = read_boxes().fillna(0).astype(int)
    #fname = "img_07915.jpg"
    #row = df.loc[fname]
    #print(row)
    #plot_with_box(os.path.join(train_dir, "ALB", fname), row["height"], row["width"], row["x"], row["y"])
    df.to_csv("train_boxes.csv")
