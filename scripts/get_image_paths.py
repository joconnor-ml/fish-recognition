import glob

for f in glob.glob("/home/alpha/fish/train/*/*.jpg"):
    print(f.replace("/home/alpha/fish/", "gs://fish_bucket/input/"))
