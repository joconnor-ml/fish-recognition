import glob

for f in glob.glob("/home/alpha/fish/test_stg1/*.jpg"):
    print(f.replace("/home/alpha/fish/", "gs://fish_bucket/input/"))
