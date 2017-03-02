import pandas as pd
import glob
import sys

threshold_distance = int(sys.argv[1])

clusters = []
for f in list(glob.glob("../input/train/*/*.jpg")):# + list(glob.glob("../input/test_stg1/*.jpg")):
    create_new = True
    for i, c in enumerate(clusters):
        if f in c:
            create_new = False
            current_cluster = i
            break
    if create_new:
        clusters.append(set([f]))
        current_cluster = len(clusters) - 1
        
    # inspect distance file
    try:
        distances = pd.read_csv(f.replace("jpg", "csv"), header=None, index_col=0)
    except pd.io.common.EmptyDataError:  # empty neighbours file
        continue
    distances = distances[(distances[1] > 0) &
                          (distances[1] < threshold_distance)]
    to_go = []
    for i in distances.index:
        clusters[current_cluster].add(i)
        for j, c in enumerate(clusters):
            if i in c and current_cluster != j:
                clusters[current_cluster] |= c
                to_go.append(j)
    clusters = [c for j, c in enumerate(clusters) if j not in to_go]
    print([len(c) for c in clusters])
    

for i, cluster in enumerate(clusters):
    print(len(cluster))
    pd.Series(list(cluster)).to_csv("clusters/cluster_{}.threshold_{}.csv".format(i, threshold_distance), index=False)
