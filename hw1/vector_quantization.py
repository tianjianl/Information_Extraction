import sys
import numpy as np
import random
import matplotlib.pyplot as plt

def kmeans(vectors, n_clusters):
    # Initialize the centeroids
    
    #centeroids = vectors[:n_clusters, :]
    centeroids = np.random.uniform(low=[0, 0], high=[1, 1], size=(n_clusters,2))
    print(centeroids)
    labels = []
    while True:

        # Assign each vector to the closest centeroid
        new_labels = []
        for index, vector in enumerate(vectors):
            print([np.linalg.norm(vector - center) for center in centeroids])
            new_labels.append(np.argmin([np.linalg.norm(vector - center) for center in centeroids]))
        
        # If no labels are changed, then exit loop
        if np.array_equal(labels, new_labels):
            break
        
        labels = new_labels
        #print(labels)
        # Update the centeriods 
        for i in range(n_clusters):
            points_in_this_cluster = [vectors[j] for j in range(len(vectors)) if labels[j] == i]
            if (len(points_in_this_cluster) == 0):
                continue
            points_in_this_cluster = np.array(points_in_this_cluster)
            centeroids[i, :] = np.mean(points_in_this_cluster, axis=0)

        #print(centeroids)
    return labels, centeroids

def quantize(vectors, n_clusters, args):
    
    # perform K-means
    random.shuffle(vectors)
    vectors = np.array(vectors)
    labels, centeroids = kmeans(vectors, n_clusters)

    # plot 
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'c', 'y']
    for i in range(n_clusters):
        points_in_this_cluster = [vectors[j] for j in range(len(vectors)) if labels[j] == i]
        if len(points_in_this_cluster) == 0:
            continue
        ax.scatter(*zip(*points_in_this_cluster), c=colors[i], label=f'Cluster {i}')
    ax.scatter(*zip(*centeroids), marker='x', color='k', s=200, label='Centroids')
    ax.legend()
    plt.savefig(f"vector_quantization_{args[1]}.pdf")

def main(args):
    random.seed(int(args[1]))
    np.random.seed(int(args[1]))
    f = open("hw1-data.txt", "r")
    vectors = []
    for line in f:
        line = line.strip().split()
        vectors.append((float(line[0]), float(line[1])))
    
    quantize(vectors, 3, args)

if __name__ == "__main__":
    args = sys.argv
    main(args)
