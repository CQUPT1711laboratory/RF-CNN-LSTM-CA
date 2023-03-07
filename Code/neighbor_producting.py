import numpy as np

landuse_ori = np.loadtxt('../cq21/data/cq2010_res6.txt', skiprows=6)
landuse=np.array(landuse_ori)

# the neighbor density of 9*9 window
neighbors=[]
for x in range(landuse.shape[0]):
    for y in range(landuse.shape[1]):
        s = landuse[x - 4:x + 5, y - 4:y + 5]
        # sf = s.reshape(-1)
        nei_temp=[]
        if -9999 == landuse[x,y]:
            neighbors.append([0, 0, 0, 0, 0, 0])
            continue
        for i in range(6):
            # neighbor = (len(sf[sf == i+1]) - (1 if i+1 == landuse[x,y] else 0)) / (49 - len(sf[sf == -9999]))
            neighbor = len(s[s == i+1])
            nei_temp.append(neighbor)
        neighbors.append(nei_temp)
np.savetxt(f'../cq21/driver/neig43_2010.txt', neighbors, fmt='%f', delimiter=' ')


# the g extended neighborhood enrichment data
nei81 = np.loadtxt(f'../cq21/driver/neig81_2010.txt')
nei9 = np.loadtxt(f'../cq21/driver/neig9_2010.txt')
for i in range(nei81.shape[0]):
    for j in range(nei81.shape[1]):
        if nei81[i, j] != 0:
            nei81[i, j] = nei9[i, j] / nei81[i, j]
# bili = nei9 / nei81
np.savetxt(f'../cq21/driver/neig2010part.txt', nei81, fmt='%f', delimiter=' ')

