# Reference : https://ieeexplore.ieee.org/document/5672785 ViBe: A Universal Background Subtraction Algorithm for Video Sequences

import random
import numpy as np

def euclidean_distance(i, j):
    if len(i) == 3:
        return ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2) ** 0.5
    else:
        return abs(i - j)

def get_random_nbrs(i,j,height,width):
    y = i + random.randint(-1, 1)

    if y < 0:
        y = 0
    if y > height - 1:
        y = height - 1

    x = j + random.randint(-1, 1)

    if x < 0:
        x = 0
    if x > width - 1:
        x = width - 1

    return y,x

def init_background(frame, N):
    height = frame.shape[0]
    width = frame.shape[1]
    channel = 1

    if len(frame.shape) == 3:
        channel = frame.shape[2]
        
    samples = np.zeros((height, width, N, channel), np.int32)
    for i in range(height):
        for j in range(width):
            for k in range(N):
                (y,x) = get_random_nbrs(i,j,height,width)
                samples[i][j][k][:] = frame[y, x, :]
    return samples

def update_step(frame, samples, min_count, N, Radius, Phi):
    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros((height, width), np.uint8)
    
    for i in range(height):
        for j in range(width):
            count = 0
            index = 0
            while(count < min_count and index < N):
                distance = euclidean_distance(frame[i][j], samples[i][j][index])
                if distance < Radius:
                    count += 1
                index += 1

            # Pixel is part of background
            if (count >= min_count):
                rand = random.randint(0, Phi - 1)

                # Update current pixel model
                if rand == 0:
                    rand_index = random.randint(0, N - 1)
                    samples[i][j][rand_index] = frame[i][j]

                rand = random.randint(0, Phi - 1)

                # Update neighboring pixel model
                if rand == 0:
                    rand_index = random.randint(0, N - 1)
                    y,x = get_random_nbrs(i,j,height,width)
                    samples[y][x][rand_index] = frame[i][j]

            else:
                mask[i][j] = 255 # classify as foreground

    return mask, samples    

class Vibe:
    def __init__(self, min_count=2, N=20, Radius=20, Phi=16):
        self.min_count = min_count
        self.N = N
        self.Radius = Radius
        self.Phi = Phi

    def init(self, frame):
        self.samples = init_background(frame, self.N) 
        
    def test_and_update(self, frame):
        mask, self.samples = update_step(frame, self.samples, self.min_count, self.N, self.Radius, self.Phi)
        return mask

def get_vibe_masks(inp_frames, s, e):
    vibe = Vibe()
    vibe.init(inp_frames[0][1].astype(np.int32))

    for i in range(1,s-1):
        if i % 10 == 0:
            print(i)
        vibe.test_and_update(inp_frames[i][1].astype(np.int32))
    out_frames = []
    for i in range(s-1,e):
        if i % 10 == 0:
            print(i)
        mask = vibe.test_and_update(inp_frames[i][1].astype(np.int32))
        out_frames.append((inp_frames[i][0], mask))
        
    return out_frames

