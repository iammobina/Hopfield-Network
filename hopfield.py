import numpy as np
from PIL import Image, ImageFont
import statistics

from PIL._imaging import display


def update_weights(w, a, n):
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                w[i][j] += a[i] * a[j]
            else:
                w[i][j] = 0
    return w


def letter_to_pattern(letter, n):
    pattern = letter.reshape(n)
    pattern = pattern.astype('int32')
    for i in range(0, n):
        if pattern[i] > 0:
            pattern[i] = -1
        else:
            pattern[i] = 1
    return pattern


def show(b, n, pixelN):
    result = []
    for pixel in range(0, n):
        if b[pixel] == 1:
            result.append(255)
        else:
            result.append(1)
    l = np.array(result).reshape(pixelN, pixelN)
    l = l.astype('uint8')
    ll = Image.fromarray(l)
    ll = ll.convert("L")
    display(ll)



pixel = 15
n = pixel * pixel  # number of neurons
p = 10  # number of patterns
w = np.zeros((n, n))
patterns = np.zeros((p, n))
font_size = 16
font = ImageFont.truetype("tahoma.ttf", font_size)

# train network
patternCounter = 0
for char in "ABCDEFGHIJ":
    im = Image.Image()._new(font.getmask(char))
    im = im.resize((pixel, pixel), 0)
    im.save(char + ".bmp")
    letter = np.array(im)
    patterns[patternCounter] = letter_to_pattern(letter, n)
    w = update_weights(w, patterns[patternCounter], n)
    patternCounter += 1

noisep = 0.1
noisyP = np.zeros((p, n))

for j in range(0, p):
    indexes = np.random.randint(n, size=int(noisep * n))
    tempP = np.copy(patterns[j])
    for i in indexes:
        if tempP[i] > 0:
            tempP[i] = -1
        else:
            tempP[i] = 1
    noisyP[j] = tempP


def result():
    accuracy = []
    for i in range(0, p):
        state0 = np.copy(noisyP[i])
        state1 = np.copy(state0)
        true = 0
        iteration = 0
        while iteration < 100:
            iteration += 1
            for pix in range(0, n):
                if np.sum(state1 * w[pix]) > 0:
                    state1[pix] = 1
                else:
                    state1[pix] = -1
            if state1.all == state0.all:
                break
            else:
                continue
            state0 = np.copy(state1)

        for pixels in range(0, n):
            if state1[pixels] == patterns[i][pixels]:
                true += 1

        show(noisyP[i], n, pixel)
        show(state1, n, pixel)
        accuracy.append(true / n)
        print(accuracy[i])

    print(statistics.mean(accuracy))

