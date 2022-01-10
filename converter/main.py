import os
import sys
import random
import numpy as np
from PIL import Image as im


def generate_matrix(lines, columns, fout):
    fout.write(f"{lines} {columns}\n")
    for i in range(lines):
        line = [random.randint(0, 255) for j in range(columns)]
        fout.write(f"{' '.join((str(a) for a in line))}\n")
    fout.write("\n")


def generateMatrix():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} matrix_lines matrix_columns kernel_lines kernel_columns")
        return
    ml, mc, kl, kc = [int(a) for a in sys.argv[1:]]
    filename = f"matrix_{ml}_{mc}_kernel_{kl}_{kc}.txt"
    if filename in os.listdir():
        print(filename)
        return
    with open(filename, "w") as fout:
        generate_matrix(ml, mc, fout)
        generate_matrix(kl, kc, fout)
    print(filename)


def convertFromMatrix():
    pixels = list()
    with open("output.txt", "r") as fin:
        for line in fin:
            pixels.append([float(pixel) for pixel in line.split()])
    array = np.array(pixels)
    data = im.fromarray(array).convert("L")
    data.save("gray_output.png")


def convertToMatrix():
    img = im.open("input.png").convert("L")
    with open("input.txt", "w") as fout:
        fout.write(f"{img.height} {img.width}\n")
        matrix = np.array(img).reshape((img.height, img.width))
        for line in matrix:
            fout.write(f"{' '.join((str(pixel) for pixel in line))}\n")
        fout.write("7 7\n"
                   "0.00000067 0.00002292 0.00019117 0.00038771 0.00019117 0.00002292 0.00000067\n"
                   "0.00002292 0.00078633 0.00655965 0.01330373 0.00655965 0.00078633 0.00002292\n"
                   "0.000191117 0.00655965 0.05472157 0.11098164 0.05472157 0.00655965 0.00019117\n"
                   "0.00038771 0.01330373 0.11098164 0.22508352 0.11098164 0.01330373 0.00038771\n"
                   "0.00019117 0.00655965 0.05472157 0.11098164 0.05472157 0.00655965 0.00019117\n"
                   "0.00002292 0.00078633 0.00655965 0.01330373 0.00655965 0.00078633 0.00002292\n"
                   "0.00000067 0.00002292 0.00019117 0.00038771 0.00019117 0.00002292 0.00000067\n")
    img.save("gray_input.png")


if __name__ == '__main__':
    convertToMatrix()
    # convertFromMatrix()
