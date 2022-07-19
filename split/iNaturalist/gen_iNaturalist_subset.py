import random
import numpy as np
from pdb import set_trace


def subsample_iNaturalist_to_sub1000():
  readFiles = ["iNaturalist18_train.txt", "iNaturalist18_val.txt"]
  writeFiles = ["iNaturalist18_sub1000_train.txt", "iNaturalist18_sub1000_val.txt"]

  random.seed(0)
  classNum = 8142
  classList = [i for i in range(classNum)]
  random.shuffle(classList)
  classList = classList[:1000]
  classList2newClass = {c: cnt for cnt, c in enumerate(classList)}

  for readFile, writeFile in zip(readFiles, writeFiles):
    linesFiltered = []
    with open(readFile, "r") as f:
      for line in f.readlines():
        folder = line.split('/')[1]
        address = line.split(' ')[0]
        category = int(line.split(' ')[1].strip("\n"))
        if category in classList:
          linesFiltered.append("{} {}\n".format(address, classList2newClass[category]))

    with open(writeFile, "w") as f:
      f.writelines(sorted(linesFiltered))


def sampleFull0_1subset():
  readFile = "iNaturalist18_train.txt"
  writeFile = "iNaturalist18_balance_train_0.1.txt"
  num_class = 8142

  f = open(readFile, "r")
  # print(f.readlines())

  linesEachClass = [[] for _ in range(num_class)]

  for line in f.readlines():
    address = line.split(' ')[0]
    classId = int(line.split(' ')[1][:-1])
    folder = line.split('/')[1]
    linesEachClass[classId].append(line)

  linesToWrite = []
  random.seed(0)

  linesEachClass_lenList = [len(lines) for lines in linesEachClass]
  print("smallest class number is {}".format(min(linesEachClass_lenList)))

  for cntLine, classLines in enumerate(linesEachClass):
    random.shuffle(classLines)
    linesToWrite += classLines[:5]

  # set_trace()
  # print(classDict.keys())
  with open(writeFile, 'a') as the_file:
    # print("{} {}".format(join('train', folder, path), classId))
    the_file.writelines(sorted(linesToWrite))


if __name__ == "__main__":
  sampleFull0_1subset()
