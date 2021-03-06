import os
import random

# preprocessing
!curl https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label > ./data/train_5500.label
!tr "[:upper:]" "[:lower:]" < ./data/train_5500.label > ./data/train_lc.txt






#
# if __name__ == "__main__":
#     Main()