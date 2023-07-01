# ERA_Assgn9

Write a new network that

1) has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2) total RF must be more than 44
3) one of the layers must use Depthwise Separable Convolution
4) one of the layers must use Dilated Convolution
5) use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6) use albumentation library and apply:
  * horizontal flip
  * shiftScaleRotate
  * coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7) achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
8) make sure you're following code-modularity (else 0 for full assignment)
9) upload to Github
10) Attempt S9-Assignment Solution.
