import unittest
import os

"""
Quantization test cases
"""
from sanity_check.test_qmlp import TestQMLP
from sanity_check.test_qvgg7bn import TestQVGG7BN
from sanity_check.test_qresnet18 import TestQResNet18
from sanity_check.test_qresnet20 import TestQResNet20
from sanity_check.test_qresnet56 import TestQResNet56
from sanity_check.test_qresnet50 import TestQResNet50
from sanity_check.test_qbert import TestQBert
from sanity_check.test_qcarn import TestQCARN
from sanity_check.test_qyolov5 import TestQYolov5
from sanity_check.test_qsimplevit import TestQSimpleViT
from sanity_check.test_qphi2 import TestQPhi2
from sanity_check.test_qvit import TestQViT
from sanity_check.test_qdeit import TestQDeiT

OUT_DIR = "./cache"

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    unittest.main()
