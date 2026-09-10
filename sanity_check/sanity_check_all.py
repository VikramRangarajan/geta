import os
import unittest

# from sanity_check.test_qmlp import TestQMLP as TestQMLP
# from sanity_check.test_qvgg7bn import TestQVGG7BN as TestQVGG7BN
# from sanity_check.test_qresnet18 import TestQResNet18 as TestQResNet18
# from sanity_check.test_qresnet20 import TestQResNet20 as TestQResNet20
# from sanity_check.test_qresnet56 import TestQResNet56 as TestQResNet56
# from sanity_check.test_qresnet50 import TestQResNet50 as TestQResNet50
# from sanity_check.test_qbert import TestQBert as TestQBert
# from sanity_check.test_qcarn import TestQCARN as TestQCARN
from sanity_check.test_qyolov5 import TestQYolov5 as TestQYolov5
# from sanity_check.test_qsimplevit import TestQSimpleViT as TestQSimpleViT
# from sanity_check.test_qphi2 import TestQPhi2 as TestQPhi2
# from sanity_check.test_qvit import TestQViT as TestQViT
# from sanity_check.test_qdeit import TestQDeiT as TestQDeiT
"""
Quantization test cases
"""

OUT_DIR = "./cache"

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    unittest.main()
