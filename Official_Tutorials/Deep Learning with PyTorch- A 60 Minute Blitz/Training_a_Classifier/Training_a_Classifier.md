## TRAINING A CLASSIFIER
이번 튜토리얼에서는 뉴럴 네트워크를 어떻게 정의하고, 손실(Loss)을 계산하고 네트워크의 가중치를 업데이트 하는 방법을 알아보겠습니다.

### WHAT ABOUT DATA?
일반적으로, 이미지, 텍스트, 음성 또는 비디오 데이터를 다룰 때 데이터를 Numpy Array로 불러오는 표준 Python 패키지를 사용할 수 있습니다. 그 다음 Numpy Array를 `torch.*Tensor`로 변환할 수 있습니다.

- 이미지 데이터의 경우 **Pillow**, **OpenCV**와 같은 패키지가 유용합니다.
- 음성 데이터의 경우 **scipy**, **librosa**와 같은 패키지가 있습니다.
- 텍스트 데이터의 경우 Python 또는 Cython기반 데이터 불러오기 또는 **NLTK**와 **SpaCy** 패키지가 유용합니다.

특별히 Vision 관련하여, 우리는 `torchvision`이라 불리는 패키지를 만들었습니다. 이것은 ImageNet, CIFAR10, MNIST와 같은 공용 데이터셋을 불러오거나 이미지, viz, `torchvision.datasets`과 `torch.utils.data.DataLoader`와 같은 데이터 변환 작업을 합니다.

---
