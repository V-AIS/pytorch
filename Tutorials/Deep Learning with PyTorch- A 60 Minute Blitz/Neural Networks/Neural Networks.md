## Neural Networks

신경망(Neural Networks)은 `torch.nn` 패키지에 구성되어 있습니다.
`autograd`와 `nn`을 간략하게 설명하면, 모델들을 정의하고 구별하는 역할을 합니다. `nn.Module`은 신경망의 레이어와 결과를 반환하는 메소드로 `forward(input)`을 입력받아 `output`을 출력합니다.

간단한 예제를 살펴보면, 숫자 이미지를 분류하는 네트워크는 다음과 같습니다.

![](https://github.com/V-AIS/pytorch/blob/master/Tutorials/Deep%20Learning%20with%20PyTorch-%20A%2060%20Minute%20Blitz/Neural%20Networks/mnist.png)

단순한 피드 포워드 네트워크(Feed-Forward Network)입니다. 입력을 받아, 여러 레이어를 차례로 전달한 다음 최종적으로 출력을 제공합니다.

신경망의 일반적인 학습 단계는 다음과 같습니다.

- 학습 가능한 매개변수(또는 가중치)가 있는 신경망을 정의합니다.
- 입력 데이터셋에 대해 반복
- 네트워크를 통해 입력 프로세스 (설계된 네트워크 구조와 초기화된 가중치에 대해 전방 계산을 의마하는 듯 합니다. [Process input through the network])
- 출력이 정확해질 때 까지 손실(Loss)을 계산합니다.
- 그라디언트를 네트워크 매개변수로 다시 전파합니다. (Back-Propagate 방법)
- 가중치를 업데이트 합니다. 일반적으로 간단한 법칙은 다음과 같습니다. `weight = weight - learning_rate * gradient`


### Define the Network
