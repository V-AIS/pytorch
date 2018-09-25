## Neural Networks

신경망(Neural Networks)은 `torch.nn` 패키지에 구성되어 있습니다.
`autograd`와 `nn`을 간략하게 설명하면, 모델들을 정의하고 구별하는 역할을 합니다. `nn.Module`은 신경망의 레이어와 결과를 반환하는 메소드로 `forward(input)`을 입력받아 `output`을 출력합니다.

간단한 예제를 살펴보면, 숫자 이미지를 분류하는 네트워크는 다음과 같습니다.

![](https://pytorch.org/tutorials/_images/mnist.png)


