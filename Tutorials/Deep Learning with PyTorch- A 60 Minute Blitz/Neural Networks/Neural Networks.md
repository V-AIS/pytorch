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

---
원문 내용을 남깁니다. (번역이 조금 이상함)
- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient


### Define the Network

다음과 같이 정의된 네트워크가 있습니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

Out:
```python
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

Neural Network를 설계 시 사용하려고 하는 `forward` 함수를 정의해야 하고, `autograd`를 사용하여 `backward`함수가 자동으로 정의됩니다. 또한 `forward`함수에서 Tensor 연산 중 어떤 것이든 사용할 수 있습니다.

[You just have to define the forward function, and the backward function (where gradients are computed) is automatically defined for you using autograd. You can use any of the Tensor operations in the forward function.]

모델의 학습 가능한 매개변수는 `net.parameters()`에 의해 반환됩니다.

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

Out:
```python
10
torch.Size([6, 1, 5, 5])
```

`params[0]`을 출력해보면 다음과 같습니다.

```python
print(params)
```

Out:
```python
Parameter containing:
tensor([[[[ 0.1034, -0.0988,  0.1250,  0.0279,  0.0660],
          [-0.0900,  0.0648,  0.0913, -0.1100, -0.0952],
          [ 0.1193, -0.0503, -0.0158, -0.0222, -0.0587],
          [-0.0113, -0.0887, -0.1037,  0.1452, -0.1831],
          [-0.1495, -0.1265,  0.1146, -0.1421,  0.0318]]],


        [[[ 0.0027, -0.1232, -0.1548, -0.0617,  0.0479],
          [-0.0604, -0.1905,  0.0868, -0.1042, -0.0927],
          [-0.1623,  0.0648,  0.0696,  0.0634, -0.0589],
          [-0.1135, -0.0570, -0.1559, -0.1959,  0.0934],
          [-0.1314,  0.1166, -0.1650, -0.0665, -0.1742]]],


        [[[ 0.1190, -0.0092, -0.1896,  0.1622,  0.0927],
          [-0.1759,  0.0650,  0.1655,  0.1558, -0.1435],
          [ 0.1056, -0.1333, -0.0357, -0.0320, -0.0244],
          [-0.0703, -0.0899, -0.1310,  0.0903,  0.0123],
          [-0.1549,  0.0136,  0.1115,  0.0952,  0.1671]]],


        [[[-0.0549,  0.0117,  0.1551, -0.0947,  0.0417],
          [ 0.0998,  0.0179, -0.0164, -0.0906, -0.1041],
          [ 0.0213, -0.0065, -0.1621,  0.0399,  0.0773],
          [ 0.1727, -0.0342, -0.0354,  0.0622,  0.1634],
          [ 0.0222,  0.1923,  0.1532, -0.1708, -0.0405]]],


        [[[-0.0308, -0.1026,  0.1333, -0.0865, -0.0271],
          [-0.0888, -0.0807,  0.0762,  0.0232, -0.1233],
          [-0.0884, -0.1353,  0.1691,  0.0181,  0.1718],
          [ 0.1438,  0.0204,  0.0967, -0.1715, -0.0327],
          [ 0.1255, -0.1491,  0.0587,  0.1702, -0.0402]]],


        [[[ 0.0071, -0.1428,  0.1008,  0.0564, -0.1433],
          [ 0.1807,  0.1830,  0.0948, -0.1903,  0.1351],
          [-0.0362, -0.1320, -0.0197,  0.0628, -0.0599],
          [ 0.1773, -0.0578,  0.0933, -0.1763, -0.1591],
          [ 0.0528, -0.0764, -0.0728,  0.1600,  0.0447]]]],
       requires_grad=True)
```

`6, 1, 5, 5`가 의미하는 바는 `5x5` Array 데이터가 `1`개 존재하고, 이러한 데이터가 `6`개 있다. 로 해석할 수 있습니다.


다음과 같이 32x32 크기의 임의의 값을 갖는 입력을 수행해보겠습니다. 참고: 이 네트워크(LeNet)에 대한 예상 입력 크기는 32x32입니다. 이 네트워크를 MNIST 데이터 세트에 사용하려면 데이터 세트의 이미지를 32x32 크기로 조정해야 합니다.

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

Out:
```python
tensor([[ 0.1246, -0.0511,  0.0235,  0.1766, -0.0359, -0.0334,  0.1161,  0.0534,
          0.0282, -0.0202]], grad_fn=<ThAddmmBackward>)
```

임의의 기울기(Gradients)를 갖는 역전파(Backprops)와 모든 파라미터들의 기울기 버퍼를 0으로 합니다:
```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

#### NOTE
`torch.nn`은 오직 미니-배치(mini-batches)만을 지원합니다. 전체 `torch.nn`패키지는 단일 샘플이 아닌 샘플의 미니-배치 입력만을 지원합니다.
예를 들어, `nn.Conv2d`는 `nSamples x nChannels x Height x Width`의 4D Tensor를 가져옵니다.
만약 단일 샘플이 있는 경우, `input.unsqueeze(0)`를 사용하여 가짜 배치-차원(batch-dimension)을 만들어야 합니다.

계속 진행하기에 앞서 지금까지의 내용을 요약하겠습니다.
#### Recap:
- `torch.Tensor` - `backward()`와 같은 `autograd` 연산을 지원하는 다차원 배열입니다. 또한 Tensor에 관한 그라디언트를 유지합니다.
- `nn.Module` - 신경망 모듈(Neural Network Module)입니다. 매개 변수를 캡슐화 하는 편리한 방법이며 GPU 연산 지원, 네트워크 결과 내보내기(Exporting), 네트워크 결과 불러오기(Loading) 등 다양한 기능을 지원합니다.
- `nn.Parameter` - 모듈에 속성으로 지정되면 매개 변수로 자동 등록되는 Tensor의 일종입니다.
- `autograd.Function` - `autograd` 연산의 앞(forward)과 backward 정의를 구현합니다. 모든 Tensor 연산은 Tensor를 작성하고, 그 히스토리를 인코딩하는 함수에 연결하는 하나 이상의 Function 노드를 작성합니다.

#### At this point, we covered:
- 신경망(Neural Network) 작성(정의)하기
- 입력 처리 및 backward 호출

#### Still Left:
- 손실(Loss) 계산
- 네트워크의 가중치(Weights) 업데이트

### LOSS FUNCTION
손실 함수는 (출력, 대상) 입력 쌍을 가져와서 목표 값에서 출력이 얼마나 떨어져 있는지 추정하는 값 입니다.
nn 패키지에는 여러 가지 손실 함수가 있습니다. 간단한 손실은 다음과 같습니다: `nn.MSELoss`는 입력과 대상간의 평균 제곱 오차를 계산합니다.

간단한 예제:
```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

Out:
```python
tensor(1.3638, grad_fn=<MseLossBackward>)
```

---
