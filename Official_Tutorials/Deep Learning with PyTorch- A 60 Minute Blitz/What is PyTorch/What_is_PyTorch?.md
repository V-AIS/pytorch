본 문서는 공식 홈페이지에서 제공하는 튜토리얼을 따르고 있습니다. 하지만, 튜토리얼을 따라하며 학습 시 필요한 내용이 추가될 수 있습니다.

## What is PyTorch?

Python 기반의 데이터 사이언스 패키지로 다음과 같은 특징이 있습니다.

- GPU 연산을 활용하여 NumPy 라이브러리 대체 가능
- 쉽고 빠르게 적용할 수 있는 딥러닝 플랫폼

### Getting Started

#### Tensors

텐서(Tensors)는 NumPy의 ndarrays와 유사하며, GPU를 활용하면 빠른 속도로 연산이 가능합니다.

```python
from __future__ import print_function
import torch
```

초기화하지 않은 5x3 Matrix를 선언하면 다음과 같습니다. (결과가 0.0000으로 보이지만 실제로는 초기화 되지 않은 값 입니다.)

```python
x_empty = torch.empty(5, 3)
print(x_empty)
```
Out:
```python
tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
```

또한, 임의의 값으로 초기화된 Matrix는 다음과 같습니다.

```python
x_rand = torch.rand(5, 3)
print(x_rand)
```

Out:

```python
tensor([[0.4995, 0.2337, 0.4023],
        [0.3653, 0.3278, 0.7065],
        [0.4357, 0.1649, 0.9156],
        [0.1494, 0.7832, 0.3087],
        [0.2128, 0.4301, 0.6539]])
```

특정 데이터 타입으로도 초기화가 가능합니다.

```python
x_zero = torch.zeros(5, 3, dtype=torch.float64)
print(x_zero)
```

Out:

```python
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
```

Tensor를 직접 만드는 것 또한 가능합니다.

```python
x_tensor = torch.tensor([5.5, 3])
print(x_tensor)
```

Out:

```python
tensor([5.5000, 3.0000])
```

또는, 이미 존재하는 Tensor를 기반으로 만들 수 있습니다. 이 방법은 이미 존재하는 Tensor의 특성(데이터 타입, Tensor 크기)를 다시 사용합니다. 명확한 이해를 위해 공식 문서에서 하나의 예제로 진행한 부분을 두 개의 예제로 살펴보겠습니다.

```python
tensor = torch.tensor((), dtype=torch.int32)
print(tensor)
tensor = tensor.new_ones((5, 3))
print(tensor)
```

Out:
```python
tensor([], dtype=torch.int32)
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
```

위와 같이 tensor 변수는 데이터 타입이 'int32'로 선언되었습니다. PyTorch API가 제공하는 `new_ones` 메소드를 통해 동일한 데이터 타입을 갖는 5x3 Tensor가 만들어졌습니다. 다음 예제는 Tensor의 크기를 유지한 채 데이터 타입을 변경하는 `randn_like` 메소드를 살펴보겠습니다.

```python
# zeros torch
x = torch.zeros(5, 3, dtype=torch.float64)
print(x)

# random torch
x = torch.randn_like(x, dtype=torch.float64)
print(x)
```

Out:
```python
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)

tensor([[ 0.8878,  0.8595, -1.3904],
        [ 1.2738,  1.4750,  1.4978],
        [-0.1091,  3.4508, -0.1414],
        [-0.2331, -0.1284, -1.1919],
        [ 1.1709, -0.9909,  1.3729]], dtype=torch.float64)
```

앞서 선언한 5x3 Tensor 변수의 데이터 타입이 변경되는 것을 확인할 수 있습니다. 명확한 이해를 위해 `new_ones`와 `randn_like` 메소드 정의를 살펴보겠습니다.

```python
new_ones(size, dtype=None, device=None, requires_grad=False) → Tensor
```
매개변수로 입력되는 크기에 '1'로 채워진 Tensor를 반환합니다. 크기는 리스트, 튜플 또는 `torch.Size`로 입력 가능합니다. 또한 기본값으로 지정된 'dtype', 'device'가 'None' 이면 동일한 타입으로 반환합니다. [more detail](https://pytorch.org/docs/stable/tensors.html?highlight=new_ones#torch.Tensor.new_ones)


```python
torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
```
매개변수로 입력되는 Tensor와 동일한 크기로 임의의 값을 갖는 Tensor를 반환합니다. `new_ones`와는 다르게 'input'의 데이터 타입이 'Tensor'라는 것을 주의해야 합니다. 또한 반환되는 Tensor는 평균이 0이고 분산이 1인 정규 분포를 따릅니다. [more detail](https://pytorch.org/docs/stable/torch.html?highlight=randn_like#torch.randn_like)

다음으로 Tensor의 크기를 얻는 방법을 알아보겠습니다.

```python
print(x.size())
```

Out:
```python
torch.Size([5, 3])
```
Tensor의 크기에서 유의할 점은 `torch.Size`는 튜플이며, 모든 튜플 연산을 지원합니다.

---
#### Operations

PyTorch는 다양한 연산자를 지원합니다. 아래 예제들을 따라하며 연산자를 확인해보겠습니다. 본 예제를 진행하기에 앞서 두 개의 임의의 변수를 생성합니다.
```python
x_rand = torch.rand(5, 3)
print(x_rand)

y_rand = torch.rand(5, 3)
print(y_rand)
```

Out:
```python
tensor([[0.5692, 0.5885, 0.0235],
        [0.8802, 0.6941, 0.5231],
        [0.6548, 0.3362, 0.5381],
        [0.5548, 0.0083, 0.6275],
        [0.0569, 0.1905, 0.3911]])
tensor([[0.6952, 0.9283, 0.3290],
        [0.2941, 0.3114, 0.4589],
        [0.6562, 0.2204, 0.1503],
        [0.3551, 0.1386, 0.0938],
        [0.9927, 0.1702, 0.8566]])
```

더하기: 구문 1
```python
print(x_rand + y_rand)
```

Out:
```python
tensor([[1.2644, 1.5168, 0.3525],
        [1.1742, 1.0055, 0.9819],
        [1.3110, 0.5566, 0.6884],
        [0.9098, 0.1468, 0.7213],
        [1.0496, 0.3606, 1.2477]])
```

더하기: 구문2
```python
print(torch.add(x_rand, y_rand))
```

Out:
```python
tensor([[1.2644, 1.5168, 0.3525],
        [1.1742, 1.0055, 0.9819],
        [1.3110, 0.5566, 0.6884],
        [0.9098, 0.1468, 0.7213],
        [1.0496, 0.3606, 1.2477]])
```

추가: 출력 Tensor를 임의의 변수를 통해 지정하는 것 또한 가능합니다.

```python
result = torch.empty(5, 3)
torch.add(x_rand, y_rand, out=result)
print(result)
```

Out:
```python
tensor([[1.2644, 1.5168, 0.3525],
        [1.1742, 1.0055, 0.9819],
        [1.3110, 0.5566, 0.6884],
        [0.9098, 0.1468, 0.7213],
        [1.0496, 0.3606, 1.2477]])
```

추가: In-Place 기능을 지원합니다.
```python
# adds x_rand to y_rand
y_rand.add_(x_rand)
print(y_rand)
```

Out:
```python
tensor([[1.2644, 1.5168, 0.3525],
        [1.1742, 1.0055, 0.9819],
        [1.3110, 0.5566, 0.6884],
        [0.9098, 0.1468, 0.7213],
        [1.0496, 0.3606, 1.2477]])
```

어떤 연산자들은 `_`를 후미에 붙여 Tensor를 내부적으로 변화하게 합니다. 예를 들어 `x.copy_(y)`, `x.t_()`와 같은 연산은 `x`의 값을 변경합니다.
또한, NumPy와 비슷한 방법으로 Tensor Indexing이 가능합니다.

```python
print(x_rand[:, 1])
```

Out:
```python
tensor([0.5885, 0.6941, 0.3362, 0.0083, 0.1905])
```

Resizing: Tensor의 크기를 변경하거나, 교체하고 싶으면 `torch.view`를 활용할 수 있습니다.

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print("Original Tensor : ", x)
print("View (16) : ", y)
print("View (-1, 8) : ", z)
print(x.size(), y.size(), z.size())
```

Out:
```python
Original Tensor :  tensor([[-0.6173,  1.2004, -1.2844, -0.2644],
        [ 0.6293,  0.4616, -0.3915,  0.5739],
        [-0.0571,  0.2191,  0.8552, -0.2459],
        [ 0.0922, -0.1486,  1.1691,  0.2234]])
View (16) :  tensor([-0.6173,  1.2004, -1.2844, -0.2644,  0.6293,  0.4616, -0.3915,  0.5739,
        -0.0571,  0.2191,  0.8552, -0.2459,  0.0922, -0.1486,  1.1691,  0.2234])
View (-1, 8) :  tensor([[-0.6173,  1.2004, -1.2844, -0.2644,  0.6293,  0.4616, -0.3915,  0.5739],
        [-0.0571,  0.2191,  0.8552, -0.2459,  0.0922, -0.1486,  1.1691,  0.2234]])
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

`torch.view`에 입력되는 매개변수의 값에 따라 출력되는 Tensor의 Shape가 달라지는 것을 확인할 수 있습니다.


만약 한 개의 요소를 갖는 Tensor가 있다면 `.item()`를 사용해서 Python 기본 자료형으로 변경할 수 있습니다. 다음 예제를 확인해보겠습니다.

```python
x = torch.rand(1)
print(x)
print(type(x))
```

Out:
```python
tensor([0.3797])
<class 'torch.Tensor'>
```

```python
py_x = x.item()
print(py_x)
print(type(py_x))
```

Out:
```python
0.3796881437301636
<class 'float'>
```

위와 같이 torch.Tensor 형태의 데이터가 float 형태로 변경되는 것을 확인할 수 있습니다.


---

### NumPy Bridge

PyTorch의 Tensor를 NumPy의 Array로 변환하거나, 그 반대로 변환하는 것을 매우 쉽게 할 수 있습니다.
Tensor와 Array는 기본 메모리를 공유하고 있기 때문에, 메모리의 주소를 변경하면 Tensor와 Array간 변경이 가능합니다.

- PyTorch(Tensor) -> NumPy(Array) and NumPy(Array) -> PyTorch(Tensor)

#### Converting a Torch Tensor to a NumPy Array

```python
a = torch.ones(5)
print(a)
```

Out:
```python
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
```

Out:
```python
[1. 1. 1. 1. 1.]
```

```python
a.add_(1) # add 1
print(a)
print(b)
```

Out:
```python
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```


#### Converting NumPy Array to Torch Tensor

다음 예제를 통해 NumPy Array를 자동으로 PyTorch Tensor로 변경하는 것을 확인할 수 있습니다.

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)
```

Out:
```python
[1. 1. 1. 1. 1.]
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

CharTensor를 제외한 모든 Tensors는 NumPy로 변경 또는 그 반대로 변경하는 것을 지원합니다. (All the Tensors on the CPU except a CharTensor support converting to NumPy and back.)

---

### CUDA Tensors

Tensors는 `.to` 메소드를 통해 어떤 디바이스로 이동할 수 있습니다. **NVIDIA CUDA 필**

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

Out:
```Python
tensor([1.3797], device='cuda:0')
tensor([1.3797], dtype=torch.float64)
```
