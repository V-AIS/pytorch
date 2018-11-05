## What is PyTorch?

Python 기반의 데이터 사이언스 패키지로 다음과 같은 특징이 있습니다.

- GPU 연산을 활용하여 NumPy 라이브러리 대체 가능
- 쉽고 빠르게 적용할 수 있는 딥러닝 플랫폼

### Getting Started

#### Tensors(텐서)

Tensor(텐서)는 NumPy의 ndarrays와 유사하며, GPU를 사용하면 빠른 속도로 연산이 가능합니다.

```python
from __future__ import print_function
import torch
```

특정 값으로 초기화 되지 않은 5x3 Matrix(행렬)을 선언하면 다음과 같습니다.

```python
x = torch.empty(5, 3)
print(x)
```

결과:

```python
tensor([[2.6971e+26, 2.6791e+20, 7.3781e+28],
        [1.0901e+27, 1.0900e+27, 4.7424e+30],
        [4.7737e-26, 1.0141e+31, 2.8026e-45],
        [8.0421e+37, 2.3612e+21, 2.8026e-45],
        [0.0000e+00, 9.8091e-45, 5.6052e-45]])
```


임의의 값으로 초기화된 Matrix은 다음과 같습니다.

```python
x = torch.rand(5, 3)
print(x)
```

결과:

```python
tensor([[0.7341, 0.7378, 0.2923],
        [0.8512, 0.9130, 0.9494],
        [0.3597, 0.7727, 0.1666],
        [0.6024, 0.4822, 0.8710],
        [0.1085, 0.9170, 0.9652]])
```

다음은 특정 값으로(0), `long`데이터 타입으로 초기화된 결과입니다.

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

결과:

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

데이터로부터 직접 Tensor를 구성하는 방법입니다.

```python
x = torch.tensor([5.5, 3])
print(x)
```

결과:

```python
tensor([5.5000, 3.0000])
```

또는 이미 존재하는 Tensor를 기반으로 만들 수 있습니다. 이 메소드는 사용자가 새로운 값을 입력하지 않는 한 `dtpye`과 같은 입력 Tensor의 속성을 재사용합니다. 

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 메소드는 크기를 입력받습니다.
print(x)

x = torch.randn_like(x, dtype=torch.float)    # dtype을 다시 정의합니다. 
print(x)                                      # 입력 Tensor의 크기 속성을 그대로 사용합니다.
```

결과:
```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.8845, -2.3322, -0.1802],
        [-1.0453,  1.7780, -1.1918],
        [-0.5056,  0.3984, -2.0979],
        [-0.2763,  0.9354, -0.2681],
        [ 0.8258, -1.9075, -0.9735]])
```

Tensor의 크기 가져오기

```python
print(x.size())
```

결과:
```python
torch.Size([5, 3])
```

> PyTorch에서 `torch.Size`는 튜플이며, 모든 튜플 연산을 지원합니다.


#### Operations(연산자)

PyTorch는 다양한 연산자를 지원합니다. 다음 예제를 통해 더하기 연산을 살펴 보겠습니다.

```python
x = torch.rand(5, 3)
print(x)

y = torch.rand(5, 3)
print(y)
```

결과:

```python
tensor([[0.1616, 0.1034, 0.3041],
        [0.4666, 0.0138, 0.0562],
        [0.7390, 0.0986, 0.9688],
        [0.5225, 0.1712, 0.5383],
        [0.4920, 0.2012, 0.1466]])
tensor([[0.7867, 0.3962, 0.9957],
        [0.7881, 0.7694, 0.8252],
        [0.5988, 0.9533, 0.6214],
        [0.8035, 0.4542, 0.4429],
        [0.0796, 0.0278, 0.8022]])
```

더하기: 구문 1

```python
print(x + y)
```

결과:

```python
tensor([[0.9484, 0.4996, 1.2998],
        [1.2547, 0.7832, 0.8814],
        [1.3378, 1.0519, 1.5902],
        [1.3260, 0.6254, 0.9812],
        [0.5716, 0.2290, 0.9488]])
```

더하기: 구문2

```python
print(torch.add(x, y))
```

결과:

```python
tensor([[0.9484, 0.4996, 1.2998],
        [1.2547, 0.7832, 0.8814],
        [1.3378, 1.0519, 1.5902],
        [1.3260, 0.6254, 0.9812],
        [0.5716, 0.2290, 0.9488]])
```

> 추가: Argument로 출력 Tensor를 지정할 수 있습니다.

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

결과:
```python
tensor([[0.9484, 0.4996, 1.2998],
        [1.2547, 0.7832, 0.8814],
        [1.3378, 1.0519, 1.5902],
        [1.3260, 0.6254, 0.9812],
        [0.5716, 0.2290, 0.9488]])
```

추가: In-Place 기능을 지원합니다.
```python
# y에 x를 더하기
y.add_(x)
print(y)
```

결과:
```python
tensor([[0.9484, 0.4996, 1.2998],
        [1.2547, 0.7832, 0.8814],
        [1.3378, 1.0519, 1.5902],
        [1.3260, 0.6254, 0.9812],
        [0.5716, 0.2290, 0.9488]])
```

> Underbar(언더바, `_`)를 붙인 연산자들은 In-Place기능을 나타냅니다. 예를 들어: `x.copy_(y)`, `x.t_()`는 `x`가 변경됩니다.



NumPy와 비슷한 방법으로 Tensor(or Matrix, Array, ...) Indexing이 가능합니다.

```python
print(x[:, 1])
```

결과:
```python
tensor([0.1034, 0.0138, 0.0986, 0.1712, 0.2012])
```

Resizing: 만약 Tensor를 resize/reshape을 하고 싶다면 `torch.view`를 사용하세요.

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1은 다른 차원에 의해 자동으로 결정됩니다. (4x4 = 16 크기에서 8 row으로 설정했기 때문에 2 column으로 결정됩니다.)
print(x.size(), y.size(), z.size())
```

결과:
```python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

만약 한 개의 요소를 갖는 Tensor가 있다면 `.item()`를 사용해서 Python 숫자로 가져옵니다.

```python
x = torch.rand(1)
print(x)
print(type(x))
```

결과:
```python
tensor([-0.4116])
-0.41160470247268677
```

위와 같이 torch.Tensor 형태의 데이터가 float 형태로 변경되는 것을 확인할 수 있습니다.

> 추후 확인: Transposing(전치), Indexing(인덱싱), Slicing(분할), Mathematical Operations(수학 연산), Linear Algebra(선형 대수), Random Numbers(난수) 등 100개 이상의 Tensor 연산이 지원됩니다. 자세한 설명은 [이곳](http://pytorch.org/docs/torch)을 참고하세요.




### NumPy Bridge

PyTorch의 Tensor를 NumPy의 Array로 변환하거나, 그 반대로 변환하는 것을 쉽게 할 수 있습니다.
Tensor와 Array는 기본 메모리를 공유하고 있기 때문에, 메모리의 주소를 변경하면 Tensor와 Array간 변경이 가능합니다.

- PyTorch(Tensor) -> NumPy(Array) and NumPy(Array) -> PyTorch(Tensor)

#### Converting a Torch Tensor to a NumPy Array

```python
a = torch.ones(5)
print(a)
```

결과:
```python
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
```

결과:
```python
[1. 1. 1. 1. 1.]
```

다음 결과를 통해 NumPy Array가 어떻게 변하는지 확인하세요.

```python
a.add_(1) # add 1
print(a)
print(b)
```

결과:
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
np.add(a, 1, out=a)
print(a)
print(b)
```

결과:
```python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

CharTensor를 제외한 모든 Tensor는 NumPy로 변경 또는 그 반대로 변경하는 것을 지원합니다. (All the Tensors on the CPU except a CharTensor support converting to NumPy and back.)

---

### CUDA Tensors

Tensor는 `.to` 메소드를 통해 어떤 디바이스로 이동할 수 있습니다. **NVIDIA CUDA 필수**

```python
# CUDA를 실행할 수 있는 경우에만 이 예제를 실행하세요.
# 우리는 ``torch.device`` Tensor를 GPU에 할당합니다.
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA 디바이스 객체
    y = torch.ones_like(x, device=device)  # GPU에 직접 Tensor 생성
    x = x.to(device)                       # 또는 ``.to("cuda")``방법으로 GPU 설정
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 는 dtpye도 또한 변경됩니다.
```

결과:
```Python
tensor([1.3797], device='cuda:0')
tensor([1.3797], dtype=torch.float64)
```



#### Official Tutorial Python Code
- [Download Python Source Core: tensor_tutorial.py](https://pytorch.org/tutorials/_downloads/tensor_tutorial.py)
- [Download Jupyter Notebook: tensor_tutorial.ipynb](https://pytorch.org/tutorials/_downloads/tensor_tutorial.ipynb)
