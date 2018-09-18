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

초기화하지 않은 5x3 Matrix를 선언하면 다음과 같습니다.

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


---

또는, 이미 존재하는 Tensor를 기반으로 만들 수 있습니다. 이 방법은 이미 존재하는 Tensor의 특성을 다시 사용합니다. ()
