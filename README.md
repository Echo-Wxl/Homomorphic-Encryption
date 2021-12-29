# 同态加密

## 一、同态加密的发展

同态加密(homomorphic encryption，HE)的概念是1978年由Rivest等人[^1]在题为《On data banks and privacy homomorphic》中首次提出的，允许用户直接对密文进行特定的代数运算，得到的数据仍是加密的结果，将结果解密后与明文进行同样计算步骤的结果一样。

在同态加密发展过程中先后有半同态加密，浅同态加密和全同态加密的提出。

1978年，Rivest等人[^2]利用数论构造出著名的公钥密码算法RSA，该算法安全性取决于大整数分解的困难性，具有乘法同态性，但不具备加法同态性。

Paillier于1999 年提出概率公钥加密系统，称为Paillier 同态加密[^3]。Paillier 加密是一种同态加密算法，其基于复合剩余类的困难问题，满足加法和数乘同态。

随后也有很多学者提出了基于不同理论的同态加密，但都不支持全同态加密。直到2009 年，Gentry[^4]构建了一个满足有限次同态计算的部分同态加密（Somewhat Homomorphic Encryption，SHE）算法，通过同态解密来实现密文的更新，达到全同态加密的效果，此项研究是基于理想格的全同态加密算法。

2009年至今，全同态加密技术发展很快，计算效率也不断提高，但是离大规模应用还有很长距离，目前在实际应用中更多还是采用半同态加密方案。

## 二、同态加密的定义

同态加密是一种允许在加密内容上进行特定代数运算的加密方案。大部分同态加密算法都是非对称加密，一般由4个函数组成：

- $(p k, s k) \leftarrow KeyGen(Params)$ : 密钥生成函数, 其 中 $p k$ 是对明文加密的公钥、 $s k$ 是对密文解密的私钥。
- $c \leftarrow E n c(p k, m):$ 加密函数，使用公钥 $p k$ 加密明文信息 $m$，得到密文 $c$ 。
- $m \leftarrow D e c(s k, c):$ 解密函数, 使用私钥 $s k$ 解密密文 $c$，得到明文 $m$ 。
- $c^{*} \leftarrow \operatorname{Eval}\left(c_{1}, c_{2}, \cdots, c_{n}\right):$ 密文计算函数, 在密文 $c_{1}, c_{2}, \cdots, c_{n}$ 上计算得到 $c^{*}$ 。密文 $c_{1}, c_{2}, \cdots, c_{n}$ 之间可以相互计算的前提是使用相同的密钥进行加密。

## 三、同态加密的分类

同态加密方法可以分为三类：部分同态加密（Partially Homomorphic Encryption，PHE），些许同态加密（Somewhat Homomorphic Encryption，SHE），全同态加密（Fully Homomorphic Encryption，FHE）。

### 1、部分同态加密（PHE）

PHE的特点是，要求其加密操作符运算只需要满足加密同态或者乘法同态中的一个即可，不需要两个同时满足。

**加法同态运算**

对于在明文空间M中的任意两个元素 $\mathrm{u}$ 和 $\mathrm{v}$, 其加密结果分别为 $[[\mathrm{u}]]$ 和 $[[\mathrm{v}]]$, 满足:
$$
\operatorname{Dec}_{\mathrm{sk}}([[u]]+[[v]])=\operatorname{Dec}_{\mathrm{sk}}([[u+v]])=u+v .
$$
即是加法同态加密。Paillier在1999年提出了一种可证的安全加法同态加密算法，该算法的具体过程如下：

1. 生成密钥

   随机生成两个大质数 $p$ 和 $q$，需满 足 $\operatorname{gcd}(p q,(p-1)(q-1))=1$ 。

   计算 $n=p q， \lambda=$ $\operatorname{lcm}(p-1, q-1)$，选取任意整数 $g \in \mathbb{Z}_{n^{2}}^{*}$，

   令 $\mu=\left(L\left(g^{\lambda} \bmod n^{2}\right)\right)^{-1}$ ，其中 $L(x)=\frac{x-1}{n}$ 。

   至此,生成公钥$p k=(n, g)$ 、私钥 $sk=(\lambda, \mu)$ 。

2. 加密过程

   对于任意明文 $m$，选取随机整数 $r \in \mathbb{Z}_{n^{2}}^{*}, 0<r<n$，满足 $\operatorname{gcd}(r, n)=1$​，则密文为
   $$
    c=Enc(m, r)=g^{m}r^{n} \bmod n^{2}(0<m<n)
   $$

3. 解密过程

   接收者对收到的密文 $c$ 进行解密，得到明文
   $$
   m=Dec(c)=L\left(c^{\lambda} \bmod n^{2}\right) \cdot \mu \bmod n
   $$

假定明文 $m_{1} 、 m_{2}$, 分别对其进行加密操作 $E\left(m_{1}\right)=g^{m_{1}} r_{1}^{N}$ $\bmod N^{2}$ 和 $E\left(m_{2}\right)=g^{m}{ }_{2} r_{2}^{N} \bmod N^{2}$, 得到
$$
E\left(m_{1}\right) \times E\left(m_{2}\right)=g^{m_{1}+m_{2}}\left(r_{1} r_{2}\right)^{N} \bmod N^{2}=E\left(m_{1}+m_{2}\right)
$$
由以上表达式可知，Paillier 公钥密码体制满足加法同态特性。

**乘法同态加密**

对于在明文空间 $\mathrm{M}$ 中的任意两个元素 $\mathrm{u}$ 和 $\mathrm{v}$, 其加密结果分别为 $[[\mathrm{u}]]$ 和 $[[\mathrm{v}]]$, 满足:
$$
\operatorname{Dec}_{\mathrm{sk}}([[u]] \times[[v]])=\operatorname{Dec}_{\mathrm{sk}}([[u \times v]])=u \times v
$$
即是乘法同态加密。RSA密码体制是第一个实用的公钥加密方案，于1978年由Rivest等人 提出，其安全性是基于整数分解问题的困难性。RSA算法分为以下几个步骤：

1. 生成密钥

   随机生成两个大质数 $p$ 和 $q$，$n=p q$，

   由欧拉定理有 $\varphi(n)=(p-1) \times(q-1)$，其中 $\varphi(n)$ 是 $n$ 的欧拉函数值。

   随机选择整数 $e$，满足 $1<e<\varphi(n)$，且 $\operatorname{gcd}(\varphi(n), e)=1$，

   由 $d \times e=1 \bmod (\varphi(n))$ 可计算出 $d$，

   则公钥 $p k=(n, e)$，私钥 $s k=d$ 。

2. 加密过程

   对于明文空间 $M$ 上的任意明文 $m$，加密得到密文
   $$
   c=Enc_{p k}(m)=m^{e}(\bmod n)
   $$

3. 解密过程

   对于任意密文 $c$，解密得到明文
   $$
   m=Dec_{s k}(c)=c^{d}(\bmod n)
   $$

假定明文 $m_{1} 、 m_{2}$, 使用 RSA 算法加密后得到 $Enc\left(m_{1}\right)=m_{1}$ $\bmod N, Enc\left(m_{2}\right)=m_{2}^{e} \bmod N$, 其中 $Enc\left(m_{1}\right) 、 Enc\left(m_{2}\right)$ 即为加密后的密文 $c_{1} 、 c_{2}$, 两者相乘得到
$$
Enc\left(m_{1}\right) \times Enc\left(m_{2}\right)=\left(m_{1}^{e} \times m_{2}^{e}\right) \bmod N
$$

$$
Enc\left(m_{1} m_{2}\right)=\left(m_{1}^{e} \times m_{2}^{e}\right) \bmod N
$$

所以$Enc\left(m_{1}\right) \times Enc\left(m_{2}\right)=Enc\left(m_{1} m_{2}\right)$，满足乘法同态性质。

### 2、些许同态加密（SHE）

SHE是指经过同台加密后的密文数据，在其上执行的操作（如加法、乘法等）只能是有限的次数。

SHE通过添加噪声的方式提高安全性。密文上的每一次操作都会增加密文上的噪声量，乘法操作是增长噪声量的主要方式。

当噪声量超过一个阈值后，解密函数就不能得到正确的结果了，所以绝大多数的SHE方案都要求限制计算次数。

### 3、全同态加密（FHE）

全同态加密算法允许对密文进行无限次的加法和乘法运算操作。

从2009年至今产生了许多全同态加密方案及实现与优化，第一代全同态加密方案都是遵循Gentry 复杂的构造方法。

本质上这些方案都是在各种环的理想上，首先构建一个些许(somewhat)同态加密方案(即方案只能执行低次多项式计算)。然后“压缩”解密电路(依赖稀疏子集和问题的假设)，从而执行自己的解密函数进行同态解密，达到控制密文噪声增长的目的，最终在循环安全的假设下获得全同态加密方案。尽管同态解密是实现全同态加密的基石，但是同态解密的效率很低，其复杂度为$\Omega\left(\lambda^{4}\right)$。

第二代全同态加密方案构造方法简单，基于错误学习（Learning With Errors， LWE）问题的假设 ，其安全性可以归约到一般格上的标准困难问题，打破了原有的Gentry构建全同态加密方案的框架。

接下来介绍一些基本概念：

**格**

格基密码学作为后量子密码的典型代表，是一类备受关注的抗量子计算攻击的公钥密码体制。

格(lattices) 给出一组线性无关的向量 $b_{1}, b_{2}, \cdots, b_{\mathrm{n}} \in \mathbb{R}^{\pi}$, 则格 $L$ 可以由 $b_{1}, b_{2}, \cdots$, $b_{\mathrm{n}}$ 的整系数线性组合生成, 定义为:
$$
L=\left\{z_{1} b_{1}+z_{2} b_{2}+, \cdots,+z_{n} b_{n}: z_{1}, z_{2}, \cdots, z_{n} \in \mathbb{Z}\right\} 。
$$
称 $b_{1}, b_{2}, \cdots, b_{\mathrm{n}}$ 是格 $L$ 的基。注意 $b_{1}, b_{2}, \cdots, b_{\mathrm{n}}$ 是 $\mathbb{R}$ 上的一组线性无关向量, 而不是 $\mathbb{Z}$ 上的。

**LWE问题**

给出一些关于秘密向量$s$的近似随机线性方程，其目标是恢复秘密向量$s$。例如给出吐下一些近似随机线性方程：
$$
\begin{array}{cc}
14 s_{1}+15 s_{2}+5 s_{3}+2 s_{4} \approx 8 & (\bmod 17) \\
13 s_{1}+14 s_{2}+14 s_{3}+6 s_{4} \approx 16 & (\bmod 17) \\
6 s_{1}+10 s_{2}+13 s_{3}+1 s_{4} \approx 3 & (\bmod 17) \\
10 s_{1}+4 s_{2}+12 s_{3}+16 s_{4} \approx 12 & (\bmod 17) \\
\ldots \ldots \ldots \ldots & \\
9 s_{1}+5 s_{2}+9 s_{3}+6 s_{4} \approx 9 & (\bmod 17)
\end{array}
$$

- 在上述每个方程中，加入了一个小的错误（噪声），该错误在+1和-1之间，目标是恢复向量$s$；
- 如果上述方程中没有加入错误，则使用高斯消元法就可以在多项式时间内恢复向量$s$；
- 但是由于加入了错误，使得该问题变得非常苦难 

用数学符号表达式为：
$$
(a,b=<a,s>+e \ mod \ q)
$$
**LWE问题的困难性**

第一，已知最好的求解LWE问题的算法运行时间是指数级的，即使是对量子计算机也没有任何帮助。

第二，LWE问题是LPN问题的自然扩展，而LPN问题在学习理论中被广泛研究而且普遍认为是困难的。此外LPN问题可以形式化为随机线性二元码的解码问题，如果LPN问题的求解算法有一点进步，则意味着编码理论的一个突破。

第三，也是最重要的，LWE问题被归约到最坏情况下的格上标准困难问题，例如GapSVP和SIVP。

一个非常简单的加密方案

- 加密: $b=\langle\boldsymbol{a}, s\rangle+2 e+m \bmod q$, 其中 $e$ 是小的,输出密文 $c=(a, b)$
- 解密: $[b-\langle\boldsymbol{a}, \boldsymbol{s}\rangle] \bmod q \bmod 2$

**全同态加密的两个关键问题：**

1. 获得同态性
   - 加法同态
   - 乘法同态

2. 控制噪音
   - Boostrapping技术
   - 模交换技术

 格密码天然具有加法同态和有限次的乘法同态：每做一个乘法，噪音增长，密文长度增长，密钥长度增长；

所以全同态加密有两个方向，一是层次同态加密，即满足多项式计算深度的电路即可；二是全同态加密，满足任意次乘法计算。

首先构建一个部分同态加密方案，密文计算后，用密钥交换技术控制密文向量的维数膨胀问题，然后使用模交换技术控制密文计算的噪声增长。

通过上述方法不需要同态解密技术，就可获得层次型全同态加密方案，即方案可以执行多项式级深度的电路，可以满足绝大多数应用。

要想获得“纯”的全同态加密方案，依然要依赖同态解密技术，然而同态解密技术效率低下，而且需要依赖循环安全的假设，实践中不予考虑。2013年Gentry等人 提出了一个基于近似特征向量的全同态加密方案，不需要密钥交换技术和模交换技术就可以实现层次型全同态加密方案。该方案的安全性基于LWE问题，密文的计算就是矩阵的加法与乘法，因此是非常自然的一个全同态加密方案。

**全同态加密主流方案的比较**

|              | 主流FHE方案                                                  | 高效Bootstrapping                                            | 浮点数上的FHE                                    |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| 方案         | BGV,FBV                                                      | [DM15]FHEW(GSW)<br />[CGGI16]TFHE                            | [CKKS17]HEAAN                                    |
| 明文数据类型 | 有限域（Packing)                                             | 二进制位串                                                   | 实/复数（Packing)                                |
| 计算         | 加法，乘法                                                   | 查询表/Bootstrapping                                         | 定点数的算术操作                                 |
| 算法库       | [HElib(IBM)](https://github.com/homenc/HElib)<br />[SEAL(Microsoft Research)](https://github.com/microsoft/SEAL)<br />Palisade(Duality） | [TFHE](https://github.com/tfhe/tfhe)<br />inpher,gemalto,etc. | [HEAAN(SNU)](https://github.com/snucrypto/HEAAN) |

### 4、阈值同态加密

前三节中的方法大多是单密钥算法，也就是只产生一个公钥和一个私钥。但是单密钥同态加密在实际应用中存在关于密钥使用和管理的问题，比如假设多方使用一套公私钥，虽然计算可以顺利进行，单系统安全性大大降低，系统中只要有一方被成功攻击，私钥就会泄露，另外也无法决定由哪一个参与方生成公私钥。多方联合计算最安全的途径是各自生成、保存公私钥，但由于算法限制，不同公钥加密的信息无法相互计算，导致隐私计算无法进行。

为了解决上述问题，有学者提出阈值同态加密，也就是多密钥同态加密。阈值同态加密算法存在多个私钥，一个（或多个）公钥，使用该公钥系统加密的密文之间可以相互计算，并且只有当参与解密的私钥数量达到一定阈值时，才能成功解密密文。

阈值同态加密算法同样可以概括为4个函数：

- $(p k, s k, e k) \leftarrow Keygen $($ Params )$ ：密钥生成函 数, 其中 $p k$ 是公钥、sk 是私钥、ek 是用于计算的密钥。
- $c \leftarrow \operatorname{Enc}(p k, m):$ 加密函数, 使用公钥 $p k$ 加密明文信息 $m$,得到密文 $c$ 。
- $m \leftarrow \operatorname{Dec}\left(c, s k_{1}, s k_{2}, \cdots, s k_{k}\right):$ 解密函数,最少 $k$ 个私钥参与, 才能解密得到明文。
- $c \leftarrow \operatorname{Eval}\left(\left(c_{1}, p k_{1}, e k_{1}\right),\left(c_{2}, p k_{2}, e k_{2}\right), \cdots,\left(c_{N},\right.\right.$, $\left.p k_{N}, e k_{N}\right)$ ) : 密文计算函数,在多个密文上进行计算、获得最终结果，计算过程需要计算密钥 $e k$ 参与。

现有的阈值同态加密技术大多基于单密钥全同态加密算法改进而来，不同阈值同态加密算法在细节上有所不同，但加密的计算模式一致，即支持多个私钥，不同私钥下密文可以相互计算，解密需要多个私钥参与。

阈值同态加密和普通单密钥同态加密的最大不同是: 算法支持多个私钥，并且不同公钥加密的密文可以通过计算密钥$ek $进行转换，以支持相互计算; 假设共有N 组私钥对应的密文参与了计算，则解密中最少需要K 组私钥才能解密成功，其中$K≤N$，K 的值在不同算法中不同，大部分全同态阈值算法中$K =N$。

## 四、python实现Paillier同态加密

### 1、Paillier

目前有很多同态加密的开源算法库，如微软的[SEAL](https://github.com/microsoft/SEAL)，随后陈智罡博士团队开发了一个将SEAL全同态加密库映射到Python上的接口[PySEAL](https://github.com/Huelse/SEAL-Python)，解决了机器学习库与全同态加密库对接的问题，还有IBM的同台密码开源库[HElib](https://github.com/homenc/HElib)等，两者都是实现了BFV和CKKS方案。

本文也实现了python版本的paillier同态加密算法，实现代码请见[phe](https://github.com/Echo-Wxl/Homomorphic-Encryption/tree/main/phe)

以下测试paillier同态加密的性质：

- 密文与密文相加
- 密文与明文相加
- 密文与明文相乘

```python
def paillier_he():
    public_key, private_key = paillier.generate_paillier_keypair(n_length=32)

    # 加密
    original_list = [x for x in range(10)]
    encrypted_list = [public_key.encrypt(x) for x in original_list]

    # 解密
    decrypted_list = [private_key.decrypt(x) for x in encrypted_list]
    print(decrypted_list)

    
    # 密文+明文
    data1 = [2, 3, 4]
    data2 = [5, 6, 7]

    encrypted_data1 = [public_key.encrypt(x) for x in data1]
    encrypted_data1_add_data2 = np.add(encrypted_data1, data2)
    
    decrypted_encrypted_data1_add_data2 = [private_key.decrypt(x) for x in encrypted_data1_add_data2]
    print("密文+明文：", decrypted_encrypted_data1_add_data2)
    # 输出： 密文+明文： [7, 9, 11]

    # 密文+密文
    encrypted_data2 = [public_key.encrypt(x) for x in data2]
    encrypteddata1_add_encrypteddata2 = np.add(encrypted_data1, encrypted_data2)

    decrypted_encrypted_data1_add_encrypted_data2 = [private_key.decrypt(x) for x in encrypteddata1_add_encrypteddata2]

    print("密文+密文：", decrypted_encrypted_data1_add_encrypted_data2)
    # 输出： 密文+密文： [7, 9, 11]

    # 密文*明文
    encrypted_data1_multiply_data2 = np.multiply(encrypted_data1, data2)
    decrypted_encrypted_data1_multiply_data2 = [private_key.decrypt(x) for x in encrypted_data1_multiply_data2]
    print("密文*明文：", decrypted_encrypted_data1_multiply_data2)
    # 输出： 密文*明文： [10, 18, 28]
```

### 2、phe在联邦学习中的应用

在联邦学习中，不同参与方训练出的模型参数可由一个第三方进行统一聚合。

使用加法PHE，可以在明文数据不出域、且不泄露参数的情况下，完成对模型参数的更新。

```python
"""
This example involves learning using sensitive medical data from multiple hospitals
to predict diabetes progression in patients. The data is a standard dataset from
sklearn[1].

Recorded variables are:
- age,
- gender,
- body mass index,
- average blood pressure,
- and six blood serum measurements.

The target variable is a quantitative measure of the disease progression.
Since this measure is continuous, we solve the problem using linear regression.

The patients' data is split between 3 hospitals, all sharing the same features
but different entities. We refer to this scenario as horizontally partitioned.

The objective is to make use of the whole (virtual) training set to improve
upon the model that can be trained locally at each hospital.

50 patients will be kept as a test set and not used for training.

An additional agent is the 'server' who facilitates the information exchange
among the hospitals under the following privacy constraints:

1) The individual patient's record at each hospital cannot leave the premises,
   not even in encrypted form.
2) Information derived (read: gradients) from any hospital's dataset
   cannot be shared, unless it is first encrypted.
3) None of the parties (hospitals AND server) should be able to infer WHERE
   (in which hospital) a patient in the training set has been treated.

Note that we do not protect from inferring IF a particular patient's data
has been used during learning. Differential privacy could be used on top of
our protocol for addressing the problem. For simplicity, we do not discuss
it in this example.

In this example linear regression is solved by gradient descent. The server
creates a paillier public/private keypair and does not share the private key.
The hospital clients are given the public key. The protocol works as follows.
Until convergence: hospital 1 computes its gradient, encrypts it and sends it
to hospital 2; hospital 2 computes its gradient, encrypts and sums it to
hospital 1's; hospital 3 does the same and passes the overall sum to the
server. The server obtains the gradient of the whole (virtual) training set;
decrypts it and sends the gradient back - in the clear - to every client.
The clients then update their respective local models.

From the learning viewpoint, notice that we are NOT assuming that each
hospital sees an unbiased sample from the same patients' distribution:
hospitals could be geographically very distant or serve a diverse population.
We simulate this condition by sampling patients NOT uniformly at random,
but in a biased fashion.
The test set is instead an unbiased sample from the overall distribution.

From the security viewpoint, we consider all parties to be "honest but curious".
Even by seeing the aggregated gradient in the clear, no participant can pinpoint
where patients' data originated. This is true if this RING protocol is run by
at least 3 clients, which prevents reconstruction of each others' gradients
by simple difference.

This example was inspired by Google's work on secure protocols for federated
learning[2].

[1]: http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
[2]: https://research.googleblog.com/2017/04/federated-learning-collaborative.html

Dependencies: numpy, sklearn
"""

import numpy as np
from sklearn.datasets import load_diabetes

import phe as paillier

seed = 43
np.random.seed(seed)


def get_data(n_clients):
    """
    Import the dataset via sklearn, shuffle and split train/test.
    Return training, target lists for `n_clients` and a holdout test set
    """
    print("Loading data")
    diabetes = load_diabetes()
    y = diabetes.target
    X = diabetes.data
    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple clients.
    # The selection is not at random. We simulate the fact that each client
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test


def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)


def encrypt_vector(public_key, x):
    return [public_key.encrypt(i) for i in x]


def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]


class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self, key_length):
         keypair = paillier.generate_paillier_keypair(n_length=key_length)
         self.pubkey, self.privkey = keypair

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients


class Client:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.

    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient

    def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

    def predict(self, X):
        """Score test data"""
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        """Compute and encrypt gradient.

        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        if sum_to is not None:
            return sum_encrypted_vectors(sum_to, encrypted_gradient)
        else:
            return encrypted_gradient


def federated_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    n_iter = config['n_iter']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the server and generate private and public keys
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    server = Server(key_length=config['key_length'])

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], server.pubkey))

    # The federated learning with gradient descent
    print('Running distributed gradient aggregation for {:d} iterations'
          .format(n_iter))
    for i in range(n_iter):

        # Compute gradients, encrypt and aggregate
        encrypt_aggr = clients[0].encrypted_gradient(sum_to=None)
        for c in clients[1:]:
            encrypt_aggr = c.encrypted_gradient(sum_to=encrypt_aggr)

        # Send aggregate to server and decrypt it
        aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)

        # Take gradient steps
        for c in clients:
            c.gradient_step(aggr, config['eta'])

    print('Error (MSE) that each client gets after running the protocol:')
    for c in clients:
        y_pred = c.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


def local_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], None))

    # Each client trains a linear regressor on its own data
    print('Error (MSE) that each client gets on test set by '
          'training only on own local data:')
    for c in clients:
        c.fit(config['n_iter'], config['eta'])
        y_pred = c.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


if __name__ == '__main__':
    config = {
        'n_clients': 5,
        'key_length': 1024,
        'n_iter': 50,
        'eta': 1.5,
    }
    # load data, train/test split and split training data between clients
    X, y, X_test, y_test = get_data(n_clients=config['n_clients'])
    # first each hospital learns a model on its respective dataset for comparison.
    local_learning(X, y, X_test, y_test, config)
    # and now the full glory of federated learning
    federated_learning(X, y, X_test, y_test, config)

```

### 3、phe在利用逻辑回归模型判断垃圾邮件中的应用

在这个例子中，Alice 用她拥有的一些数据进行逻辑回归训练垃圾邮件分类器。

训练完成后，她使用 Paillier 同态加密算法生成公钥/私钥对，使用公钥将模型加密。 

然后加密模型发送给 Bob， Bob 利用加密模型测试自己的数据，获得每封电子邮件的加密分数。

 Bob 将它们发送给 Alice，Alice 用私钥解密它们以获得预测垃圾邮件与非垃圾邮件。

```python
"""
In this example Alice trains a spam classifier on some e-mails dataset she
owns. She wants to apply it to Bob's personal e-mails, without

1) asking Bob to send his e-mails anywhere
2) leaking information about the learned model or the dataset she has learned
from
3) letting Bob know which of his e-mails are spam or not.

Alice trains a spam classifier with logistic regression on some data she
possesses. After learning, she generates public/private key pair with a
Paillier schema. The model is encrypted with the public key. The public key and
the encrypted model are sent to Bob. Bob applies the encrypted model to his own
data, obtaining encrypted scores for each e-mail. Bob sends them to Alice.
Alice decrypts them with the private key to obtain the predictions spam vs. not
spam.

Example inspired by @iamtrask blog post:
https://iamtrask.github.io/2017/06/05/homomorphic-surveillance/
"""

import time
import os.path
from zipfile import ZipFile
from urllib.request import urlopen
from contextlib import contextmanager

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import phe as paillier

np.random.seed(42)

# Enron spam dataset hosted by https://cloudstor.aarnet.edu.au
url = [
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/RpHZ57z2E3BTiSQ/download',
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/QVD4Xk5Cz3UVYLp/download'
]


def download_data():
    """Download two sets of Enron1 spam/ham e-mails if they are not here
    We will use the first as trainset and the second as testset.
    Return the path prefix to us to load the data from disk."""

    n_datasets = 2
    for d in range(1, n_datasets + 1):
        if not os.path.isdir('enron%d' % d):

            URL = url[d-1]
            print("Downloading %d/%d: %s" % (d, n_datasets, URL))
            folderzip = 'enron%d.zip' % d

            with urlopen(URL) as remotedata:
                with open(folderzip, 'wb') as z:
                    z.write(remotedata.read())

            with ZipFile(folderzip) as z:
                z.extractall()
            os.remove(folderzip)


def preprocess_data():
    """
    Get the Enron e-mails from disk.
    Represent them as bag-of-words.
    Shuffle and split train/test.
    """

    print("Importing dataset from disk...")
    path = 'enron1/ham/'
    ham1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron1/spam/'
    spam1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/ham/'
    ham2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/spam/'
    spam2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]

    # Merge and create labels
    emails = ham1 + spam1 + ham2 + spam2
    y = np.array([-1] * len(ham1) + [1] * len(spam1) +
                 [-1] * len(ham2) + [1] * len(spam2))

    # Words count, keep only frequent words
    count_vect = CountVectorizer(decode_error='replace', stop_words='english',
                                 min_df=0.001)
    X = count_vect.fit_transform(emails)

    print('Vocabulary size: %d' % X.shape[1])

    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Split train and test
    split = 500
    X_train, X_test = X[-split:, :], X[:-split, :]
    y_train, y_test = y[-split:], y[:-split]

    print("Labels in trainset are {:.2f} spam : {:.2f} ham".format(
        np.mean(y_train == 1), np.mean(y_train == -1)))

    return X_train, y_train, X_test, y_test


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


class Alice:
    """
    Trains a Logistic Regression model on plaintext data,
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def encrypt_weights(self):
        coef = self.model.coef_[0, :]
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights, encrypted_intercept

    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Bob:
    """
    Is given the encrypted model and the public key.

    Scores local plaintext data with the encrypted model, but cannot decrypt
    the scores without the private key held by Alice.
    """

    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        """Compute the score of `x` by multiplying with the encrypted model,
        which is a vector of `paillier.EncryptedNumber`"""
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]


if __name__ == '__main__':

    download_data()
    X, y, X_test, y_test = preprocess_data()

    print("Alice: Generating paillier keypair")
    alice = Alice()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    alice.generate_paillier_keypair(n_length=1024)

    print("Alice: Learning spam classifier")
    with timer() as t:
        alice.fit(X, y)

    print("Classify with model in the clear -- "
          "what Alice would get having Bob's data locally")
    with timer() as t:
        error = np.mean(alice.predict(X_test) != y_test)
    print("Error {:.3f}".format(error))

    print("Alice: Encrypting classifier")
    with timer() as t:
        encrypted_weights, encrypted_intercept = alice.encrypt_weights()

    print("Bob: Scoring with encrypted classifier")
    bob = Bob(alice.pubkey)
    bob.set_weights(encrypted_weights, encrypted_intercept)
    with timer() as t:
        encrypted_scores = bob.encrypted_evaluate(X_test)

    print("Alice: Decrypting Bob's scores")
    with timer() as t:
        scores = alice.decrypt_scores(encrypted_scores)
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f} -- this is not known to Alice, who does not possess "
          "the ground truth labels".format(error))

```

## 参考文献

1. R. L. Rivest, L. Adleman, M. L. Dertouzos. On data banks and privacy homomorphisms[J].Foundations of Secure Computation, 1978, 11: 169-180
2. R. L. Rivest, A. Shamir, L. Adleman. A method for obtaining digital signatures andpublic-key cryptosystems[J]. Communications of the ACM, 1978, 21(2): 120-126
3. PAILLIEＲ P． Public-key cryptosystems based on compositedegree residuosity classes［C］/ /Proceedings of InternationalConference on the Theory and Applications of CryptographicTechniques． Berlin，Germany: Springer， 1999:223-238．
4. C. Gentry. Fully homomorphic encryption using ideal lattices[D].Palo Alto: Stanford University, 2009, 169-178
5. 陈智罡，王箭. 全同态密码研究. 计算机应用研究. 2014.06 
6. 杨强，黄安埠，刘洋等. 联邦学习实战[M]. 电子工业出版社，2021.



