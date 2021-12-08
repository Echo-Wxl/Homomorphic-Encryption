# 同态加密

## 一、同态加密的发展

同态加密(homomorphic encryption，HE)的概念是1978年由Rivest等人[^1]在题为《On data banks and privacy homomorphic》中首次提出的，允许用户直接对密文进行特定的代数运算，得到的数据仍是加密的结果，将结果解密后与明文进行同样计算步骤的结果一样。

在同态加密发展过程中先后有半同态加密，浅同态加密和全同态加密的提出。

1978年，Rivest等人[^2]利用数论构造出著名的公钥密码算法RSA，该算法安全性取决于大整数分解的困难性，具有乘法同态性，但不具备加法同态性。

Paillier于1999 年提出概率公钥加密系统，称为Paillier 同态加密[^3]。Paillier 加密是一种同态加密算法，其基于复合剩余类的困难问题，满足加法和数乘同态。

随后也有很多学者提出了基于不同理论的同态加密，但都不支持全同态加密。直到2009 年，Gentry[^4]构建了一个满足有限次同态计算的部分同态加密（Somewhat Homomorphic Encryption，SHE）算法，通过同态解密来实现密文的更新，达到全同态加密的效果，此项研究是基于理想格的全同态加密算法。

2009年至今，全同态加密技术发展很快，尽管全同态加密方案的效率不断提高，但全同态加密的构造方法并没有大的突破，离实际应用依然有距离。在实际应用中更多还是采用半同态加密方案。

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
\operatorname{Dec}_{\mathrm{sk}}([[u]]+[[v]])=\operatorname{Dec}_{\mathrm{sk}}([[u+v]])=u+v
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

首先构建一个部分同态加密方案，密文计算后，用密钥交换技术控制密文向量的维数膨胀问题，然后使用模交换技术控制密文计算的噪声增长。通过上述方法不需要同态解密技术，就可获得层次型全同态加密方案，即方案可以执行多项式级深度的电路，可以满足绝大多数应用。要想获得“纯”的全同态加密方案，依然要依赖同态解密技术，然而同态解密技术效率低下，而且需要依赖循环安全的假设，实践中不予考虑。2013年Gentry等人 提出了一个基于近似特征向量的全同态加密方案，不需要密钥交换技术和模交换技术就可以实现层次型全同态加密方案。该方案的安全l生基于LWE问题，密文的计算就是矩阵的加法与乘法，因此是非常自然的一个全同态加密方案。

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

目前有很多同态加密的开源算法库，如微软的[SEAL](https://github.com/microsoft/SEAL)，随后陈智罡博士团队开发了一个将SEAL全同态加密库映射到Python上的接口[PySEAL](https://github.com/Huelse/SEAL-Python)，解决了机器学习库与全同态加密库对接的问题，还有IBM的同台密码开源库[HElib](https://github.com/homenc/HElib)等，两者都是实现了BFV和CKKS方案。

本文也实现了python版本的paillier同态加密算法，实现代码请见[phe](./phe)

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

## 参考文献

[1^]: R. L. Rivest, L. Adleman, M. L. Dertouzos. On data banks and privacy homomorphisms[J].Foundations of Secure Computation, 1978, 11: 169-180

[^2]: R. L. Rivest, A. Shamir, L. Adleman. A method for obtaining digital signatures andpublic-key cryptosystems[J]. Communications of the ACM, 1978, 21(2): 120-126
[^3]: PAILLIEＲ P． Public-key cryptosystems based on compositedegree residuosity classes［C］/ /Proceedings of InternationalConference on the Theory and Applications of CryptographicTechniques． Berlin，Germany: Springer， 1999:223-238．
[^4]: C. Gentry. Fully homomorphic encryption using ideal lattices[D].Palo Alto: Stanford University, 2009, 169-178
[^5]: 陈智罡，王箭. 全同态密码研究. 计算机应用研究. 2014.06 
[^6]: 杨强，黄安埠，刘洋等. 联邦学习实战[M]. 电子工业出版社，2021.



