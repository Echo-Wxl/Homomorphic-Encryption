import paillier
import numpy as np


def test():
    public_key, private_key = paillier.generate_paillier_keypair()

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

    # 密文+密文
    encrypted_data2 = [public_key.encrypt(x) for x in data2]
    encrypteddata1_add_encrypteddata2 = np.add(encrypted_data1, encrypted_data2)

    decrypted_encrypted_data1_add_encrypted_data2 = [private_key.decrypt(x) for x in encrypteddata1_add_encrypteddata2]

    print("密文+密文：", decrypted_encrypted_data1_add_encrypted_data2)

    # 密文*明文
    encrypted_data1_multiply_data2 = np.multiply(encrypted_data1, data2)
    decrypted_encrypted_data1_multiply_data2 = [private_key.decrypt(x) for x in encrypted_data1_multiply_data2]
    print("密文*明文：", decrypted_encrypted_data1_multiply_data2)


if __name__ == '__main__':
    test()

