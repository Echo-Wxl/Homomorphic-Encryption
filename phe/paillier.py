from encoding import EncodedNumber
import utils
import random


def generate_paillier_keypair(n_length=1024):
    """return a new keypair :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.
    """
    p = q = n = None
    n_len = 0

    while n_len != n_length:
        p = utils.getprimeover(n_length // 2)
        q = p
        while q == p:
            q = utils.getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    return public_key, private_key


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.
    """

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        hashcode = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def apply_obfuscator(self, ciphertext, random_value=None):
        """ 
        """
        r = random_value or random.SystemRandom().randrange(1, self.n)
        obfuscator = utils.powmod(r, self.n, self.nsquare)

        return (ciphertext * obfuscator) % self.nsquare

    def raw_encrypt(self, plaintext, random_value=None):
        """
        """
        if not isinstance(plaintext, int):
            raise TypeError("plaintext should be int, but got: %s" %
                            type(plaintext))

        if plaintext >= (self.n - self.max_int) and plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            ciphertext = utils.invert(neg_ciphertext, self.nsquare)
        else:
            ciphertext = (self.n * plaintext + 1) % self.nsquare

        ciphertext = self.apply_obfuscator(ciphertext, random_value)

        return ciphertext

    def encrypt(self, value, precision=None, random_value=None):
        """Encode and Paillier encrypt a real number value.
        """
        encoding = EncodedNumber.encode(value, self.n, self.max_int, precision)
        obfuscator = random_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, random_value=obfuscator)
        encryptednumber = Paillier_EncryptedText(self, ciphertext, encoding.exponent)
        if random_value is None:
            encryptednumber.apply_obfuscator()

        return encryptednumber


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.
    """

    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError("given public key does not match the given p and q")
        if p == q:
            raise ValueError("p and q have to be different")
        self.public_key = public_key
        if q < p:
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p
        self.qsquare = self.q * self.q
        self.q_inverse = utils.invert(self.q, self.p)
        self.hp = self.h_func(self.p, self.psquare)
        self.hq = self.h_func(self.q, self.qsquare)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<PaillierPrivateKey {}>".format(hashcode[:10])

    def h_func(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page.
        """
        return utils.invert(self.l_func(utils.powmod(self.public_key.g,
                                                     x - 1, xsquare), x), x)

    def l_func(self, x, p):
        """computes the L function as defined in Paillier's paper.
        """

        return (x - 1) // p

    def crt(self, mp, mq):
        """the Chinese Remainder Theorem as needed for decryption. 
           return the solution modulo n=pq.
       """
        u = (mp - mq) * self.q_inverse % self.p
        x = (mq + (u * self.q)) % self.public_key.n

        return x

    def raw_decrypt(self, ciphertext):
        """return raw plaintext.
        """
        if not isinstance(ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" %
                            type(ciphertext))

        mp = self.l_func(utils.powmod(ciphertext,
                                      self.p - 1, self.psquare),
                         self.p) * self.hp % self.p

        mq = self.l_func(utils.powmod(ciphertext,
                                      self.q - 1, self.qsquare),
                         self.q) * self.hq % self.q

        return self.crt(mp, mq)

    def decrypt(self, encrypted_number):
        """return the decrypted & decoded plaintext of encrypted_number.
        """
        if not isinstance(encrypted_number, Paillier_EncryptedText):
            raise TypeError("encrypted_number should be an Paillier_EncryptedText, \
                             not: %s" % type(encrypted_number))

        if self.public_key != encrypted_number.public_key:
            raise ValueError("encrypted_number was encrypted against a different key!")

        encoded = self.raw_decrypt(encrypted_number.get_ciphertext(be_secure=False))
        encoded = EncodedNumber(encoded,
                                encrypted_number.exponent,
                                self.public_key.n,
                                self.public_key.max_int)
        decrypt_value = encoded.decode()

        return decrypt_value


class Paillier_EncryptedText(object):
    """Represents the Paillier encryption of a float or int.
    """

    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscator = False

        if not isinstance(self.ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" % type(self.ciphertext))

        if not isinstance(self.public_key, PaillierPublicKey):
            raise TypeError("public_key should be a PaillierPublicKey, not: %s" % type(self.public_key))

    def get_ciphertext(self, be_secure=True):
        """return the ciphertext of the Paillier_EncryptedText.
        """
        if be_secure and not self.__is_obfuscator:
            self.apply_obfuscator()

        return self.ciphertext

    def apply_obfuscator(self):
        """ciphertext by multiplying by r ** n with random r
        """
        self.__ciphertext = self.public_key.apply_obfuscator(self.ciphertext)
        self.__is_obfuscator = True

    def __add__(self, other):
        if isinstance(other, Paillier_EncryptedText):
            return self.__add_encryptednumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __mul__(self, scalar):
        """return Multiply by an scalar(such as int, float)
        """

        encode = EncodedNumber.encode(scalar, self.public_key.n, self.public_key.max_int)
        plaintext = encode.encoding

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError("Scalar out of bounds: %i" % plaintext)

        if plaintext >= self.public_key.n - self.public_key.max_int:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = utils.invert(self.get_ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            ciphertext = utils.powmod(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            ciphertext = utils.powmod(self.get_ciphertext(False), plaintext, self.public_key.nsquare)

        exponent = self.exponent + encode.exponent

        return Paillier_EncryptedText(self.public_key, ciphertext, exponent)

    def increase_exponent_to(self, new_exponent):
        """return Paillier_EncryptedText:
           new Paillier_EncryptedText with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError("New exponent %i should be great than old exponent %i" % (new_exponent, self.exponent))

        factor = pow(EncodedNumber.BASE, new_exponent - self.exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)

        return x, y

    def __add_scalar(self, scalar):
        """return Paillier_EncryptedText: z = E(x) + y
        """
        encoded = EncodedNumber.encode(scalar,
                                       self.public_key.n,
                                       self.public_key.max_int,
                                       max_exponent=self.exponent)

        return self.__add_fixpointnumber(encoded)

    def __add_fixpointnumber(self, encoded):
        """return Paillier_EncryptedText: z = E(x) + EncodedNumber(y)
        """
        if self.public_key.n != encoded.n:
            raise ValueError("Attempted to add numbers encoded against different public keys!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, encoded)

        encrypted_scalar = x.public_key.raw_encrypt(y.encoding, 1)
        encryptednumber = self.__raw_add(x.get_ciphertext(False), encrypted_scalar, x.exponent)

        return encryptednumber

    def __add_encryptednumber(self, other):
        """return Paillier_EncryptedText: z = E(x) + E(y)
        """
        if self.public_key != other.public_key:
            raise ValueError("add two numbers have different public key!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, other)

        encryptednumber = self.__raw_add(x.get_ciphertext(False), y.get_ciphertext(False), x.exponent)

        return encryptednumber

    def __raw_add(self, e_x, e_y, exponent):
        """return the integer E(x + y) given ints E(x) and E(y).
        """
        ciphertext = e_x * e_y % self.public_key.nsquare

        return Paillier_EncryptedText(self.public_key, ciphertext, exponent)
