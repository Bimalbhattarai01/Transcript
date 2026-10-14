"""Password hashing for the team accounts: stdlib scrypt, no dependencies."""
import hashlib, hmac, os

N, R, P = 2 ** 14, 8, 1
MIN_LEN = 10


def hash_password(pw):
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=N, r=R, p=P, dklen=32)
    return f"scrypt${N}${R}${P}${salt.hex()}${h.hex()}"


def verify_password(pw, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(h) // 2)
        return hmac.compare_digest(got.hex(), h)
    except Exception:
        return False


DUMMY = hash_password("not-a-real-password")  # verified when the username doesn't exist, so timing doesn't reveal valid names
