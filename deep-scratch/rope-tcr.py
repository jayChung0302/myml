# Rope data structure
# add very large string, how can you insert single character into the middle of it in constant time?

# todo
# insert
# delete
# substring
# concatenation

# API
def to_rope(string):
    return String(string)

class Rope:
    def substring(self, start, length):
        return Substring(self, start, length)

    def concatenate(self, right):
        return Concatenation(self, right)
class String(Rope):
    def __init__(self, string):
        self.string = string
    
    def  __str__(self):
        return self.string
        
class Substring(Rope):
    def __init__(self, rope, start, length):
        self.rope = rope
        self.start = start
        self.length = length

    def __str__(self):
        return str(self.rope)[self.start :  self.start + self.length]

class Concatenation(Rope):
    def __init__(self, rope, right):
        self.rope = rope
        self.right = right

    def __str__(self):
        return self.rope.string + self.right.string

assert str(to_rope("ABC")) == "ABC"
assert str(to_rope("ABCDE").substring(1, 3)) == "BCD"
assert str(to_rope("ABCDE").substring(1, 3).substring(1,1)) == "C"
assert str(to_rope("ABC").concatenate(to_rope("DE"))) == "ABCDE"
