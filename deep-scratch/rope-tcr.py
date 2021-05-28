# Rope data structure
# add very large string, how can you insert single character into the middle of it in constant time?

#TODO
# insert
# delete # done
# substring # done
# concatenation # done

# API
def to_rope(string):
    return String(string)

class Rope:
    def substring(self, start, length):
        return Substring(self, start, length)

    def concatenate(self, right):
        return Concatenation(self, right)
    
    def delete(self, start, length):
        return Deletion(self, start, length)

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
    def __init__(self, left, right): # left: rope
        self.left = left
        self.right = right

    def __str__(self):
        return self.left.string + self.right.string

class Deletion(Rope):
    def __init__(self, rope, start, length):
        self.rope = rope
        self.start = start
        self.length = length

    def __str__(self):
        return str(self.rope)[:self.start] + str(self.rope)[self.start + self.length:]

# Testing Framework
def equals(rope, expected):
    actual = str(rope)
    if actual == expected:
        return 
    print(f"{actual} didn't equal {expected}")
    raise Exception()

equals(to_rope("ABC"), "ABC")
equals(to_rope("ABCDE").substring(1,3), "BCD")
equals(to_rope("ABCDE").substring(1,3).substring(1,1), "C")
equals(to_rope("ABC").concatenate(to_rope("DE")), "ABCDE")
equals(to_rope("ABCDE").delete(1, 3), "AE")
