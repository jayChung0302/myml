# Rope data structure
# add very large string, how can you insert single character into the middle of it in constant time?

#TODO
# [] notation
# reorder
# + for concatenation
# len() function # done
# insert # done
# delete # done
# substring # done
# concatenation # done

# API
def to_rope(string):
    return String(string)

class Rope:
    def substring(self, start, length):
        return Substring(self, start, length)

    def delete(self, start, length):
        return Deletion(self, start, length)
    
    def insert(self, rope, start):
        left = self.substring(0, start) # AB
        right = self.substring(start, len(self) - start)
        return left + rope + right
    
    def __len__(self):
        raise Exception("should have been overriden")

    def __add__(self, addend):
        return Concatenation(self, addend)
    
    def __getitem__(self, slice):
        print(slice)
        return self.substring(slice.start, slice.stop - slice.start)

class String(Rope):
    def __init__(self, string):
        self.string = string
    
    def  __str__(self):
        return self.string
    
    def __len__(self):
        return len(self.string)
        
class Substring(Rope):
    def __init__(self, rope, start, leng):
        self.rope = rope
        self.start = start
        self.leng = leng

    def __str__(self):
        return str(self.rope)[self.start :  self.start + self.leng]
    
    def __len__(self):
        return self.leng


class Concatenation(Rope):
    def __init__(self, left, right): # left: rope
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.left) + str(self.right)
    
    def __len__(self):
        return len(self.left) + len(self.right)

class Deletion(Rope):
    def __init__(self, rope, start, leng):
        self.rope = rope
        self.start = start
        self.leng = leng

    def __str__(self):
        return str(self.rope)[:self.start] + str(self.rope)[self.start + self.leng:]

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
equals(to_rope("ABC") + to_rope("DE"), "ABCDE")
equals(to_rope("ABCDE").delete(1, 3), "AE")
equals(len(to_rope("ABCDE").substring(1,3)), "3")
equals(len(to_rope("ABC") + to_rope("DE")), "5")
equals(to_rope("ABE").insert(to_rope("CD"), 2), "ABCDE")

print(to_rope("ABCDE")[1:2])
