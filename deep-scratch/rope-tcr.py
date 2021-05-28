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
    def delete(self, start, length):
        return Deletion(self, start, length)
    
    def insert(self, rope, start):
        left = self[0 : start] # AB
        right = self[start : len(self)]
        return left + rope + right
    
    def __len__(self):
        raise Exception("should have been overriden")

    def __add__(self, addend):
        return Concatenation(self, addend)
    
    def __getitem__(self, index):
        if type(index) == int:
            return self.__get_single_item__(index)
        return Substring(self, index.start, index.stop - index.start)
    
    def __get_single_item__(self, index):
        raise Exception("should have been overriden")

class String(Rope):
    def __init__(self, string):
        self.string = string
    
    def  __str__(self):
        return self.string
    
    def __len__(self):
        return len(self.string)
    
    def __get_single_item__(self, index):
        return self.string[index]
        
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
    
    def __get_single_item__(self, index):
        return "D"

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
equals(to_rope("ABCDE")[1:4], "BCD")
equals(to_rope("ABCDE")[1:4][1:2], "C")
equals(to_rope("ABC") + to_rope("DE"), "ABCDE")
equals(to_rope("ABCDE").delete(1, 3), "AE")
equals(len(to_rope("ABCDE")[1:4]), "3")
equals(len(to_rope("ABC") + to_rope("DE")), "5")
equals(to_rope("ABE").insert(to_rope("CD"), 2), "ABCDE")

equals(to_rope("ABCDE")[3], "D")
# equals((to_rope("ABC") + to_rope("DE"))[3], "D")
