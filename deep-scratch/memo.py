import sys, os
import mypy
from datetime import datetime
import pickle

class Script:
    def __init__(self, name:str):
        self.content = ""
        self.hashtag = []
        self.name = name
        self.date = str(datetime.today()).split()[0]
        self.time = str(datetime.today()).split()[1].split('.')[0]
    
    def add_hashtag(self, name):
        self.hashtag.append(name)
    
    def del_hashtag(self, name):
        pass

class Memo:
    def __init__(self, memo_dir=None):
        self.memo = {}
        self.datelist = []
        self.hashlist = {}
        self.name = "./auto_saved_memo"
        # self.last_name = ""
        self.today = self.set_today()

        if memo_dir is not None:
            self.memo_dir = memo_dir
            self.load_memo(memo_dir)
        
    
    def load_memo(self, memo_dir):
        try:
            with open(memo_dir, 'rb') as file:
                self.memo = pickle.load(file)
                self.datelist = pickle.load(file)
                self.hashlist = pickle.load(file)

        except:
            raise FileNotFoundError(f'file doesn\'t exist in {self.memo_dir}')

    def set_today(self):
        today = str(datetime.today()).split()[0]
        if today in self.datelist:
            self.load_script(today)
        
        return today
    

    def total_save(self, save_dir="./"):
        with open(save_dir + self.name + '.pkl', 'wb') as file:
            pickle.dump(self.memo, file)
            pickle.dump(self.datelist, file)
            pickle.dump(self.hashlist, file)


    def load_script(self, load_dir="./"):
        pass

    def see(self, ):
        pass

if __name__ == '__main__':
    memo = Memo()
    memo.total_save()
    print()
