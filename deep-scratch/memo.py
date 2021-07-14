import sys, os
import mypy
from typing import List, Dict, Any
from datetime import datetime
import pickle

class Script:
    def __init__(self, name:str):
        self.content = ""
        self.hashtag = []
        self.name = name
        self.date = str(datetime.today()).split()[0]
        self.time = str(datetime.today()).split()[1].split('.')[0]
        self.content = self.content + ", " + self.date

    def add_hashtag(self, name:Any):
        if isinstance(name, str):
            self.hashtag.append(name)
        elif isinstance(name, List):
            self.hashtag += name
    
    def del_hashtag(self, name):
        pass

class Memo:
    def __init__(self, memo_dir=None):
        self.memo = {}
        self.datelist = {}
        self.hashlist = {}
        self.name = "./auto_saved_memo"
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
            raise FileNotFoundError(f'File doesn\'t exist in {self.memo_dir}')

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

    def put_script(self, script: Script):
        if isinstance(script, Script):
            if script.name in self.memo.keys():
                print('Same script file name already exist.')
                self.memo[script.name] = self.memo[script.name] + \
                    f'\nAuto saved in {script.date}, {script.time} \n' + script.content + '\n'
                print('Auto saved')
            else:
                self.memo[script.name] = script.content + '\n'

            for hashname in script.hashtag:
                if hashname in self.hashlist.keys():
                    self.hashlist[hashname].append(script.name)
                else:
                    self.hashlist[hashname] = [script.name]

            if script.date not in self.datelist.keys():
                self.datelist[script.date] = [script.name]
            else:
                self.datelist[script.date].append(script.name)
        else:
            raise ValueError(f'{script} is not a Script instance')

    def show_memo(self):
        print(self.memo.keys())

    def load_script(self, load_dir="./"):
        pass

    def read_script(self, script_name):
        print(self.memo[script_name])

if __name__ == '__main__':
    # try:
    #     memo = Memo()
    #     script = Script('0714_example')
    #     script.content = '메모장을 만들어 봅시다 헤헤'
    #     script.hashtag = ['ss', 'ww']
    #     memo.put_script(script)
    #     memo.total_save()
    #     memo.show_memo()
    #     memo.read_script('0714_example')
    #     print(memo.datelist)
    #     print(memo.hashlist)
    try:
        script = Script('0714_hi')
        script.content = 'hello world..!'
        script.hashtag = ['ss', 'kk']
        memo = Memo()
        memo.load_memo('./auto_saved_memo.pkl')

        memo.put_script(script)
        memo.total_save()
        memo.show_memo()
        print(memo.datelist)
        print(memo.hashlist)
        memo.read_script('0714_example')
        print(memo.memo)
        print(memo.read_script('0714_hi'))
        
    except:
        raise KeyError
