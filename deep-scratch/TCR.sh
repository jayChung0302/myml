#!/bin/sh
(python memo.py && (git add . && git commit -am "TCR wip")) || git reset --hard 
