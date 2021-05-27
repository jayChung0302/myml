#!/bin/sh
(python rope-tcr.py && (git add . && git commit -am "TCR wip")) || git reset --hard 
