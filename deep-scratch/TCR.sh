#!/bin/sh
(python test.py && (git add . && git commit -am working)) || git reset --hard
