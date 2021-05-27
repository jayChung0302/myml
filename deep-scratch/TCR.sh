#!/bin/sh
(python test.py && (git add . && git commit -am "TCR wip")) || git reset --hard
