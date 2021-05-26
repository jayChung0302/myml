#!/bin/sh
(python test.py && (git add . && git commit -am "TCR working")) || git reset --hard
