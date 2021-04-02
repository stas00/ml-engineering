#!/usr/bin/env python

# profile different implementations of the same function using 3 different profilers:
# `cProfile`, `line_profiler` and `timeit`
# to get a better indication of the actual speed. 3-ways as sometimes some of them may quite disagree.
#
# usage: call each of the following:
#
# ./speed_profile_3_ways.py -c
#
# ./speed_profile_3_ways.py -t
#
# pip install line_profiler
# kernprof -l speed_profile_3_ways.py -l; python -m line_profiler speed_profile_3_ways.py.lprof
#
# NB: I tried cProfile together with timeit and the latter was getting invalid results, so run each separately

import argparse

import math

# first always check that the various ways produce the same result
assert 2**2 == math.pow(2, 2)

# if randomness is used make sure to set the same seed before each test and that it depends on the RNG engine used (python vs. torch vs. numpy, etc.)

# if using cuda make sure to run
# gc.collect(); torch.cuda.empty_cache()
# after each test to reset things

# the ways being tested
def way1():
    for i in range(1000000):
        x = i**2

def way2():
    for i in range(1000000):
        x = math.pow(i, 2)

#### cProfile ####
# ./speed_profile_3_ways.py -c

import cProfile

def cprofileme():
    print("--------------- cProfile -----------------")
    cProfile.run("way1()", sort=-1)
    cProfile.run("way2()", sort=-1)

#### timeit ####
# ./speed_profile_3_ways.py -t
import timeit

def timeme():
    print("--------------- timeit -----------------")
    print(f'way1={timeit.Timer("way1()", globals=globals()).timeit(number=1)}')
    print(f'way2={timeit.Timer("way2()", globals=globals()).timeit(number=1)}')

#### line_profiler ####
# this one requires a special way to be called
# pip install line_profiler
# kernprof -l speed_profile_3_ways.py -l; python -m line_profiler speed_profile_3_ways.py.lprof

def line_profileme():
    print("--------------- line_profiler -----------------")
    profile(way1)()
    profile(way2)()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", action="store_true", help="use line_profiler")
    parser.add_argument("-c", action="store_true", help="use cProfile")
    parser.add_argument("-t", action="store_true", help="use timeit")
    args = parser.parse_args()
    if args.l  : line_profileme()
    elif args.c: cprofileme()
    elif args.t: timeme()
