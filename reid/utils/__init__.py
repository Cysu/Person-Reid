#!/usr/bin/python2
# -*- coding: utf-8 -*-

def cell(default, *args):
    result = [default for __ in xrange(args[0])]

    if len(args) > 1:
        for i in xrange(args[0]):
            result[i] = cell(default, *(args[1:]))

    return result