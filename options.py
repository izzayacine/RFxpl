#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##
##  Created on: Oct 08, 2020
##      Author: Yacine Izza
##      E-mail: yacine.izza@gmail.com
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
import sys


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        """
            Constructor.
        """

        # actions
        self.train = False
        self.encode = 'none'
        self.explain = ''
        
        # explaining options
        self.xtype = 'abd'
        self.etype = 'sat'
        self.smallest = False
        
        # enum expl
        self.xnum = 1
        

        # training options
        self.algo = 'breiman' # ['skl', 'breiman']
        self.n_estimators = 100
        self.maxdepth = 3
        self.testsplit = 0.2
        self.seed = 7
        self.cat_data = False

        # other options
        self.files = None
        self.output = 'Classifiers'
        self.mapfile = None
        self.separator = ',' 
        self.solver = 'g3'
        self.relax = 3
        self.verb = 0

        
        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """

        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    'a:e:hcd:Mn:o:s:tvx:X:E:N:',
                                    ['algo=','encode=', 'help', 'cat-data=',
                                     'maxdepth=', 'minimum', 'nbestims=',
                                     'output=', 'seed=', 'solver=', 'testsplit=',
                                     'train', 'verbose', 'explain=', 'xtype=', 
                                     'etype=', 'xnum=' ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-a', '--algo'):
                self.algo = str(arg)
                assert (self.algo in ['skl', 'breiman'])
            elif opt in ('-c', '--cat-data'):
                self.cat_data = True
            elif opt in ('-d', '--maxdepth'):
                self.maxdepth = int(arg)
            elif opt in ('-e', '--encode'):
                self.encode = str(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)

            elif opt in ('-M', '--minimum'):
                self.smallest = True
            elif opt in ('-n', '--nbestims'):
                self.n_estimators = int(arg)
            elif opt in ('-o', '--output'):
                self.output = str(arg)
    
            elif opt == '--seed':
                self.seed = int(arg)
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt == '--testsplit':
                self.testsplit = float(arg)
            elif opt in ('-t', '--train'):
                self.train = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-x', '--explain'):
                self.explain = str(arg)
            elif opt in ('-X', '--xtype'):
                self.xtype = str(arg)
            elif opt in ('-E', '--etype'):
                self.etype = str(arg)
            elif opt in ('-N', '--xnum'):
                self.xnum = str(arg)
                self.xnum = -1 if self.xnum == 'all' else int(self.xnum)               
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        if self.encode == 'none':
            self.encode = None

        self.files = args

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] input-file')
        print('Options:')
        #print('        -a, --accmin=<float>       Minimal accuracy')
        #print('                                   Available values: [0.0, 1.0] (default = 0.95)')
        print('        -c, --cat-data             Treat categorical features as categorical')
        print('                                   (with categorical features info if available)')
        print('        -d, --maxdepth=<int>       Maximal depth of a tree')
        print('                                   Available values: [1, INT_MAX] (default = 3)')
        #print('        -e, --encode=<smt>         Encode a previously trained model')
        print('        -E, --etype=<sat>          Type of encoding: SAT-Enc for RFmv and  MxSAT-Enc for RFmw')
        print('                                   Available values: sat, maxsat (default = sat)')        
        print('        -h, --help                 Show this message')
  
        #print('        -m, --map-file=<string>    Path to a file containing a mapping to original feature values. (default: none)')
        print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        -n, --nbestims=<int>       Number of trees in the ensemble')
        print('                                   Available values: [1, INT_MAX] (default = 100)')
        print('        -N, --xnum=<int>           Number of explanations to compute')
        print('                                   Available values: [1, INT_MAX], all (default = 1)')
        print('        -o, --output=<string>      Directory where output files will be stored (default: \'temp\')')
        print('        --seed=<int>               Seed for random splitting')
        print('                                   Available values: [1, INT_MAX] (default = 7)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        print('        -s, --solver=<string>      A SAT oracle to use')
        print('                                   Available values: glucose3, minisat (default = g3)')
        print('        -t, --train                Train a model of a given dataset')
        print('        --testsplit=<float>        Training and test sets split')
        print('                                   Available values: [0.0, 1.0] (default = 0.2)')
        print('        -v, --verbose              Increase verbosity level')
        print('        -x, --explain=<string>     Explain a decision for a given comma-separated sample (default: none)')
        print('        -X, --xtype=<string>       Type of explanation to compute: abductive or contrastive')
