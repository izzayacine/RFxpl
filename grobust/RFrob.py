#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## robust.py
##
##  Created on: Feb 23, 2023
##      Author: Yacine Izza
##      E-mail: yacine.izza@gmail.com
##

import sys
import os
import random
import resource
import getopt

import numpy as np

import collections
from itertools import combinations
from six.moves import range
import six
import math

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from xrf import RFSklearn, Forest
from xrf import SATEncoder
from pysat.solvers import Solver

from pysat.formula import CNF, WCNF, IDPool
from pysat.card import CardEnc, EncType


#==============================================================================
class RobustEncoder(SATEncoder):
    """
        Encoder for robustness verifier.
    """

    def __init__(self, forest, feature_names, from_file=None):
        """
            Initialiser.
        """
        super(RobustEncoder, self).__init__(forest, feature_names, None)
         
        
#     def duplicate_var(self, var):
#         # create a copy for encoding B
#         return self.newVar(f'{self.nameVar(abs(var))}_b')
    
    def duplicate(self, clause):
        sign = lambda a: (a>0) - (a<0)
        copy = []
        for lit in clause:
            assert self.nameVar(abs(lit)) is not None
            copy.append(sign(lit)*self.newVar(f'{self.nameVar(abs(lit))}_b'))
        return copy
    
                
    def encode_predict(self):
        """
           encode the majority votes: [kappa(x)≠kappa(r)] 
           using cardinality constraints
        """
        num_tree = len(self.forest.trees)
        
        pv1 = [[self.newVar(f'class{j}_tr{k}') for j in range(self.num_class)] for k in range(num_tree)]
        pv2 = [[self.newVar(f'class{j}_tr{k}_b') for j in range(self.num_class)] for k in range(num_tree)]
                
        
        if(self.num_class == 2):
            rhs = math.ceil(num_tree / 2)      
            lhs = [pv1[k][0] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            rhs = math.floor(num_tree / 2) + 1
            lhs = [pv2[k][1] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)            
            # c0 <=> -c1
            for k in range(num_tree):
                # copy A
                self.cnf.append([-pv1[k][0], -pv1[k][1]])
                self.cnf.append([pv1[k][0], pv1[k][1]])
                # copy B
                self.cnf.append([-pv2[k][0], -pv2[k][1]])
                self.cnf.append([pv2[k][0], pv2[k][1]])
        else:
            # create auxiliary vars
            bv = [self.newVar(f'b_{i}') for i in range(self.num_class)]
            bv2 = self.newVar(f'b_0_b') 
            wv = [self.newVar(f'w_{k}_b') for k in range (num_tree)]
            zv1 = [self.newVar(f'z_{k}_a') for k in range (num_tree)]
            zv2 = [self.newVar(f'z_{k}_b') for k in range (num_tree)]
            sv1 = [self.newVar(f's_{k}_a') for k in range(self.num_class)]
            sv2 = [self.newVar(f's_{k}_b') for k in range(self.num_class)]
            
            # encode majority class for copy A
            for i in range(self.num_class):
                lhs = zv1 + [-pv1[k][i] for k in range(num_tree)] + [bv[i]]
                rhs = num_tree + 1
                atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
                self.cnf.extend(atls)
            
            for i in range(self.num_class):
                self.cnf.extend([[-sv1[i], -bv[k]] for k in range(i)])
                self.cnf.extend([[-sv1[i], bv[k]] for k in range(i+1, self.num_class)])    
                
            # class copy A ≠ class copy B
            self.cnf.extend([ [-s]+sv2[:i]+sv2[i+1:] for i,s in enumerate(sv1)])
            self.cnf.extend([[-s, -sv2[i]] for i,s in enumerate(sv1)])
            
            
            # pick a class in copy A and activate it in copy B
            for j in range(self.num_class):
                self.cnf.extend([[-sv1[j], -zv1[i], pv1[i][j]] for i in range(num_tree)])
                self.cnf.extend([[-sv1[j], zv1[i], -pv1[i][j]] for i in range(num_tree)])
                self.cnf.extend([[-sv1[j], -wv[i], pv2[i][j]] for i in range(num_tree)])
                self.cnf.extend([[-sv1[j], wv[i], -pv2[i][j]] for i in range(num_tree)])
            

            # enforce copy B to pick a class different from copy A
            lhs = zv2 + [-w for w in wv] + [bv2]
            rhs = num_tree + 1
            atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)   
            # class order
            for i in range(self.num_class):
                self.cnf.extend([[-sv2[i], -sv1[k], -bv2] for k in range(i)])
                self.cnf.extend([[-sv2[i], -sv1[k], bv2] for k in range(i+1, self.num_class)])
                
            # pick a class in copy B
            for j in range(self.num_class):
                self.cnf.extend([[-sv2[j], -zv2[i], pv2[i][j]] for i in range(num_tree)])
                self.cnf.extend([[-sv2[j], zv2[i], -pv2[i][j]] for i in range(num_tree)])
                
            # class copy B ≠ class copy A
            self.cnf.extend([[-sv2[i]]+sv1[:i]+sv1[i+1:] for i in range(self.num_class)])
                
            # exactly1 class (copy A)
            eq1 = CardEnc.equals(lits=sv1, vpool=self.vpool, encoding=EncType.cardnetwrk) # bound = 1
            self.cnf.extend(eq1)
            # exactly1 class (copy B)
            eq1 = CardEnc.equals(lits=sv2, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eq1)            
            

    def encode(self, delta):
        """
            Do the job.
        """
        ###print('Encode RF into SAT ...')

        self.cnf = CNF()
        
        num_tree = len(self.forest.trees)

        # traverse all trees and extract all possible intervals
        # for each feature
        self.compute_intervals()
        
        # duplicate dicts: ivars and thvars
        self.ivars2 = {f: self.duplicate(self.ivars[f]) for f in self.ivars}
        self.thvars2 = {f: self.duplicate(self.thvars[f]) for f in self.thvars}
        
        
        ##print("encode trees ...")
        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            i = len(self.cnf.clauses)
            # encoding the tree     
            self.traverse(tree, k, [])
            # duplicate tree encoding and save as copy b
            self.cnf.extend([self.duplicate(cl) for cl in self.cnf.clauses[i:]])
            # exactly one class var is true in tree(k)
            leaves = [self.newVar(f'class{j}_tr{k}') for j in range(self.num_class)]
            am1 = CardEnc.atmost(lits=leaves, vpool=self.vpool,encoding=EncType.cardnetwrk)
            self.cnf.extend(am1)
            self.cnf.append(leaves)
            # copy b
            leaves = self.duplicate(leaves)
            am1 = CardEnc.atmost(lits=leaves, vpool=self.vpool,encoding=EncType.cardnetwrk) 
            self.cnf.extend(am1)            
            self.cnf.append(leaves)
        
        
        # encode the prediction \kappa(x) and exactly one class (leaf/path) 
        # per tree is true, i.e. \sum_j(c_j^k) = 1
        self.encode_predict()
        
        
#         # enforce exactly one of the feature values to be chosen
#         # (for categorical features)
#         categories = collections.defaultdict(lambda: [])
#         for f in self.feature_names:
#             if '_' in f:
#                 categories[f.split('_')[0]].append(self.newVar(f))        
#         for c, fvars in six.iteritems(categories):
#             # exactly-one feat is True
#             am1 = CardEnc.atmost(lits=fvars, vpool=self.vpool, encoding=EncType.cardnetwrk)
#             self.cnf.extend(am1)
#             self.cnf.append(fvars)
#             # duplicate categorical domain encoding
#             fvars = self.duplicate(fvars)
#             am1 = CardEnc.atmost(lits=fvars, vpool=self.vpool, encoding=EncType.cardnetwrk)
#             self.cnf.extend(am1)
#             self.cnf.append(fvars)
        
        
        i = len(self.cnf.clauses)
        
        # domain encoding
        self.encode_fdom()
        
        # duplicate continous domain encoding for copy b
        self.cnf.extend([self.duplicate(cl) for cl in self.cnf.clauses[i:]]) 
        
        # encode (x_i = r_i)
        nof_feats = len(set([f[1:].split('_')[0] for f in self.feature_names]))
        eqv = [self.newVar(f'eq_{i}') for i in range(nof_feats)]
        for f in self.categories:
            av2 = self.duplicate(self.categories[f])
            self.cnf.extend([[-a1, -a2, eqv[int(f[1:])]] for a1,a2 in zip(self.categories[f],av2)])
            self.cnf.extend([[-a1, a2, -eqv[int(f[1:])]] for a1,a2 in zip(self.categories[f],av2)])
        
        for f in self.ivars:
            # (a1 /\ a2) => eq
            # (a1 /\ -a2) => -eq
            self.cnf.extend([[-a1, -a2, eqv[int(f[1:])]] for a1,a2 in zip(self.ivars[f],self.ivars2[f])])
            self.cnf.extend([[-a1, a2, -eqv[int(f[1:])]] for a1,a2 in zip(self.ivars[f],self.ivars2[f])])
        
        # Hamming distance
        rhs = nof_feats - math.ceil(delta * nof_feats)
        atls = CardEnc.atleast(lits=eqv, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
        self.cnf.extend(atls)
        
        return self.cnf, self.intvs, self.imaps, self.ivars

#==============================================================================
class RRF(object):
    """
        class to encode and verify global robustness of RFs.
    """
    
    def __init__(self, model, feature_names, class_names, verb=0):
        self.cls = model
        self.verbose = verb
        self.feature_names = feature_names # data feature names
        self.class_names = class_names
        self.fnames = [f'f{i}' for i in range(len(feature_names))]
        self.f = Forest(model.estimators(), self.fnames)
        
        if self.verbose > 2:
            self.f.print_trees()
        if self.verbose:    
            print("c RF sz:", self.f.sz)
            print('c max-depth:', self.f.md)
            print('c nof DTs:', len(self.f.trees))
        
    def __del__(self):
        if 'enc' in dir(self):
            del self.enc
        if 'slv' in dir(self):
            if self.slv is not None:
                self.slv.delete()
            del self.slv
        del self.f
        self.f = None
        del self.cls
        self.cls = None
        
    def encode(self, delta, etype='sat'):
        """
            Encode a tree ensemble trained previously.
        """
        if 'f' not in dir(self):
            self.f = Forest(self.cls.estimators(), self.fnames)
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime            
        
        assert (etype == 'sat') 
        self.enc = RobustEncoder(self.f, self.fnames)
        
        self.enc.encode(delta)
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time        
        
        if self.verbose:
            print('c nof vars:', self.enc.nVars) # number of variables 
            print('c nof clauses:', self.enc.nClauses) # number of clauses    
            print('c encoding time: {0:.3f}'.format(time)) 
    
    
    def verify(self, delta, etype='sat'):
        """
            |x - r| ≤ delta
        """
        if 'enc' not in dir(self):
            self.encode(delta, etype)
        
        self.slv = Solver(name="glucose3")
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime  
        # pass the cnf formula
        self.slv.append_formula(self.enc.cnf)
        # call the sat oracle
        res = self.slv.solve()           
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time          
        if self.verbose:
            if res:
                hd = math.ceil(delta * len(self.fnames))
                print(f' model non-robust within delta={delta:.2f} (HD={hd})')
            else:
                print(f' model {delta:.2f}-robust')
            print(' time: {0:.3f}'.format(time)) 
            
        return (res == False)
    

#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'hd:o:v',
                                   ['help',
                                    'delta=',
                                    'oracle=',
                                    'verb'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    # init 
    verb = 0
    delta = 0.05 # (%)
    oracle = 'sat'   

    for opt, arg in opts:
        if opt in ('-d', '--delta'):
            delta = float(str(arg))                       
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-o', '--oracle'):
            oracle = str(arg)            
        elif opt in ('-v', '--verb'):
            verb += 1
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)


    return delta, oracle, verb, args    
#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] dataset neural-network')
    print('Options:')
    print('        -h, --help')
    print('        -d, --delta              Hamming distance delta |x - r|≤d (default = 0.05)')
    print('        -o, --oracle             Encoding to encode the RF')
    print('                                 Available values: mxs, sat. (default = sat)')    
    print('        -v, --verb               Be verbose (show comments)')
    
#
#==============================================================================

def main(delta, oracle, filename, verb=0):
    
    cls = RFSklearn(filename)
    feature_names, targets = cls.feature_names, cls.targets
    v = RRF(cls, feature_names, targets, verb)
    
    res = v.verify(delta) 
    enc_info = (v.enc.nVars, v.enc.nClauses, math.ceil(delta * len(v.fnames)))
    
    return res, enc_info

#
#==============================================================================
if __name__ == '__main__':
    
    delta, oracle, verb, files = parse_options() 
    if len(files) == 0:
        print('.pkl file is missing!')
        exit()  
    
    res, _ = main(delta, oracle, files[0], verb)
    print(res)