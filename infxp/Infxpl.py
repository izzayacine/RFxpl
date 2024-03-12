#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## RFxpl.py
##
##  Created on: April 7, 2023
##      Author: Yacine Izza
##      E-mail: yacine.izza@gmail.com
##

#==============================================================================
from __future__ import print_function
import os
import sys
import pickle
import resource


import numpy as np

import collections
from itertools import combinations
from six.moves import range
import six
import math

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

#from data import Data
from options import Options
from xrf import RFSklearn, Forest
from xrf import SATEncoder, SATExplainer
from xrf import XRF, Dataset
from xrf import RFSklearn, RFBreiman

from pysat.solvers import Solver
from pysat.formula import CNF, WCNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.examples.lbx import LBX


                                   
#==============================================================================
class Explainer(SATExplainer):
    """
        Explainer (inflated XP) 
    """

    def __init__(self, encoding, inps, preamble, target_name, verb=1):
        """
            Initialiser.
        """
        super(Explainer, self).__init__(encoding, inps, preamble, target_name, verb)
        
    def compute_axp(self, smallest=False):
        """
            Compute an Abductive eXplanation
        """         
        self.assums = sorted(set(self.assums))
        if self.verbose:
            print('  # hypos:', len(self.assums))   
        
        #create a SAT solver
        self.slv = Solver(name="glucose3")
        
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)    
        
        # compute an inflated axp
        infx = self._mus()
        
        self.infx = infx.copy()

        expl = sorted([self.sel2fid[h] for h in self.assums if h>0 ]) # AXp
        assert len(expl)
        
        if self.verbose:
            print("expl-selctors: ", expl)
            preamble = []
            for i in infx:
                feat = f'f{i}'
                preamble.append([])
                if feat in self.enc.intvs:
                    dom = sorted([int(self.enc.nameVar(p).split('_')[1][4:]) for p in infx[i]])
                    assert (len(self.enc.intvs[feat]) - len(dom))
                    dom = [dom[0]]+[dom[j] for j in range(1, len(dom)) if (dom[j] - dom[j-1] > 1) or j==len(dom)-1 ]
                    for j in dom:
                        if j == 0:
                            preamble[-1].append(f'{feat}<={self.enc.intvs[feat][j]:.3f}')
                        elif len(self.enc.intvs[feat]) == 2 or j == len(self.enc.intvs[feat]) - 1:
                            # binary
                            preamble[-1].append(f'{feat}>{self.enc.intvs[feat][j-1]:.3f}')
                        else:
                            preamble[-1].append('{1:.3f}<{0}<={2:.3f}'.format(feat, self.enc.intvs[feat][j-1], 
                                                                              self.enc.intvs[feat][j]))
                else:
                    dom = [int(self.enc.nameVar(p).split('_')[1]) for p in infx[i]]
                    assert len(self.enc.categories[feat]) - len(dom)
                    for j in dom:
                        preamble[-1].append('{0}={1}'.format(feat,j))
                        
            print('  expl: "IF ({0}) THEN class={1}"'.format(') AND ('.join([' OR '.join(f) for f in preamble]), 
                                                          self.target_name[self.enc.target]))        
            #preamble = [self.preamble[i] for i in expl]
            #print('  explanation: "IF {0} THEN class={1}"'.format(' AND '.join(preamble), self.target_name[self.enc.target]))
            print('  # hypos left:', len(expl))
            
        return expl
    
    def coverage(self, infx):
        cov = math.prod([len(infx[i]) for i in infx])
        cov = cov / (math.prod([len(self.enc.categories['f{0}'.format(i)]) 
                              for i in infx if 'f{0}'.format(i) in self.enc.categories] +
                            [len(self.enc.intvs['f{0}'.format(i)]) for i in infx 
                             if 'f{0}'.format(i) in self.enc.intvs] ))
        
        return cov*100.0
    
    def cov_axp(self, expl):
        return 100.0 / (math.prod([len(self.enc.categories['f{0}'.format(i)]) 
                              for i in expl if 'f{0}'.format(i) in self.enc.categories] +
                            [len(self.enc.intvs['f{0}'.format(i)]) for i in expl 
                             if 'f{0}'.format(i) in self.enc.intvs] )) 
        
        
        
    def _mus(self):
        # simple deletion-based linear search for extracting inflated AXp
        for i, p in enumerate(self.assums):
            to_test = self.assums[:i] + self.assums[(i + 1):] + [-p, -self.sel2v[p]]
            sat = self.slv.solve(assumptions=to_test)
            if not sat:
                self.assums[i] = -p 
                
        xhypos = [p for p in self.assums if p>0] 
        infx =  {self.sel2fid[h]:[self.sel2v[h]] for h in xhypos}
        
        for i,h in enumerate(xhypos):
            
            if len(self.sel2vid[h]) > 1:
                # categorical data
                domain = self.enc.categories['f{0}'.format(self.sel2fid[h])] 
            else:
                assert (len(self.sel2vid[h]) == 1)
                # interval data
                domain = self.enc.ivars[self.inps[self.sel2vid[h][0]]]
            
            for q in domain:
                if q == self.sel2v[h]:
                    continue
                if len(infx[self.sel2fid[h]]) == len(domain) - 1:
                    continue
                to_test = xhypos[:i] + xhypos[i+1:] + [q]
                sat = self.slv.solve(assumptions=to_test)
                if not sat:    
                    infx[self.sel2fid[h]].append(q)
            
            if len(infx[self.sel2fid[h]]) > 1: # inflated
                xhypos[i] = self.enc.newVar('s_f{0}'.format(self.sel2fid[h]))
                cl = [-xhypos[i]] + infx[self.sel2fid[h]] # (x_i = v_i) or (x_i=v_j) or....(x_i=v_n) 
                self.slv.add_clause(cl)
                self.slv.append_formula([[-xhypos[i], -q] for q in domain 
                                         if (q not in infx[self.sel2fid[h]])])
                self.slv.add_clause([-h]) # deactivate selv_f_i
                    
        return infx
    
    def _mcs(self):
        """
            compute (minimal) CXp
        """
        wcnf = WCNF()
        for cl in self.enc.cnf:
            wcnf.append(cl)    
        for p in self.assums:
            wcnf.append([p], weight=1)
            
        # mcs solver
        self.slv = LBX(wcnf, use_cld=True, solver_name='g3')
        mcs = self.slv.compute()
        #expl = sorted([self.sel2fid[self.assums[i-1]] for i in mcs])
        
        xhypos = [self.assums[i-1] for i in mcs] 
        self.infx =  {self.sel2fid[h]:[] for h in xhypos}
        
        self.slv.delete() # rm LBX
        self.slv = Solver(name="glucose3") # create SAT slv
        self.slv.append_formula(self.enc.cnf)          
        
        for i,h in enumerate(xhypos):
            
            if len(self.sel2vid[h]) > 1:
                # categorical data
                domain = self.enc.categories['f{0}'.format(self.sel2fid[h])] 
            else:
                assert (len(self.sel2vid[h]) == 1)
                # interval data
                domain = self.enc.ivars[self.inps[self.sel2vid[h][0]]] 
                
            for q in domain:
                if q == self.sel2v[h]:
                    continue
                if len(self.infx[self.sel2fid[h]]) == len(domain) - 1:
                    continue
                to_test = [h for h in self.assums if (h not in xhypos)] + [q] 
                sat = self.slv.solve(assumptions=to_test)
                if sat:    
                    self.infx[self.sel2fid[h]].append(q) 
                    
        return [self.sel2fid[self.assums[i-1]] for i in mcs]    
        
#==============================================================================
class INFXRF(XRF):
    """
        class to encode and verify global robustness of RFs.
    """
    
    def __init__(self, model, features, classes, verb=0):
        super(INFXRF, self).__init__(model, features, classes, verb)
        
    

    def explain(self, inst, xtype='abd', etype='sat', smallest=False):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime          
        
        if 'enc' not in dir(self):
            self.encode(inst, etype)
        
        #inpvals = np.asarray(inst)
        inpvals = self.readable_data(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
                    
        inps = self.ffnames # input (feature value) variables
        #print("inps: {0}".format(inps))
        
        assert etype == 'sat'
        
        self.x = Explainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        
        if smallest:
            raise NotImplementedError()
            
        expl = self.x.explain(np.array(inst), xtype, smallest)

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        
        if self.verbose:
            print("c Total time: {0:.3f}".format(time))
            
        return expl

#
#==============================================================================
def show_info():
    """
        Print info message.
    """
    print("c RFxp: Random Forest explainer.")
    print('c')


    
#
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)
    
    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

        
        
    if options.files:
        cls = None
        xrf = None
            
        if options.train:
            print("loading data ...")
            data = Dataset(filename=options.files[0], separator=options.separator, 
                           use_categorical = options.cat_data)
            #data = Dataset(filename=options.files[0], mapfile=options.mapfile,
            #        separator=options.separator,
            #        use_categorical = options.use_categorical)
                    
            params = {'n_trees': options.n_estimators,
                        'depth': options.maxdepth}
            
            if options.algo == 'breiman':
                cls = RFBreiman(**params)
            else:
                cls = RFSklearn(**params)
                
            train_accuracy, test_accuracy = cls.train(data)
            
            if options.verb == 1:
                print("----------------------")
                print("Train accuracy: {0:.2f}".format(100. * train_accuracy))
                print("Test accuracy: {0:.2f}".format(100. * test_accuracy))
                print("----------------------")           
            
            xrf = XRF(cls, data.m_features_, data.targets, options.verb)
            #xrf.test_tree_ensemble()          
            
            bench_name = os.path.basename(options.files[0])
            assert (bench_name.endswith('.csv'))
            bench_name = os.path.splitext(bench_name)[0]
            bench_dir_name = os.path.join(options.output, bench_name)
            try:
                os.stat(bench_dir_name)
            except:
                os.makedirs(bench_dir_name)

            basename = (os.path.join(bench_dir_name, bench_name +
                            "_nbestim_" + str(options.n_estimators) +
                            "_maxdepth_" + str(options.maxdepth)))

            filename =  basename + '.mod.pkl'
            print("saving  model to ", filename)
            cls.save_model(filename)
            #filename =  basename + f'.{options.algo}.pkl'
            #pickle_save_file(filename, cls)            


        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]
            
            if not xrf:
                print("loading model ...")
                #cls = pickle_load_file(options.files[0])
                cls = RFSklearn(from_file=options.files[0])
                feature_names, targets = cls.feature_names, cls.targets
                
                if options.verb or options.cat_data:
                    assert (len(options.files) == 2)
                    
                    print("loading data ...")
                    data = Dataset(filename=options.files[1], use_categorical=options.cat_data)
                    feature_names, targets = data.features, data.targets
                    
                    # print test accuracy of the RF model
                    _, X_test, _, y_test = data.train_test_split()
                    X_test = data.transform(X_test) 
                    cls.print_accuracy(X_test, y_test) 
                    
                
                xrf = INFXRF(cls, feature_names, targets, options.verb)
                
            if options.cat_data:
                xrf.ffnames = data.m_features_
                xrf.readable_data = lambda x: data.readable_sample(data.transform_inverse(x)[0])
                options.explain = data.transform(np.array(options.explain))[0]
                
#             if options.xnum > 1:
#                 expls = xrf.enumerate(options.explain, options.xtype, options.etype,
#                                       options.smallest, options.xnum)
#                 expls = list(expls)
#                 print("nof expl:",len(expls))
#                 length = [len(x) for x in expls]
#                 print(f"avg: {sum(length)/len(expls):.1f}, min: {min(length)}, max: {max(length)}")
#             else:
            expl = xrf.explain(options.explain, options.xtype, options.etype, options.smallest)
            print(f"expl len: {len(expl)}")
            
            del xrf.enc
            del xrf.x            
          
            