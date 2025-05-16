#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## GiAXp.py
##
##  Created on: July 7, 2024
##
##

    
## ./GiAXp.py -v -X abd -M  -x '5.4,3.0,4.5,1.5' ../tests/iris/iris_nbestim_10_maxdepth_2.mod.pkl   ../tests/iris/iris.csv

#==============================================================================
from __future__ import print_function
import os
import sys
import pickle
import resource


import numpy as np

#from itertools import combinations
from six.moves import range
#import six
import math

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

#from data import Data
from options import Options
from xrf import RFSklearn, Forest
from xrf import SATExplainer
from xrf import XRF, Dataset
from xrf import RFSklearn, RFBreiman


from grbxplainer import ILPExplainer

#==============================================================================
class INFXRF(XRF):
    """
        class to encode and verify global robustness of RFs.
    """
    
    def __init__(self, model, features, classes, verb=0):
        super(INFXRF, self).__init__(model, features, classes, verb)
        
    

    def explain(self, inst, xtype='abd', etype='sat', optimal=False, x_bounds=None):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """       
        
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
        
#         if etype == 'sat':
#             self.x = Explainer(self.enc, inps, preamble, self.class_names, x_bounds, verb=self.verbose)
#         else: 
        self.x = ILPExplainer(self.enc, inps, preamble, self.class_names, x_bounds, verb=self.verbose)
        
        expl = self.x.explain(np.array(inst), xtype, optimal)
            
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
                X_min, X_max = np.zeros(len(feature_names)), np.ones(len(feature_names))
                
                if options.verb or options.cat_data:
                    assert (len(options.files) == 2)
                    
                    print("loading data ...")
                    data = Dataset(filename=options.files[1], use_categorical=options.cat_data)
                    feature_names, targets = data.features, data.targets
                    
                    # print test accuracy of the RF model
                    _, X_test, _, y_test = data.train_test_split()
                    X_test = data.transform(X_test)
                    cls.print_accuracy(X_test, y_test)
                    # min, max data
                    X = data.transform(data.X)
                    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)                    
                    
                
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
            expl = xrf.explain(options.explain, options.xtype, options.etype, options.smallest, (X_min, X_max))
            print(f"expl len: {len(expl)}")
            
            del xrf.enc
            del xrf.x            
          
            