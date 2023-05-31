#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## RFxpl.py
##
##  Created on: Oct 08, 2020
##      Author: Yacine Izza
##      E-mail: yacine.izza@gmail.com
##

#
#==============================================================================
from __future__ import print_function
from data import Data
from options import Options
import os
import sys
import pickle
import resource

import numpy as np

from xrf import XRF, Dataset
from xrf import RFBreiman, RFSklearn



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
def pickle_save_file(filename, data):
    try:
        f =  open(filename, "wb")
        pickle.dump(data, f)
        f.close()
    except:
        print("Cannot save to file", filename)
        exit()

def pickle_load_file(filename):
    try:
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()    


    
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
                    
                
                xrf = XRF(cls, feature_names, targets, options.verb)
                
            if options.cat_data:
                xrf.ffnames = data.m_features_
                xrf.readable_sample = lambda x: data.readable_sample(data.transform_inverse(x)[0])
                options.explain = data.transform(np.array(options.explain))[0]
                
            if options.xnum > 1:
                expls = xrf.enumerate(options.explain, options.xtype, options.etype,
                                      options.smallest, options.xnum)
                expls = list(expls)
                print("nof expl:",len(expls))
                length = [len(x) for x in expls]
                print(f"avg: {sum(length)/len(expls):.1f}, min: {min(length)}, max: {max(length)}")
            else:
                expl = xrf.explain(options.explain, options.xtype, options.etype, options.smallest)
                print(f"expl len: {len(expl)}")
            
            del xrf.enc
            del xrf.x            
          
            