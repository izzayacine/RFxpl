

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import sys
import os
import resource


from data import Data
from .tree import Forest

from .encode import SATEncoder, MXEncoder
from .explain import SATExplainer, MXExplainer



    

#
#==============================================================================
class Dataset(object):
    """
        Class for representing dataset (transactions).
    """
    def __init__(self, filename=None, from_data=None, 
                 fpointer=None, mapfile=None,
                separator=',', use_categorical=False):
        
        if filename or fpointer:
            from_data = Data(filename, fpointer, mapfile, separator, use_categorical)
        
        assert from_data 
        
        # split data into X and y
        self.features = from_data.names[:-1]
        self.nb_features = len(self.features)
        self.cat_data = use_categorical
        self.targets = None
        
        samples = np.asarray(from_data.samps)
        if not all(c.isnumeric() for c in samples[:, -1]):            
            le = LabelEncoder()
            le.fit(samples[:, -1])
            samples[:, -1]= le.transform(samples[:, -1])
            # self.class_names = le.classes_
            # self.target_name = le.transform(le.classes_)
            self.targets = le.classes_
            #print(le.classes_)
            #print(samples[1:4, :])
        
        samples = np.asarray(samples, dtype=np.float32)
        self.X = samples[:, 0: self.nb_features]
        self.y = samples[:, self.nb_features]
        self.num_class = len(set(self.y))
        if self.targets is None:
            self.targets = list(range(self.num_class))          
        
        print("c nof features: {0}".format(self.nb_features))
        print("c nof classes: {0}".format(self.num_class))
        print("c nof samples: {0}".format(len(samples)))
        
        # check if we have info about categorical features
        if (self.cat_data):
            self.categorical_features = from_data.categorical_features
            self.categorical_names = from_data.categorical_names             
            self.binarizer = {}
            for i in self.categorical_features:
                self.binarizer.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                self.binarizer[i].fit(self.X[:,[i]])
        else:
            self.categorical_features = []
            self.categorical_names = []            
            self.binarizer = []           
        #feat map
        self.mapping_features()        
        
    
    @property    
    def nClass(self):
        assert self.num_class == len(self.target)
        return sefl.num_class
    
    @property    
    def nFeatures_(self):
        # if not self.cat_data:
        #     assert len(feature_names) == len(self.ffeat_names)
        return len(self.extended_features)
    
    @property    
    def m_features_(self):
        if self.cat_data:
            return self.ffeat_names
        return self.features    
    
            
    def train_test_split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
           

    def transform(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.cat_data):
            assert(self.binarizer != [])
            tx = []
            for i in range(self.nb_features):
                #self.binarizer[i].drop = None
                if (i in self.categorical_features):
                    self.binarizer[i].drop = None
                    tx_aux = self.binarizer[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.cat_data):
            assert(self.binarizer != [])
            inverse_x = []
            for i, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.binarizer[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_features):
            return self.extended_features[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_features.values()):
            keys = (list(self.extended_features.keys())[list( self.extended_features.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_features = {}
        self.ffeat_names = []
        counter = 0
        if (self.cat_data):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.binarizer[i].categories_[0]):
                        self.extended_features.update({counter:  (self.features[i], j)})
                        self.ffeat_names.append("f{}_{}".format(i,j)) # str(self.features[i]), j))
                        counter = counter + 1
                else:
                    self.extended_features.update({counter: (self.features[i], None)})
                    self.ffeat_names.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_features.update({counter: (self.features[i], None)})
                self.ffeat_names.append("f{}".format(i))#(self.features[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)

      
#
#==============================================================================
class XRF(object):
    """
        class to encode and explain Random Forest classifiers.
    """
    
    def __init__(self, model, features, classes, verb=0):
        self.cls = model
        #self.data = dataset
        self.verbose = verb
        self.feature_names = features # data feature names
        self.class_names = classes
        self.ffnames = [f'f{i}' for i in range(len(features))]
        self.readable_data = lambda x: [str(i) for i in x]
        self.f = Forest(model.estimators(), self.ffnames)
        
        if self.verbose > 2:
            self.f.print_trees()
        if self.verbose:    
            print("c RF sz:", self.f.sz)
            print('c max-depth:', self.f.md)
            print('c nof DTs:', len(self.f.trees))
            
        
    def __del__(self):
        if 'enc' in dir(self):
            del self.enc
        if 'x' in dir(self):
            if self.x.slv is not None:
                self.x.slv.delete()
            del self.x
        del self.f
        self.f = None
        del self.cls
        self.cls = None
        
    def encode(self, inst, etype='maxsat'):
        """
            Encode a tree ensemble trained previously.
        """
        if 'f' not in dir(self):
            self.f = Forest(self.cls.estimators(), self.ffnames)
            #self.f.print_tree()
        elif self.f.attr_names != self.ffnames:
            self.f = Forest(self.cls.estimators(), self.ffnames)
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime            
        
        if etype == 'sat':    
            self.enc = SATEncoder(self.f, self.ffnames)
        else:
            assert (etype == 'maxsat') 
            self.enc = MXEncoder(self.f, self.ffnames)
        
        #inst = self.data.transform(np.array(inst))[0]
        self.enc.encode(np.array(inst))
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time        
        
        if self.verbose:
            print('c nof vars:', self.enc.nVars) # number of variables 
            print('c nof clauses:', self.enc.nClauses) # number of clauses    
            print('c encoding time: {0:.3f}'.format(time))            
        
    def explain(self, inst, xtype='abd', etype='maxsat', smallest=False):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime          
        
        if 'enc' not in dir(self):
            self.encode(inst, etype)
        
        inpvals = self.readable_data(inst)
        #inpvals = np.asarray(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
                    
        inps = self.ffnames # input (feature value) variables
        assert (self.f.attr_names == self.ffnames)
        #print("inps: {0}".format(inps))
        
        if etype == 'maxsat':
            self.x = MXExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        else:    
            self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        #inst = self.data.transform(np.array(inst))[0]
        expl = self.x.explain(np.array(inst), xtype, smallest)

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        
        if self.verbose:
            print("c Total time: {0:.3f}".format(time))
            
        return expl
    
    def enumerate(self, inst, xtype='con', etype='maxsat', smallest=True, xnum=100):
        """
            list all XPs
        """
        if 'enc' not in dir(self):
            self.encode(inst, etype)
            
        if 'x' not in dir(self):
            #inpvals = np.asarray(inst)
            inpvals = self.readable_data(inst)
            preamble = []
            for f, v in zip(self.feature_names, inpvals):
                if f not in str(v):
                    preamble.append('{0} = {1}'.format(f, v))
                else:
                    preamble.append(v)
                    
            inps = self.ffnames
            assert (self.f.attr_names == self.ffnames)
            
            if etype == 'sat':
                self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)             
            else: # maxsat
                self.x = MXExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
 
        if etype == 'sat':  
            # enumerate model (mx or adx)
            expls = []
            for expl in self.x.enumerate2(np.array(inst), xtype, smallest, xnum):
                yield expl
#             expls = self.x.enumerate(np.array(inst), xtype, smallest, xnum)
#             return expls        
        else:        
            # marco-based enum
            axps, cxps = self.x.enumerate(np.array(inst), xtype, False, xnum)
            if xtype == 'abd':
                return axps
            else:
                return cxps  
            
#         for expl in self.x.enumerate(np.array(inst), xtype, smallest):
#             yield expl
