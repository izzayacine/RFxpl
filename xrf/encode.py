

import collections
from decimal import Decimal
from itertools import combinations
from six.moves import range
import six
import math

import resource

#from data import Data
#from .forest import RF2001, VotingRF
from .tree import Forest

from pysat.formula import CNF, WCNF, IDPool
from pysat.card import CardEnc, EncType

from .mxreason import ClassEnc


#
#==============================================================================
class SATEncoder(object):
    """
        Encoder of Random Forest classifier into SAT.
    """
    
    def __init__(self, forest, feature_names, nof_classes=None, from_file=None):
        self.forest = forest
        self.num_class = forest.n_classes
        self.vpool = IDPool()
        self.feature_names = feature_names # on the form f_i or f_i_j (OHE)
        
        #encoding formula
        self.cnf = None

        # for interval-based encoding
        self.intvs, self.imaps, self.ivars, self.thvars = None, None, None, None
       
        
    def newVar(self, name):
        """
            If a variable named 'name' already exists then
            return its id; otherwise create a new var
        """
        if name in self.vpool.obj2id: #var has been already created 
            return self.vpool.obj2id[name]
        var = self.vpool.id('{0}'.format(name))
        return var
    
    def nameVar(self, vid):
        """
            input a var id and return a var name
        """
        return self.vpool.obj(abs(vid))
    
    def printLits(self, lits):
        print(["{0}{1}".format("-" if p<0 else "",self.vpool.obj(abs(p))) for p in lits])
        
    @property    
    def nVars(self):
        return self.vpool.top
    
    @property
    def nClauses(self):
        if self.cnf is None:
            return 0
        return len(self.cnf.clauses)
    
    
    def traverse(self, tree, k, clause):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            f = tree.name
            v = tree.threshold
            pos = neg = []
            if f in self.intvs:
                d = self.imaps[f][v]
                pos, neg = self.thvars[f][d], -self.thvars[f][d]
            else:
                var = self.newVar(tree.name)
                pos, neg = var, -var
                #print("{0} => {1}".format(tree.name, var))
                
            assert (pos and neg)
            self.traverse(tree.children[0], k, clause + [-neg])
            self.traverse(tree.children[1], k, clause + [-pos])            
        else:  # leaf node
            cvar = self.newVar('class{0}_tr{1}'.format(tree.label,k))
            self.cnf.append(clause + [cvar])
            #self.printLits(clause + [cvar])

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """

        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """

            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    self.intvs[f].add(v)

                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])

        # initializing the intervals
        self.intvs = {'{0}'.format(f): set([]) for f in self.feature_names if '_' not in f}

        for tree in self.forest.trees:
            traverse_intervals(tree)
                
        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in six.iterkeys(self.intvs)}

        self.imaps, self.ivars = {}, {}
        self.thvars = {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            self.thvars[feat] = []
            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i

                ivar = self.newVar('{0}_intv{1}'.format(feat, i))
                self.ivars[feat].append(ivar)
                
                if ub != math.inf:
                    #assert(i < len(intvs)-1)
                    thvar = self.newVar('{0}_th{1}'.format(feat, i))
                    self.thvars[feat].append(thvar)

                    
    def encode_predict(self, sample):
        """
           encode the majority votes: [kappa(x)≠c] 
           using cardinality constraints to capture a j^th class
           such than c_j≠c
        """
        vtaut = self.newVar('Tautology')
        num_tree = len(self.forest.trees)
        
        ctvars = [[self.newVar(f'class{j}_tr{k}') for j in range(self.num_class)] for k in range(num_tree)]
        
        for k in range(num_tree):
            # exactly one class var is true
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool, encoding=EncType.cardnetwrk) 
            self.cnf.extend(card.clauses)            
        
        # calculate the majority class   
        self.target = self.forest.predict(sample)        
        
        if(self.num_class == 2):
            rhs = math.floor(num_tree / 2) + 1
            if(self.target==1 and not num_tree%2):
                rhs = math.floor(num_tree / 2)      
            lhs = [ctvars[k][1 - self.target] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            
        else:
            zvars = []
            zvars.append([self.newVar('z_0_{0}'.format(k)) for k in range (num_tree) ])
            zvars.append([self.newVar('z_1_{0}'.format(k)) for k in range (num_tree) ])
            ##
            rhs = num_tree
            lhs0 = zvars[0] + [ - ctvars[k][self.target] for k in range(num_tree)]
            ##self.printLits(lhs0)
            atls = CardEnc.atleast(lits = lhs0, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            ##
            #rhs = num_tree - 1
            rhs = num_tree + 1
            ###########
            lhs1 =  zvars[1] + [ - ctvars[k][self.target] for k in range(num_tree)]
            ##self.printLits(lhs1)
            atls = CardEnc.atleast(lits = lhs1, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)            
            #
            pvars = [self.newVar('p_{0}'.format(k)) for k in range(self.num_class + 1)]
            ##self.printLits(pvars)
            for k,p in enumerate(pvars):
                for i in range(num_tree):
                    if k == 0:
                        z = zvars[0][i]
                        #self.cnf.append([-p, -z, vtaut])
                        self.cnf.append([-p, z, -vtaut])       
                        #self.printLits([-p, z, -vtaut])
                        #print()
                    elif k == self.target+1:
                        z = zvars[1][i]
                        self.cnf.append([-p, z, -vtaut])       
                        
                        #self.printLits([-p, z, -vtaut])
                        #print()                       
                        
                    else:
                        z = zvars[0][i] if (k<self.target+1) else zvars[1][i]
                        self.cnf.append([-p, -z, ctvars[i][k-1] ])
                        self.cnf.append([-p, z, -ctvars[i][k-1] ])  
                        
                        #self.printLits([-p, -z, ctvars[i][k-1] ])
                        #self.printLits([-p, z, -ctvars[i][k-1] ])
                        #print()                      
            #
            self.cnf.append([-pvars[0], -pvars[self.target+1]])
            ##
            lhs1 =  pvars[:(self.target+1)]
            ##self.printLits(lhs1)
            eqls = CardEnc.equals(lits = lhs1, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)            
            
            lhs2 = pvars[(self.target + 1):]
            ##self.printLits(lhs2)
            eqls = CardEnc.equals(lits = lhs2, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)
        
        
    def encode_fdom(self):
        """
            encode feature domains
        """
        
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        self.categories = collections.defaultdict(lambda: [])
        for f in self.feature_names:
            if '_' in f:
                self.categories[f.split('_')[0]].append(self.newVar(f))        
        for c, feats in six.iteritems(self.categories):
            # exactly-one feat is True
            self.cnf.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
            
        # lits of intervals   
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            # enforcing that there is at least one interval 
            # am1 interval is expressed in dom+nodes encoding (below)
            self.cnf.append(intvs) 
            #self.printLits(intvs)                    
        
        for f, threshold in six.iteritems(self.thvars):
            for j, thvar in enumerate(threshold):
                d = j+1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d] 
                
                if j == 0:
                    assert(len(neg) == 1)
                    self.cnf.append([thvar, neg[-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                else:
                    self.cnf.append([thvar, neg[-1], -threshold[j-1]])
                    self.cnf.append([-thvar, threshold[j-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                
                if j == len(threshold) - 1:
                    assert(len(pos) == 1)
                    self.cnf.append([-thvar, pos[0]])
                    self.cnf.append([thvar, -pos[0]])
                else:
                    self.cnf.append([-thvar, pos[0], threshold[j+1]])
                    self.cnf.append([thvar, -pos[0]])
                    self.cnf.append([thvar, -threshold[j+1]]) 
                    
        

    def encode(self, sample):
        """
            Do the job.
        """
        ###print('Encode RF into SAT ...')

        self.cnf = CNF()
        
        num_tree = len(self.forest.trees)
        # define Tautology var
        vtaut = self.newVar('Tautology')
        self.cnf.append([vtaut])

        # traverse all trees and extract all possible intervals
        # for each feature
        self.compute_intervals()
        
        #print(self.intvs)
        #print([len(self.intvs[f]) for f in self.intvs])
        #print(self.imaps) 
        #print(self.ivars)
        #print(self.thvars)
        
        
        ##print("encode trees ...")
        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            # encoding the tree     
            self.traverse(tree, k, [])
            # exactly one class var is true
            #card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool,encoding=EncType.cardnetwrk) 
            #self.cnf.extend(card.clauses)
        
        
        # encodode the prediction \kappa(x) and exactly one class (leaf/path) 
        # per tree is true, i.e. \sum_j(c_j^k) = 1
        self.encode_predict(sample)
        
            
        # domain encoding    
        self.encode_fdom()          

        
        return self.cnf, self.intvs, self.imaps, self.ivars


#
#==============================================================================
class MXEncoder(SATEncoder):
    """
        Encoder for the MaxSAT-based reasoner.
    """

    def __init__(self, forest, feature_names, from_file=None, relax=3):
        """
            Initialiser.
        """

        super(MXEncoder, self).__init__(forest, feature_names, None)
        self.relax = relax # floating point precision
        self.leaves = None
        self.soft = None

        
#     @property
#     def nClauses(self):
#         if self.cnf is None:
#             return 0
#         nof_cl = len(self.cnf.clauses)
#         nof_cl += sum([len(wcnf.hard)+len(wcnf.soft) for _,wcnf in self.soft.items()])
#         return nof_cl
    
    
#     def traverse2(self, tree, k, clause):
#         """
#             Traverse a tree and encode each node.
#         """

#         if tree.children:
#             f = tree.name
#             v = tree.threshold
#             #print(tree.id)
#             #print(tree.parent)
#             pos = neg = []
#             if f in self.intvs:
#                 d = self.imaps[f][v]
#                 pos, neg = self.thvars[f][d], -self.thvars[f][d]
#             else:
#                 var = self.newVar(tree.name)
#                 pos, neg = var, -var
#             assert (pos and neg)
#             if tree.parent is None:
#                 # root node
#                 #v = self.newVar(f'a{tree.id}')
#                 self.cnf.append([self.newVar(f'a_tr{k}_{tree.id}')])
#                 #self.printLits([self.newVar(f'a_tr{k}_{tree.id}')])
#             else:
#                 a1 = self.newVar(f'a_tr{k}_{tree.id}')
#                 a0 = self.newVar(f'a_tr{k}_{tree.parent.id}')
#                 self.cnf.append([-a0, clause[-1], a1])
#                 self.cnf.append([-a1, a0])
#                 self.cnf.append([-a1, -clause[-1]])
#                 #
#                 #self.printLits([-a0, clause[-1], a1])
                
#             self.traverse(tree.children[0], k, clause + [-neg])
#             self.traverse(tree.children[1], k, clause + [-pos])            
#         else:  # leaf node
#             leaf = self.newVar(f'class{tree.label}_tr{k}_{tree.id}')
#             a0 = self.newVar(f'a_tr{k}_{tree.parent.id}')
#             self.cnf.append([-a0, clause[-1], leaf]) #  a_k,O_j,m <=> t_r             
#             self.cnf.append([-leaf, a0])
#             self.cnf.append([-leaf, -clause[-1]])

    def traverse(self, tree, k, clause):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            f = tree.name
            v = tree.threshold
            pos = neg = []
            if f in self.intvs:
                d = self.imaps[f][v]
                pos, neg = self.thvars[f][d], -self.thvars[f][d]
            else:
                var = self.newVar(tree.name)
                pos, neg = var, -var
                #print("{0} => {1}".format(tree.name, var))
                
            assert (pos and neg)
            self.traverse(tree.children[0], k, clause + [-neg])
            self.traverse(tree.children[1], k, clause + [-pos])            
        else:  # leaf node
            leaf = self.newVar(f'tr{k}_{tree.id}')
            self.cnf.append(clause + [leaf]) #  O_1,k,...,O_j,m => t_r  
            for v in clause:
                self.cnf.append([-v, -leaf]) # t_r => O_1,k, ..., t_r => O_j,m
    
            
                
    def encode_predict(self, sample):
        """
            Use MaxSAT to capture the predicted class
        """
        def traverse_leaves(node, k, lvars):
            if node.children:
                traverse_leaves(node.children[0], k, lvars)
                traverse_leaves(node.children[1], k, lvars)
            else:
                leaf = self.newVar(f'tr{k}_{node.id}')
                for label, proba in enumerate(node.proba): 
                    wght = round(Decimal(str(proba)), self.relax)
                    self.leaves[label].append((leaf, wght))
                lvars.append(leaf)
                #self.printLits([leaf])
           
        
        # calculate the majority class: argmax proba   
        self.target = self.forest.predict_proba(sample)
        
        # all leaves to be used in the formula and am1 constraints
        self.leaves, atmosts = collections.defaultdict(lambda: []), [] 
        # computes leaves and exactly one class (leaf) constraint
        for k, tree in enumerate(self.forest.trees):
            am1 = []
            traverse_leaves(tree, k, am1)
            atmosts.append(am1)
            # exactly one terminal-node/path is true
            card = CardEnc.atmost(lits=am1, vpool=self.vpool, encoding=EncType.cardnetwrk) 
            self.cnf.extend(card.clauses)
            self.cnf.append(am1)
            #self.printLits(am1)
        
        
        # encode the soft clauses
        if not self.soft:
            del self.soft
        self.soft = collections.defaultdict(lambda: WCNF())
        
        for clid in range(self.num_class):
            if clid == self.target:
                continue
            # weight and cost of the current & target class
            wghts, cost = collections.defaultdict(lambda: 0), 0
            for (lit, w1), (lit2, w2) in zip(self.leaves[clid], self.leaves[self.target]):
                assert (lit == lit2)
                wghts[lit] = w1 - w2
            
            # flipping literals with negative weights
            lits = list(wghts.keys())
            for l in lits:
                if wghts[l] < 0:
                    cost += -wghts[l]
                    wghts[-l] = -wghts[l]
                    del wghts[l]    

            # filtering out those with zero-weights
            wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))

            # maximum value of the objective function
            self.soft[clid].vmax = sum(wghts.values())

            # processing all AtMost1 constraints
            # to improve cores extraction in the MaxSAT solver   
            atmosts_clid = set([tuple([l for l in am1 if (l in wghts and wghts[l] != 0)]) for am1 in atmosts])
            for am1 in sorted(atmosts_clid, key=lambda am1: len(am1), reverse=True):
                if len(am1) < 2:
                    continue
                cost += self.process_am1(self.soft[clid], am1, wghts, self.vpool)
            
            ## here is the start cost
            self.soft[clid].cost = cost
            #print(sum(wghts.values()), cost)

            # adding remaining leaves with non-zero weights as soft clauses
            for lit, wght in wghts.items():
                if wght != 0:
                    self.soft[clid].append([ lit], weight=wght)
            
#             print('class:',clid)
#             for cl,w in zip(self.soft[clid].soft, self.soft[clid].wght):
#                 print(w,end=' ')
#                 self.printLits(cl)

    def encode_fdom(self):
        """
            encode feature domains
        """
        
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        self.categories = collections.defaultdict(lambda: [])
        for f in self.feature_names:
            if '_' in f:
                self.categories[f.split('_')[0]].append(self.newVar(f))        
        for c, feats in six.iteritems(self.categories):
            # exactly-one feat is True
            self.cnf.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
            
        # lits of intervals   
        for f, intvs in six.iteritems(self.ivars):
            if len(intvs) > 2:
                # enforcing that there is at least one interval    
                self.cnf.append(intvs)  
                
        n = len(self.cnf.clauses) 
        #
        for f, threshold in six.iteritems(self.thvars):
            # main part: order + domain encoding
            for j, thvar in enumerate(threshold[:-1]):
                ivar = self.ivars[f][j+1]
                lvar = threshold[j+1]
                prev = threshold[j]
                
                # order encoding
                self.cnf.append([prev, -lvar])
                
                # domain encoding
                self.cnf.append([ -ivar, prev ])
                self.cnf.append([ -ivar, -lvar ])
                self.cnf.append([ ivar, -prev, lvar ])
                
            if len(threshold):    
                # separate case of the first interval    
                self.cnf.append([ -self.ivars[f][0], -threshold[0] ])
                self.cnf.append([ self.ivars[f][0], threshold[0] ])
            
                # separate case of the last interval (till "+inf")
                self.cnf.append([ -self.ivars[f][-1], threshold[-1] ])
                self.cnf.append([ self.ivars[f][-1], -threshold[-1] ])
                
                       
#         for f, threshold in six.iteritems(self.thvars):
#             for j, thvar in enumerate(threshold):
#                 d = j+1
#                 pos, neg = self.ivars[f][d:], self.ivars[f][:d] 
                
#                 if j == 0:
#                     assert(len(neg) == 1)
#                     self.cnf.append([thvar, neg[-1]])
#                     self.cnf.append([-thvar, -neg[-1]])
#                 else:
#                     self.cnf.append([thvar, neg[-1], -threshold[j-1]])
#                     self.cnf.append([-thvar, threshold[j-1]])
#                     self.cnf.append([-thvar, -neg[-1]])
                
#                 if j == len(threshold) - 1:
#                     assert(len(pos) == 1)
#                     self.cnf.append([-thvar, pos[0]])
#                     self.cnf.append([thvar, -pos[0]])
#                 else:
#                     self.cnf.append([-thvar, pos[0], threshold[j+1]])
#                     self.cnf.append([thvar, -pos[0]])
#                     self.cnf.append([thvar, -threshold[j+1]]) 

#         for cl in self.cnf.clauses[n:]:
#            self.printLits(cl)
                
    def process_am1(self, formula, am1, wghts, vpool):
        """
            Detect AM1 constraints between the leaves of one tree and add the
            corresponding soft clauses to the formula.
        """

        cost = 0

        # filtering out zero-weight literals
        am1 = [l for l in am1 if wghts[l] != 0]

        # processing the literals until there is only one literal left
        while len(am1) > 1:
            minw = min(map(lambda l: wghts[l], am1))
            cost += minw * (len(am1) - 1)

            lset = frozenset(am1)
            if lset not in vpool.obj2id:
                selv = vpool.id(lset)

                # adding a new hard clause
                formula.append(am1 + [-selv])
            else:
                selv = vpool.id(lset)

            # adding a new soft clause
            formula.append([selv], weight=minw)

            # filtering out non-zero weight literals
            i = 0
            while i < len(am1):
                wghts[am1[i]] -= minw

                if wghts[am1[i]] == 0:
                    am1[i] = am1[len(am1) - 1]
                    am1.pop()
                else:
                    i += 1

        return cost

            