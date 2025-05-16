
import resource

from .encode import SATEncoder, MXEncoder

from pysat.examples.hitman import Hitman
from pysat.formula import CNF, WCNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.lbx import LBX
from pysat.examples.mcsls import MCSls
from pysat.examples.rc2 import RC2
from pysat.pb import PBEnc

from .mxreason import MXReasoner, ClassEnc

#
#==============================================================================
class SATExplainer(object):
    """
        An SAT-inspired minimal explanation extractor for Random Forest models.
    """

    def __init__(self, encoding, inps, preamble, target_name, verb=1):
        """
            Constructor.
        """
        self.enc = encoding
        self.inps = inps  # input (feature value) variables
        self.target_name = target_name
        self.preamble = preamble
        self.verbose = verb
        self.slv = None
        self.cnf = None
      
    def prepare_selectors(self, sample):
        # adapt the solver to deal with the current sample
        self.cnf = self.enc.cnf.copy()
        #self.csel = []
        self.assums = []  # var selectors to be used as assumptions
        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids
        self.sel2v = {} # selectors to (categorical/interval) values
        
        #for i in range(self.enc.num_class):
        #    self.csel.append(self.enc.newVar('class{0}'.format(i)))
        #self.csel = self.enc.newVar('class{0}'.format(self.enc.target))
               
        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, sample), 1):
            if '_' in inp:
                # binarized (OHE) features
                assert (inp not in self.enc.intvs)
                
                feat = inp.split('_')[0]
                selv = self.enc.newVar('selv_{0}'.format(feat))
            
                self.assums.append(selv)   
                if selv not in self.sel2fid:
                    self.sel2fid[selv] = int(feat[1:])
                    self.sel2vid[selv] = [i - 1]
                else:
                    self.sel2vid[selv].append(i - 1)
                    
                p = self.enc.newVar(inp) 
                if not val: # val == 0.
                    p = -p
                else:
                    self.sel2v[selv] = p
                    
                self.cnf.append([-selv, p])
                #self.enc.printLits([-selv, p])
                    
            elif len(self.enc.intvs[inp]):        
                v = next((intv for intv in self.enc.intvs[inp] if intv >= val), None)     
                assert(v is not None)
                
                selv = self.enc.newVar('selv_{0}'.format(inp))     
                self.assums.append(selv)  
                
                assert (selv not in self.sel2fid)
                self.sel2fid[selv] = int(inp[1:])
                self.sel2vid[selv] = [i - 1]
                            
                for j,p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        self.sel2v[selv] = p
                    else:
                        cl += [-p]
                    
                    self.cnf.append(cl)
                    #self.enc.printLits(cl)

        
    
    def explain(self, sample, xtype='abd', smallest=False):
        """
            Hypotheses minimization.
        """
        if self.verbose:
            pred = self.target_name[self.enc.target]
            print('  explaining:  "IF {0} THEN class={1}"'.format(' AND '.join(self.preamble), pred))
                    
        
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        self.prepare_selectors(sample)
        
        if xtype == 'abd':
            # abductive (PI-) explanation
            expl = self.compute_axp(smallest) 
        else:
            # contrastive explanation
            expl = self.compute_cxp(smallest)
 
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
    
        # clear cnf
        del self.cnf
        self.cnf = None
        # delete sat solver
        self.slv.delete()
        self.slv = None
        
        if self.verbose:
            print('  time: {0:.3f}'.format(self.time))

        return expl    

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
        self.slv.append_formula(self.cnf)    
        
        if not smallest:
            #minimal()
            self._mus()
        else:
            self._mmus()

        expl = sorted([self.sel2fid[h] for h in self.assums if h>0 ])
        assert len(expl), 'Abductive explanation cannot be an empty-set! otherwise RF fcn is const, i.e. predicts only one class'
        
        if self.verbose:
            print("expl-selctors: ", expl)
            preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN class={1}"'.format(' AND '.join(preamble), self.target_name[self.enc.target]))
            print('  # hypos left:', len(expl))
            
        return expl
    
    
    def _mus(self):
        # simple deletion-based linear search for extracting an MUS (AXp)
        for i, p in enumerate(self.assums):
            to_test = self.assums[:i] + self.assums[(i + 1):] + [-p, -self.sel2v[p]]
            sat = self.slv.solve(assumptions=to_test)
            if not sat:
                self.assums[i] = -p         
        return [p for p in self.assums if p>0]
    

    def _mmus(self):
        """
            Compute a cardinality-minimal abductive explanation.
        """

        with Hitman(bootstrap_with=[[i for i in range(len(self.assums)) ]]) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.assums):
                to_test = self.assums[:i] + self.assums[(i + 1):]
                if self.slv.solve(assumptions=to_test):
                    hitman.hit([i])
            
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)
                
                if self.slv.solve(assumptions=[self.assums[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.assums))).difference(set(hset)))
                    model = self.slv.get_model()
                    for h in removed:
                        # assert '_' not in self.inps[self.sel2fid[self.assums[h]]]
                        # categorical data are not handled
                        if model[abs(self.assums[h]) - 1] != self.assums[h]:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)
                    # print(unsatisfied, hset)
                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        to_test = [self.assums[i] for i in hset] + [self.assums[h]]
                        if self.slv.solve(assumptions = to_test):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    for i in range(len(self.assums)):
                        if i not in hset:
                            self.assums[i] = -self.assums[i]
                    #self.assums = [self.assums[i] for i in hset]
                    break
                    
        return [p for p in self.assums if p>0]            

    
        
    def compute_cxp(self, smallest=False):
        """
            Compute a Contrastive eXplanation
        """         
        self.assums = sorted(set(self.assums))
        if self.verbose:
            print('  # hypos:', len(self.assums))   
        
        if not smallest:
            expl = self._mcs()
        else:
            expl = self._mmcs()
       
        expl = sorted(expl)
        assert len(expl), 'Contrastive explanation cannot be an empty-set!'         
        if self.verbose:
            print("expl-selctors: ", expl)
            preamble = [self.preamble[i] for i in expl]
            pred = self.target_name[self.enc.target]
            print(f'  explanation: "IF {" AND ".join([f"!({p})" for p in preamble])} THEN class ≠ {pred}"')
            
        return expl
    
    
    def _mcs(self):
        """
            compute (minimal) CXp
        """
        wcnf = WCNF()
        for cl in self.cnf:
            wcnf.append(cl)    
        for p in self.assums:
            wcnf.append([p], weight=1)
            
        # mcs solver
        self.slv = LBX(wcnf, use_cld=True, solver_name='g3')
        mcs = self.slv.compute()
        #expl = sorted([self.sel2fid[self.assums[i-1]] for i in mcs])
        
        return [self.sel2fid[self.assums[i-1]] for i in mcs]
        
        
    def _mmcs(self):
        """
            compute minimum/smallest CXp
        """
        wcnf = WCNF()
        for cl in self.cnf:
            wcnf.append(cl)    
        for p in self.assums:
            wcnf.append([p], weight=1)
            
        # mxsat solver
        self.slv = RC2(wcnf)
        model = self.slv.compute()
        model = [p for p in model if abs(p) in self.assums]            
        #expl = sorted([self.sel2fid[-p] for p in model if p<0 ])
        
        return [self.sel2fid[-p] for p in model if p<0]
            
    
    def enumerate(self, sample, xtype='con', smallest=False, xnum=10):
        """
            list all CXp's or AXp's
        """
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        if 'assums' not in dir(self):
            self.prepare_selectors(sample)
            self.assums = sorted(set(self.assums))
            #
        
        if xtype == 'abd':
            expls = self._enumus(self.slv, xnum, smallest)
        else:
            expls = self._enumcs(self.slv, xnum, smallest)
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        if self.verbose:
            print('c expl time: {0:.3f}'.format(time))
        #
        #self.slv.delete()
        #self.slv = None
        return expls
    
    def enumerate2(self, sample, xtype='con', smallest=False, xnum=10):
        """
            list all CXp's or AXp's
        """
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        if 'assums' not in dir(self):
            self.prepare_selectors(sample)
            self.assums = sorted(set(self.assums))
            #
        
        if xtype == 'abd':
            for expl in self._enumus2(self.slv, xnum, smallest):
                yield expl
        else:
            for expl in self._enumcs(self.slv, xnum, smallest):
                yield expl
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        if self.verbose:
            print('c expl time: {0:.3f}'.format(time))
#         #
#         #self.slv.delete()
#         #self.slv = None
#         return expls    
        
        
    def _enumus(self,  oracle, xnum, smallest=False):
        """
            marco-based algo for AXp's enumeration
        """
        expls = [] # AXp's
        duals = [] # CXp's
        
        if oracle is None:
            oracle = Solver(name="glucose3")
            oracle.append_formula(self.cnf)
            
            wcnf = WCNF()
            for cl in self.cnf:
                wcnf.append(cl)
            for h in self.assums:
                wcnf.append([h], weight=1)                
            xmcs = MCSls(wcnf, use_cld=True, solver_name='g3')
             
        def reduce_coxp(hset, unsatisfied):
            to_hit = []
            to_hit = xmcs.compute(enable=[i+1 for i in hset])
            xmcs.block(to_hit)
            assert (to_hit)
            # expl = [self.sel2fid[self.assums[i-1]] for i in mcs]
            to_hit = [h-1 for h in to_hit]             
#            # linear search
#             for h in unsatisfied:
#                 to_test = [self.assums[i] for i in hset] + [self.assums[h]]
#                 if oracle.solve(assumptions = to_test):
#                     hset.append(h)
#                 else:
#                     to_hit.append(h)                    
            return to_hit

        
        with Hitman(bootstrap_with=[[i for i in range(len(self.assums))]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.assums):
                to_test = self.assums[:i] + self.assums[(i + 1):]
                if oracle.solve(assumptions=to_test):
                    hitman.hit([i])
                    duals.append([self.sel2fid[hypo]])
            
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1
#                 if self.verbose > 1:
#                     print('iter:', iters)
#                     print('cand:', hset)                    
                if hset == None:
                     break                    
                
                if oracle.solve(assumptions=[self.assums[i] for i in hset]):                   
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.assums))).difference(set(hset)))
                    # print(removed)

                    model = oracle.get_model()
                    for h in removed:
                        # assert '_' not in self.inps[self.sel2fid[self.assums[h]]]
                        # categorical data are not handled
                        if model[abs(self.assums[h]) - 1] != self.assums[h]:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)
                    # print(unsatisfied, hset)
                    # computing an MCS (expensive)
                    to_hit = reduce_coxp(hset, unsatisfied)
                    #if self.verbose > 1:
                    #    print('coex:', to_hit)
                    hitman.hit(to_hit)
                    duals.append([self.sel2fid[self.assums[i]] for i in to_hit])
                    if len(duals)>=xnum:
                        break
                else:
                    #if self.verbose > 2:
                    #    print('expl:', hset)                    
                    expls.append([self.sel2fid[self.assums[i]] for i in hset])
                    #print(expls[-1])
                    hitman.block(hset)
                    if len(expls) >= xnum:
                        break                      
        
        oracle.delete()
        
        return expls  
    
    
    
    def _enumus2(self,  oracle, xnum, smallest=False):
        """
            marco-based algo for AXp's enumeration
        """
        expls = [] # AXp's
        duals = [] # CXp's
        v2fmap = {h:i for i,h in enumerate(self.assums)}
        
        if oracle is None:
            oracle = Solver(name="glucose3")
            oracle.append_formula(self.cnf)    
            
        with Hitman(bootstrap_with=[[i for i in range(len(self.assums))]]) as hitman:
            # computing unit-size MCS/MUS
            for i, hypo in enumerate(self.assums):
                to_test = self.assums[:i] + self.assums[(i + 1):]
                if oracle.solve(assumptions=to_test):
                    hitman.block([i]) # mcs
                    duals.append([self.sel2fid[hypo]])
                elif not oracle.solve(assumptions=[hypo]):
                    hitman.hit([i]) # mus
                    expls.append([self.sel2fid[hypo]])
                    yield expls[-1]
            
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1                 
                if hset == None:
                     break 
                        
                removed = list(set(range(len(self.assums))).difference(set(hset)))
                #print(hset, removed)
                to_test = [self.assums[j] for j in removed]
                if not oracle.solve(assumptions=to_test):                 
                    core = oracle.get_core()
                    core = [h for h in core if h in to_test]

                    if len(core) > 1 : # reduce to MUS
                        for i, p in enumerate(core):
                            to_test = core[:i] + core[(i + 1):]
                            if not oracle.solve(assumptions=to_test):
                                core[i] = -p 
                        core = [p for p in core if p > 0]  
                        
                    to_hit = [v2fmap[h] for h in core]
                    
                    expls.append([self.sel2fid[self.assums[i]] for i in to_hit])
                    hitman.hit(to_hit) 
                    
                    yield expls[-1]
                    if len(expls) >= xnum:
                        break                     
                else:
                    #print(hset)
                    duals.append([self.sel2fid[self.assums[i]] for i in hset])
                    hitman.block(hset)
                    if len(duals) >= xnum:
                        break                    
        oracle.delete()
        
        #return expls                     
        
            
    def _enumcs(self, oracle, xnum, smallest=False):        
        # compute CXp's/AE's    
        if oracle is None:    
            wcnf = WCNF()
            for cl in self.cnf:
                wcnf.append(cl)    
            for p in self.assums:
                wcnf.append([p], weight=1)
            if smallest:    
                # incremental maxsat solver    
                oracle = RC2(wcnf, adapt=True, exhaust=True, minz=True)
            else:
                # mcs solver
                oracle = LBX(wcnf, use_cld=True, solver_name='g3')
                #oracle = MCSls(wcnf, use_cld=True, solver_name='g3')                
                
        if smallest:  
            # minimum cardinality
            for model in oracle.enumerate(block=-1):
                #model = [p for p in model if abs(p) in self.assums]
                expl = sorted([self.sel2fid[-p] for p in model if (p<0 and (-p in self.assums))])
#                 cxp_feats = [f'f{j}' for j in expl]
#                 advx = []
#                 for f in cxp_feats:
#                     ps = [p for p in model if (p>0 and (p in self.enc.ivars[f]))]
#                     assert(len(ps) == 1)
#                     advx.append(tuple([f,self.enc.nameVar(ps[0])]))   
#                 yield advx
                yield expl
                if xnum <= j+1:
                    break                
        else:
            # subset minimal
            for j, mcs in enumerate(oracle.enumerate()):
                expl = sorted([self.sel2fid[self.assums[i-1]] for i in mcs])
                #assumptions = [-p if(i in mcs) else p for i,p in enumerate(self.assums, 1)]
                #for k, model in enumerate(oracle.oracle.enum_models(assumptions), 1):
#                 assumptions = [-p if(i in mcs) else p for i,p in enumerate(self.assums, 1)]                
#                 assert (oracle.oracle.solve(assumptions))
#                 model = oracle.oracle.get_model()
#                 cxp_feats = [f'f{j}' for j in expl]
#                 advx = []
#                 for f in cxp_feats:
#                     ps = [p for p in model if (p>0 and (p in self.enc.ivars[f]))]
#                     assert(len(ps) == 1)
#                     advx.append(tuple([f,self.enc.nameVar(ps[0])]))
#                 yield advx
                oracle.block(mcs)
                yield expl
                if xnum <= j+1:
                    break
                
        oracle.delete()
        oracle = None
        
        
        
#
#==============================================================================
class MXExplainer(SATExplainer):
    """
        A MaxSAT-inspired minimal explanation extractor for Random Forest models.
    """

    def __init__(self, enc, inps, preamble, target_name, verb=1):
        super(MXExplainer, self).__init__(enc, inps, preamble, target_name, verb)


    def _prepare_encoding(self, encoding):
        for label in range(self.enc.num_class):
            if label == self.enc.target:
                encoding[label] = ClassEnc(formula=WCNF(), leaves=self.enc.leaves[label], trees=[])
            else:
                formula = WCNF()
                for cl in self.cnf:
                    formula.append(cl)
                for cl in self.enc.soft[label].hard:
                    formula.append(cl)
                for cl,w in zip(self.enc.soft[label].soft, self.enc.soft[label].wght):
                    formula.append(cl, weight=w)
                formula.vmax =  self.enc.soft[label].vmax
                formula.cost =  self.enc.soft[label].cost
                encoding[label] = ClassEnc(formula, self.enc.leaves[label], [])        

    def _mus(self):
        """
            deletion-based linear search algo for extracting an AXp (MUS)
        """
                
        encoding = {}
        self._prepare_encoding(encoding)                

        # MaxSAT oracle
        oracle = MXReasoner(encoding, self.enc.target, solver='g3', oracle='int', am1=True,
                exhaust=True, minz=True, trim=0)       
                
        for i, p in enumerate(self.assums):
            to_test = self.assums[:i] + self.assums[(i + 1):] + [-p, -self.sel2v[p]]
            # oracle.get_coex(to_test, full_instance=[True,False],  early_stop=True)
            if not oracle.get_coex(to_test,  early_stop=True):
                self.assums[i] = -p 
        core = [p for p in self.assums if p > 0]     
        oracle.delete()
        
        return core
               
        
    def _mmus(self):
        #raise NotImplementedError('Computing smallest abductive explanations is not yet implemented.')
        return self._mus()    
    
        
    def _mcs(self):
        """
            Linear search algo for extracting a CXp (MCS)
        """
        encoding = {}
        self._prepare_encoding(encoding)  
        
        # MaxSAT oracle
        self.slv = MXReasoner(encoding, self.enc.target, solver='g3', oracle='int', am1=True,
                exhaust=True, minz=True, trim=0)  
        
#         assert(self.slv.get_coex(self.assums,  early_stop=True) is None)

        # setting preferred polarities
        for clid in self.slv.oracles:
            self.slv.oracles[clid].oracle.set_phases(self.assums) 
        model = self.slv.get_coex([], early_stop=True)
        assert (len(model))
        # self.assums = [-h for h in self.assums]
        self.assums = [p if model[abs(p)-1]>0 else -p for p in self.assums]
        # self.enc.printLits(self.assums) 
        
        for i, p in enumerate(self.assums):
            if p > 0:
                continue
            to_test = self.assums[:i] + self.assums[(i + 1):] + [-p, self.sel2v[-p]]
            if self.slv.get_coex(to_test,  early_stop=True):
                self.assums[i] = -p
        mcs = [self.sel2fid[-p] for p in self.assums if p < 0]
        #oracle.delete() 
        
        return mcs
    
    
    def _mmcs(self):
        # computed smallest CXp
        cxps = []
        for cid in range(self.enc.num_class):
            if cid == self.enc.target:
                continue
            wcnf = WCNF()
            #wcnf = WCNFPlus()
            wcnf.extend(self.cnf.clauses)
        
            # weight and the current & target class
            wght, lhs, cost = [], [], 0
            for (lit, w1), (lit2, w2) in zip(self.enc.leaves[cid], self.enc.leaves[self.enc.target]):
                assert (lit == lit2)
                if (w1 - w2):
                    w = (w1 - w2)*(10 ** 1)
                    cost += 0 if w>0 else -w
                    w, l = (w, lit) if w>0 else (-w, -lit)           
                    wght.append(w)
                    lhs.append(l)
#                     wght.append(w1 - w2)
#                     wght[-1] *= (10 ** 2)
#                     lhs.append(lit)                    
              
#             wcnf.extend(self.enc.soft[cid].hard)
#             lhs = [cl[0] for cl in self.enc.soft[cid].soft]
#             wght = [w *(10 ** self.enc.relax) for w in self.enc.soft[cid].wght]
#             #print(min(wght), max(wght), min(self.enc.soft[cid].wght), max(self.enc.soft[cid].wght))
#             print("PB", len(lhs), len(self.enc.soft[cid].wght))
#             cost = self.enc.soft[cid].cost*(10 ** self.enc.relax)
#             cls = PBEnc.atleast(lits=lhs, weights=wght, bound=cost, vpool=self.enc.vpool)
#             print(len(cls.clauses))
            
            cls = PBEnc.atleast(lits=lhs, weights=wght, bound=cost, vpool=self.enc.vpool) # EncType.native=6
#             print(len(cls.atmosts[0][0]), (cls.atmosts[0][1]), len(cls.atmosts[0][2]))
#             for atms in cls.atmosts: 
#                 wcnf.append(atms, is_atmost=True)
            wcnf.extend(cls.clauses)
            
            for h in self.assums:
                wcnf.append([h], weight=1)
            
            with RC2(wcnf, solver='g3', adapt=False, exhaust=True, minz=False, trim=5) as self.slv:
                model = self.slv.compute()
                assert (model)
                expl = [self.sel2fid[p] for p in self.assums if (model[abs(p) - 1] < 0)]
                cxps.append(expl)
            if len(expl) == 1:
                break
            #print(cxps)    
            expl = min(cxps, key=lambda x: len(x))   
        return expl
        
    
    def enumerate(self, sample, xtype='con', smallest=False, xnum=100):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        if 'assums' not in dir(self):
            self.prepare_selectors(sample)
            self.assums = sorted(set(self.assums))
            
        encoding = {}
        self._prepare_encoding(encoding)  
        
        # MaxSAT oracle
        oracle = MXReasoner(encoding, self.enc.target, solver='g3', oracle='int', am1=True,
                exhaust=True, minz=True, trim=0)
        
        if xtype == 'abd':
            axps, cxps = self._enumus(oracle, xnum, smallest)
        else:
            # xtype == 'con'
            cxps, axps = self._enumcs(oracle, xnum, smallest) 
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        if self.verbose:
            print('c expl time: {0:.3f}'.format(time))
            
        oracle.delete()  
        return axps, cxps
    
            
    def _enumus(self, oracle, xnum, smallest=False):        
        # result
        axps = []
        # just in case, let's save dual (contrastive) explanations
        cxps = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.assums))]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i,p in enumerate(self.assums):
                to_test = self.assums[:i] + self.assums[(i + 1):]    
                if oracle.get_coex(to_test, early_stop=True):
                    hitman.hit([i])
                    cxps.append([self.sel2fid[p]])
            
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                # self.calls += 1
                #print(hset, [self.assums[i] for i in hset])
                coex = oracle.get_coex([self.assums[i] for i in hset], early_stop=True)
                if coex:
                    to_hit = []
                    satisfied, unsatisfied = [], []
                    
                    removed = list(set(range(len(self.assums))).difference(set(hset)))
                    for j in removed:
                        if coex[abs(self.assums[j]) - 1] != self.assums[j]:
                            unsatisfied.append(j)
                        else:
                            hset.append(j)

                    unsatisfied = list(set(unsatisfied))
                    hset = list(set(hset))

                    # computing an MCS (expensive)
                    for j in unsatisfied:
                        # self.calls += 1
                        to_test = [self.assums[i] for i in hset]+[self.assums[j]]
                        if oracle.get_coex(to_test, early_stop=True):
                            hset.append(j)
                        else:
                            to_hit.append(j)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    cxps.append([self.sel2fid[self.assums[i]] for i in to_hit])
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    axps.append([self.sel2fid[self.assums[i]] for i in hset])
                    hitman.block(hset)
                    if len(axps) >= xnum:
                        break
                    
        return axps, cxps  
    
    
    def _enumcs(self, oracle, xnum, smallest=False):        
        # result
        cxps = []
        # just in case, let's save dual (abductive) explanations
        duals = [] 
        
        with Hitman(bootstrap_with=[[i for i in range(len(self.assums))]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, h in enumerate(self.assums):
                to_test = self.assums[:i] + self.assums[(i + 1):]
                if not oracle.get_coex([h], early_stop=True):
                    hitman.hit([i])
                    duals.append([self.sel2fid[h]])                    
                elif oracle.get_coex(to_test, early_stop=True):
                    hitman.block([i])
                    cxps.append([self.sel2fid[h]]) 
                    
            v2fmap = {p:i for i,p in enumerate(self.assums)}
            
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                # self.calls += 1
                
                removed = list(set(range(len(self.assums))).difference(set(hset)))
                to_test = [self.assums[j] for j in removed]
                if not oracle.get_coex(to_test, early_stop=True):
                    to_hit = oracle.get_reason()
                    #print(to_hit)

                    if len(to_hit) > 1 : # reduce to MUS
                        to_hit = list(to_hit)
                        for i, p in enumerate(to_hit):
                            to_test = to_hit[:i] + to_hit[(i + 1):]
                            if not oracle.get_coex(to_test,  early_stop=True):
                                to_hit[i] = -p 
                        to_hit = [p for p in to_hit if p > 0]  
                        
                    to_hit = [v2fmap[h] for h in to_hit]
                    
                    duals.append([self.sel2fid[self.assums[i]] for i in to_hit])

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    cxps.append([self.sel2fid[self.assums[i]] for i in hset])
                    # print(cxps[-1])
                    hitman.block(hset)
                    
                    if len(cxps) >= xnum:
                        break

        return cxps, duals