import os
import numpy as np

from decimal import Decimal
from six.moves import range
import math

import gurobipy as gp
from gurobipy import GRB

from hitman2 import Hitman3 
from xrf import SATExplainer

from pysat.solvers import Solver
from pysat.formula import WCNF
#from pysat.card import CardEnc
from pysat.examples.lbx import LBX


#==============================================================================
class Explainer(SATExplainer):
    """
        Explainer (inflated XP) 
    """

    def __init__(self, encoding, inps, preamble, target_name, x_bounds, verb=1):
        """
            Initialiser.
        """
        super(Explainer, self).__init__(encoding, inps, preamble, target_name, verb)
        
        self.x_min, self.x_max = x_bounds
        
        
    def prepare_selectors(self, sample):
        # adapt the solver to deal with the current sample
        self.enc.cnf.encoded = []
        self.cnf = self.enc.cnf.copy()
        #self.csel = []
        self.assums = []  # var selectors to be used as assumptions
        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids
        self.sel2v = {} # selectors to (categorical/interval) values
        
        self.in2Iid = [] # inst to interval/cat-value id
               
        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, sample), 1):
            if '_' in inp:
                # binarized (OHE) features
                assert (inp not in self.enc.intvs)
                
                feat, j = inp.split('_')
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
                    self.in2Iid.append(int(j))
                    
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
                        self.in2Iid.append(j)
                        cl += [p]
                        self.sel2v[selv] = p
                    else:
                        cl += [-p]
                        
                    self.cnf.append(cl)
            else:
                self.in2Iid.append(None)
        
    
    def compute_axp(self, optimal=False):
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
        
        if optimal:
            # compute maximum/optimum coverage axp
            infx = self._mxmus()
        else:    
            # compute an inflated axp
            infx = self._mus()
        
        self.infx = infx.copy()

        expl = sorted([self.sel2fid[h] for h in self.assums if h>0 ]) # AXp
        assert len(expl)
        
        if self.verbose:
            preamble = self.preamble_expl(infx)
            print("expl-selctors: ", expl)            
            print('  expl: "IF [{0}] THEN class={1}"'.format('] AND ['.join([' OR '.join(f) for f in preamble]), 
                                                          self.target_name[self.enc.target]))        
            #preamble = [self.preamble[i] for i in expl]
            #print('  explanation: "IF {0} THEN class={1}"'.format(' AND '.join(preamble), self.target_name[self.enc.target]))
            print('  # hypos left:', len(expl))
            print('  cov:', f"{self.coverage(infx,log=True):.2f}")
            #print([len(infx[i]) for i in infx])
            #print('  cov:', f"{self.coverage(infx)}")
            print('  calls:', self.calls)
        return expl
    
    def preamble_expl(self, infx):
        preamble = []
        for i in infx:
            feat = f'f{i}'
            preamble.append([])
            if feat in self.enc.intvs:
                #dom = sorted([int(self.enc.nameVar(p).split('_')[1][4:]) for p in infx[i]])
                dom = infx[i]
                assert (len(self.enc.intvs[feat]) - len(dom))
                dom = [dom[0]]+[dom[j] for j in range(1, len(dom)) if (dom[j] - dom[j-1] > 1) or j==len(dom)-1 ]
#                 for j in dom:
#                     if j == 0:
#                         preamble[-1].append(f'{feat}<={self.enc.intvs[feat][j]:.3f}')
#                     elif len(self.enc.intvs[feat]) == 2 or j == len(self.enc.intvs[feat]) - 1:
#                         # binary
#                         preamble[-1].append(f'{feat}>{self.enc.intvs[feat][j-1]:.3f}')
#                     else:
#                         preamble[-1].append('{1:.3f}<{0}<={2:.3f}'.format(feat, self.enc.intvs[feat][j-1], 
#                                                                           self.enc.intvs[feat][j]))               
                if dom[0] == 0:
                    preamble[-1].append('{0}<={1:.3f}'.format(feat, self.enc.intvs[feat][dom[-1]]))
                elif dom[-1] == len(self.enc.intvs[feat]) - 1:
                    preamble[-1].append(f'{feat}>{self.enc.intvs[feat][dom[0]-1]:.3f}')
                else:
                    preamble[-1].append('{1:.3f}<{0}<={2:.3f}'.format(feat, self.enc.intvs[feat][dom[0]-1], 
                                                                          self.enc.intvs[feat][dom[-1]]))                        
            else:
                dom = [int(self.enc.nameVar(p).split('_')[1]) for p in infx[i]]
                assert len(self.enc.categories[feat]) - len(dom)
                for j in dom:
                    preamble[-1].append('{0}={1}'.format(feat,j))
                    
        return preamble
    
    
    def coverage(self, infx, log=False):
        if log:
            cov = 0.
            x = self.x_max - self.x_min
            for i in range(len(x)):
                if not x[i]:
                    x[i] = 1
            topw = np.log(x)
            for i in infx:
                f = "f{0}".format(i)
                splits = [self.x_min[i]] + self.enc.intvs[f][:-1] + [self.x_max[i]]
                cov += math.log(splits[infx[i][-1]+1] - splits[infx[i][0]])   
            lowc = 0.
            for _,i in self.sel2fid.items():
                f = "f{0}".format(i)
                splits = [self.x_min[i]] + self.enc.intvs[f][:-1] + [self.x_max[i]]
                lowc += math.log(splits[self.in2Iid[i]+1] - splits[self.in2Iid[i]]) 
            cov -= lowc            
            cov += sum([topw[i] for _,i in self.sel2fid.items() if (i not in infx)])
            cov /= sum([topw[i] for _,i in self.sel2fid.items()]) - lowc
        else:
            cov = 1.
            for i in infx:
                f = "f{0}".format(i)
                splits = [self.x_min[i]] + self.enc.intvs[f][:-1] + [self.x_max[i]]
                cov *= (splits[infx[i][-1]+1] - splits[infx[i][0]]) / (self.x_max[i] - self.x_min[i])
        
        return cov*100.0
    
        
    def _mus(self):
        # simple deletion-based linear search for extracting inflated AXp
        for i, p in enumerate(self.assums):
            to_test = self.assums[:i] + self.assums[(i + 1):] + [-p, -self.sel2v[p]]
            sat = self.slv.solve(assumptions=to_test)
            if not sat:
                self.assums[i] = -p 
                
        xhypos = [p for p in self.assums if p>0] 
        iAXp =  {self.sel2fid[h]:[self.in2Iid[self.sel2fid[h]]] for h in xhypos}
        
        self.calls = len(self.assums)
        
        for i,h in enumerate(xhypos):
            fid = self.sel2fid[h]
            if len(self.sel2vid[h]) > 1:
                # categorical data
                domain = self.enc.categories['f{0}'.format(fid)] 
                for j,q in enumerate(domain):
                    if q == self.sel2v[h]:
                        continue
                    if len(iAXp[self.sel2fid[h]]) == len(domain) - 1:
                        continue
                    to_test = xhypos[:i] + xhypos[i+1:] + [q]
                    sat = self.slv.solve(assumptions=to_test)
                    if not sat:    
                        iAXp[self.sel2fid[h]].append(j)                
                    self.calls += 1
            else:
                assert (len(self.sel2vid[h]) == 1)
                # interval data
                domain = self.enc.ivars['f{0}'.format(fid)] 
                l, u = iAXp[fid][0]-1, iAXp[fid][0]+1
                to_test = xhypos[:i] + xhypos[i+1:]
                
                # inflate upper bound
                while (u < len(domain)) and not self.slv.solve(assumptions=to_test+[domain[u]]):
                    iAXp[fid].append(u)
                    u += 1
                    self.calls += 1
                # inflate lower bound    
                while (l >= 0) and not self.slv.solve(assumptions=to_test+[domain[l]]):
                    iAXp[fid].append(l)
                    l -= 1  
                    self.calls += 1
            iAXp[fid].sort()
            if len(iAXp[fid]) > 1: # feature has been inflated
                xhypos[i] = self.enc.newVar('s_f{0}'.format(fid))
                cl = [domain[j] for j in iAXp[fid]] # (x_i = v_i) or (x_i=v_j) or....(x_i=v_n) 
                self.slv.add_clause(cl+[-xhypos[i]])
#                 for j,q in enumerate(domain):
#                     if (j not in iAXp[fid]):
#                         self.slv.add_clause([-xhypos[i], -q])
#                 self.slv.add_clause([-h]) # deactivate selv_f_i
        
        #print(iAXp)            
        return iAXp
    
    
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
        
        xhypos = [self.assums[i-1] for i in mcs] 
        self.infx =  {self.sel2fid[h]:[] for h in xhypos}
        
        self.slv.delete() # rm LBX
        self.slv = Solver(name="glucose3") # create SAT slv
        self.slv.append_formula(self.cnf)          
        
        for i,h in enumerate(xhypos):
            
            if len(self.sel2vid[h]) > 1:
                # categorical data
                domain = self.enc.categories['f{0}'.format(self.sel2fid[h])] 
            else:
                assert (len(self.sel2vid[h]) == 1)
                # interval data
                domain = self.enc.ivars[self.inps[self.sel2vid[h][0]]] 
                
            for j,q in enumerate(domain):
                if q == self.sel2v[h]:
                    continue
                if len(self.infx[self.sel2fid[h]]) == len(domain) - 1:
                    continue
                to_test = [h for h in self.assums if (h not in xhypos)] + [q] 
                sat = self.slv.solve(assumptions=to_test)
                if sat:    
                    self.infx[self.sel2fid[h]].append(j)
                    
        return [self.sel2fid[self.assums[i-1]] for i in mcs]
    
     

#==============================================================================
class ILPExplainer(Explainer):
    """
        ILP-based Explainer (computing inflated abductive explanation) 
    """
    
    
    def _mxmus(self):
        """
            compute a cardinality-maximal Inflated AXp
        """
        save_file_name = './temp/grb.lp'
        grb =  gp.Model(os.path.basename(save_file_name))
        grb.setParam(GRB.Param.OutputFlag, 0)
        grb.setParam(GRB.Param.LogToConsole, 0)
        
        to_hit, subject_to = [], []
        weights = []
        for _,i in self.sel2fid.items():
            f = "f{0}".format(i)
            splits = [self.x_min[i]] + self.enc.intvs[f][:-1] + [self.x_max[i]]
            vars, wght = [], []
            for j in range(self.in2Iid[i]+1):
                for k in range(self.in2Iid[i],len(self.enc.ivars[f])):
                    y = grb.addVar(vtype=GRB.BINARY, name=f'f{i}_l{j}_u{k}')
                    w = math.log(splits[k+1] - splits[j])
                    vars.append(y)
                    wght.append(w) # subject_to maximze
            
            to_hit.append(vars)
            grb.addLConstr(gp.LinExpr([1]*len(vars), vars), GRB.EQUAL, 1)
#             s = grb.addVar(vtype=GRB.CONTINUOUS, name=f"feat_{i}")
#             grb.addLConstr(gp.LinExpr(wght+[-1], vars+[s]), GRB.EQUAL, 0)
#             subject_to.append(s)
            subject_to.extend(vars)
            weights.extend(wght)
            
        # Set objective
        #grb.setObjective(gp.LinExpr([(1.0, x) for x in subject_to]), GRB.MAXIMIZE)
        grb.setObjective(gp.LinExpr(weights, subject_to), GRB.MAXIMIZE)

        fid2Sid = {self.sel2fid[h]:i for i,h in enumerate(self.assums)}
        self.calls = 0
        
        # encode lower and upper bounds
        LBs, UBs = {}, {}
        for _,fid in self.sel2fid.items():
            f = "f{0}".format(fid)
            lb = [grb.addVar(vtype=GRB.BINARY, name=f'l{fid}>T{j}') for j in range(self.in2Iid[fid])]
            ub = [grb.addVar(vtype=GRB.BINARY, name=f'u{fid}<T{j}') for j in range(self.in2Iid[fid],len(self.enc.intvs[f])-1)]   
            LBs[fid], UBs[fid] = lb, ub
                
            for j in range(len(lb)-1):
                grb.addLConstr(lb[j+1] - lb[j], GRB.LESS_EQUAL, 0) # e.g. [li ≥ 0.2] → [li ≥ −0.4]
            for j in range(len(ub)-1):
                grb.addLConstr(ub[j] - ub[j+1], GRB.LESS_EQUAL, 0) # e.g. [ui < −0.4] → [ui < 0.2]
            
                
            for j in range(self.in2Iid[fid]+1):
                r = len(self.enc.ivars[f]) - self.in2Iid[fid]
                for k in range(r):
                    y = to_hit[fid2Sid[fid]][j*r+k]    
                    coefs, vars = [], []
                    #lits = []
                    if j == 0:
                        if j<self.in2Iid[fid]:
                            #lits.append(-lb[j])
                            vars.append(lb[j])
                            coefs.append(-1)
                    elif j < self.in2Iid[fid]:
                        #lits.append(lb[j-1])
                        #lits.append(-lb[j])
                        vars.append(lb[j-1])
                        vars.append(lb[j])
                        coefs.append(1)
                        coefs.append(-1)
                    else: # j == self.in2Iid[fid]
                        #lits.append(lb[j-1])
                        vars.append(lb[j-1])
                        coefs.append(1)
                    if k == 0:
                        if k < r-1:
                            #lits.append(ub[k])
                            vars.append(ub[k])
                            coefs.append(1)
                    elif k < r-1:
                        #lits.append(-ub[k-1])
                        #lits.append(ub[k])
                        vars.append(ub[k-1])
                        vars.append(ub[k])
                        coefs.append(-1)
                        coefs.append(1)
                    else:
                        #lits.append(-ub[k-1])
                        vars.append(ub[k-1])
                        coefs.append(-1)
                        
                    #assert len(lits)
                    #formula.append([y]+[-l for l in lits])
                    #formula.extend([[-y, l] for l in lits])
                    
                    rhs = len(coefs) - 1 - len([c for c in coefs if c<0])
                    grb.addLConstr(gp.LinExpr(coefs+[-1], vars+[y]), GRB.LESS_EQUAL, rhs)
                    for c,x in zip(coefs, vars):
                        rhs = 1 if c<0 else 0
                        grb.addLConstr(gp.LinExpr([1, -c], [y, x]), GRB.LESS_EQUAL, rhs)
        
        #grb.write('temp/grb.lp') 
        grb.update()
        """
        grb.optimize()
        print('runtime:', grb.Runtime)
        if grb.Status == GRB.OPTIMAL: # get predicted class
            print([(v.VarName, v.X) for hs in to_hit for v in hs if v.X>0]) 
            print('Obj: %g' % grb.ObjVal)    
        else:
            assert (grb.Status == GRB.INFEASIBLE)
            print(grb.Status)
        """
            
        # compute CXp's of size 1 if any exists
        for i,h in enumerate(self.assums):
            fid = self.sel2fid[h]
            to_test = [self.sel2v[p] for p in self.assums[:i]+self.assums[i+1:]]
            if self.slv.solve(assumptions=to_test):
                model = self.slv.get_model()
                intvs = self.enc.ivars['f{0}'.format(fid)]
                icxp = [j for j,p in enumerate(intvs) if model[abs(p)-1]>0 ]
                assert len(icxp) == 1
                assert (icxp[0] != self.in2Iid[fid])
                l, u = icxp[0]-1, icxp[0]+1
                # inflate upper bound
                while (u < self.in2Iid[fid]) and self.slv.solve(assumptions=to_test+[intvs[u]]):
                    icxp.append(u)
                    u += 1
                # inflate lower bound    
                while (l > self.in2Iid[fid]) and self.slv.solve(assumptions=to_test+[intvs[l]]):
                    icxp.append(l)
                    l -= 1
                icxp.sort()

                k = len(intvs)-self.in2Iid[fid]        
                if icxp[0] > self.in2Iid[fid]:
                    to_block = [to_hit[i][j*k+(icxp[0]-self.in2Iid[fid])+u] for j in range(self.in2Iid[fid]+1) 
                                    for u in range(len(intvs) - icxp[0])]
                    b = UBs[fid][icxp[0]-self.in2Iid[fid]-1]
                else:
                    assert icxp[-1] < self.in2Iid[fid] 
                    to_block = to_hit[i][:(icxp[-1]+1)*k]
                    ##
                    b= LBs[fid][icxp[-1]]

                grb.addLConstr(b, GRB.GREATER_EQUAL, 1)
                    
                if self.verbose > 1:    
                    print('to_block:', b.VarName) 
            
        # Counterexample-guided abstraction refinement (CEGAR) loop
        iters, otime = 0, 0.
        while True:
            grb.optimize()
            assert (grb.Status == GRB.OPTIMAL)
            hset = [v for hs in to_hit for v in hs if v.X>0]           

            assert len(hset) == len(self.assums)
            if self.verbose > 1:
                print('\nhset:', [p.VarName for p in hset])
            iters += 1
            self.calls += 1
            #print(grb.Runtime, grb.ObjVal)
            otime += grb.Runtime

            
            xhypos, iAXp = [], {}
            for i,p in enumerate(hset): # same order as self.assums
                f,j,k = p.VarName.split('_')
                fid, j, k = int(f[1:]), int(j[1:]), int(k[1:])

                if k-j+1 < len(self.enc.ivars[f]):
                    h = self.enc.newVar(f's{fid}_{iters}')
                    cl = self.enc.ivars[f][j:k+1]
                    cl.append(-h)
                    self.slv.add_clause(cl)
                    xhypos.append(h)
                    iAXp[fid] = list(range(j,k+1))
                else: # free f_i
                    xhypos.append(-self.assums[i])
            

            if self.slv.solve(assumptions=xhypos):
                model = self.slv.get_model()
                to_test = [model[self.sel2v[h]-1] for h in self.assums]
                for i,h in enumerate(to_test):
                    if h<0:
                        to_test[i] = -h                            
                        if self.slv.solve(assumptions=to_test+xhypos):
                            model = self.slv.get_model() # save last sat assignment          
                        else:
                            to_test[i] = h

                icxp, xhypos = {}, []   
                for i,h in enumerate(to_test): 
                    if h<0: # free features
                        #f = "f{0}".format(self.sel2fid[self.assums[i]])
                        f = self.inps[self.sel2vid[self.assums[i]][0]]
                        for j,I in enumerate(self.enc.ivars[f]):
                            if model[I-1]>0:
                                icxp[self.sel2fid[self.assums[i]]] = [j]
                                break  
                        fid = self.sel2fid[self.assums[i]]
                        if j < self.in2Iid[fid]:
                            cl = self.enc.ivars[f][:self.in2Iid[fid]]
                        else: 
                            cl = self.enc.ivars[f][self.in2Iid[fid]+1:]
                        cl.append(- self.enc.newVar(f'a{fid}_{iters}'))
                        self.slv.add_clause(cl)
                    else:
                        # fixed features
                        xhypos.append(self.assums[i])

                #print(icxp)
                # inflate CXp 
                for fid in icxp:
                    intvs = self.enc.ivars[f'f{fid}']
                    to_test = xhypos + [self.enc.newVar(f'a{i}_{iters}') for i in icxp if (i < fid)]

                    l, u = icxp[fid][0]-1, icxp[fid][0]+1
                    # inflate upper bound
                    while (u < self.in2Iid[fid]) and self.slv.solve(assumptions=to_test+[intvs[u]]):
                        icxp[fid].append(u)
                        u += 1
                    # inflate lower bound    
                    while (l > self.in2Iid[fid]) and self.slv.solve(assumptions=to_test+[intvs[l]]):
                        icxp[fid].append(l)
                        l -= 1
                    if (icxp[fid][0] < self.in2Iid[fid] and  len(icxp[fid]) < len(intvs[:self.in2Iid[fid]])) or \
                        (icxp[fid][0] > self.in2Iid[fid] and len(icxp[fid]) < len(intvs[self.in2Iid[fid]+1:])):
                        h = self.enc.newVar(f'a{fid}_{iters}')
                        self.slv.add_clause([intvs[j] for j in icxp[fid]]+[-h])

                    icxp[fid].sort() 
                    # icxp[fid] = [icxp[fid][0]] + ([icxp[fid][-1]] if (icxp[fid][-1] > icxp[fid][0]) else []) 
                    icxp[fid] = (icxp[fid][0], icxp[fid][-1]) # (lb, ub)

                #print(icxp)
                to_block = []
                # block bounds                    
                for fid in icxp:
                    if (icxp[fid][-1] < self.in2Iid[fid]):
                        b = LBs[fid][icxp[fid][-1]]  
                    else: 
                        assert (icxp[fid][0] > self.in2Iid[fid])
                        b = UBs[fid][icxp[fid][0]-self.in2Iid[fid]-1]

                    to_block.append(b)

                if self.verbose > 1:    
                    print('to_block:', [p.VarName for p in to_block])
                
                #hitman.oracle.add_clause(to_block)
                grb.addLConstr(gp.LinExpr([1]*len(to_block), to_block), GRB.GREATER_EQUAL, 1)
                
            else:
                #self.enc.printLits(xhypos)   
                free = [h for h in xhypos if h<0] 
                for i,h in enumerate(self.assums):
                    if -h in free:
                        self.assums[i] = -h
                # end of cegar loop
                break
            
        if self.verbose:
            print('oracle time:', f'{otime:.3f}')
            
        return iAXp
        