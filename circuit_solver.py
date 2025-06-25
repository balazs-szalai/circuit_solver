# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:21:56 2025

@author: balazs
"""
from copy import copy
from typing import List
import numpy as np
from sympy import Matrix

global_branches = []
class Branch:
    def __init__(self, endpoints, props = None, global_branches = global_branches):
        self.props = props
        
        self.endpoints = endpoints
        if self in global_branches:
            raise RuntimeError(f'Branch {self} already exists')
        
        endpoints[0].branches.append(self)
        endpoints[1].branches.append(self)
        
        global_branches.append(self)
    
    def add_point(self, node):
        assert len(self.endpoints) < 2
        self.endpoints.append(node)
        node.branches.append(self)
    
    def other(self, node):
        assert node in self.endpoints
        if node == self.endpoints[0]:
            return self.endpoints[1]
        return self.endpoints[0]
    
    def __repr__(self):
        return f'{str(self.endpoints)}: {str(self.props)}'
    
    def __eq__(self, o):
        return self.endpoints == o.endpoints or self.endpoints == o.endpoints[::-1]
    
    def sgn(self, node):
        assert node in self.endpoints
        
        if node == self.endpoints[1]:
            return 1
        return -1
    
    def sgn2(self, endpoints):
        if self.endpoints == endpoints:
            return 1
        elif self.endpoints == endpoints[::-1]:
            return -1 
        else:
            raise RuntimeError('Incorrect endpoints given to check')
    

class Node:
    def __init__(self, branches = None, name = None):
        if branches == None:
            branches = []
        self.branches = branches
        self.name = name
    
    def reachable_nods(self):
        reachable = []
        for b in self.branches:
            reachable.append(b.other(self))
        return reachable
    
    def traverse(self, callback = None, args = (), visited = None, last = None):
        if visited == None:
            visited = []
        
        
        if self in visited:
            visited.append(self)
            return visited
        visited.append(self)
        
        reachable = self.reachable_nods()
        # print(reachable)
        if last:
            reachable.remove(last)
        
        # print(last, self, reachable)
        
        if len(reachable) >= 1:
            visited_copy = copy(visited)
            for node in reachable:
                visited = node.traverse(visited = copy(visited_copy), last = self, callback = callback, args=args)
        
        if visited and callback:
            callback(visited, *args)
    
    def serialize(self):
        tree = []
        
        cb = lambda x, y: y.append(x)
        
        self.traverse(callback=cb, args=(tree, ))
        return tree
    
    def __repr__(self):
        if self.name:
            return self.name
        else:
            return id(self)


#if there's a loop it is in the very end end characterised by a node appearing twice
def find_loops(paths):
    loops = []
    
    for path in paths:
        is_loop = False
        loop = []
        path = path[::-1]
        
        loop.append(path[0])
        
        for node in path[1:]:
            loop.append(node)
            if node == loop[0]:
                is_loop = True
                break
        
        if is_loop:
            loops.append(loop)
    
    return loops        

class Circuit:
    def __init__(self):
        self.global_branches = []
        self.nodes = []
    
    def add_node(self, node: Node):
        self.nodes.append(node)
        return node
    
    def add_branch(self, endpoints: List[Node], props = None):
        if endpoints[0] not in self.nodes or endpoints[1] not in self.nodes:
            raise RuntimeError('Both endpoints need to be in the circuit')
        if not props:
            props = {'R': 0, 'C': None, 'U': 0}
        
        Branch(endpoints, props, self.global_branches)
    
    def find_loops(self):
        paths = self.nodes[0].serialize()
        loops = find_loops(paths)
        return loops
    
    def Kirchoff_1(self):
        equations = []
        for node in self.nodes:
            eq = []
            node_branches = node.branches
            for branch in self.global_branches:
                if branch in node_branches:
                    eq.append(branch.sgn(node))                       
                else:
                    eq.append(0)
            eq.append(0)
            equations.append(eq)
            
        return equations
    
    def get_loop_branches(self, loop):
        branches = []
        indecies = []
        sgns = []
        for i in range(len(loop) - 1):
            ind = self.global_branches.index(Branch(loop[i: i+2], global_branches=[]))
            branch = self.global_branches[ind]
            branches.append(branch)
            indecies.append(ind)
            sgns.append(branch.sgn2(loop[i: i+2]))
        return branches, indecies, sgns
    
    def parse_branch_props(self, branch):
        props = branch.props
        ret = 0
        U = 0
        
        if props['R']:
            ret += props['R']
        if props['C']:
            ret += -1j/props['C']
        if props['U']:
            U = props['U']
        
        return ret, U
    
    def Kirchoff_2(self):
        equations = []
        loops = self.find_loops()
        
        for loop in loops:
            branches, indecies, sgns = self.get_loop_branches(loop)
            eq = [0 for _ in range(len(self.global_branches) + 1)]
            
            for ind, branch, sgn in zip(indecies, branches, sgns):
                ret, U = self.parse_branch_props(branch)
                eq[ind] += sgn*ret
                eq[-1] += sgn*U
            equations.append(eq)
        
        return equations
    
    def Kirchoff(self):
        eqs = self.Kirchoff_1()
        eqs.extend(self.Kirchoff_2())
        
        # print(eqs)
        
        A = np.array(eqs)
        # print(A)
        b = A[:, -1]
        A = A[:, :-1]
        
        return A, b
    
    def solve_numpy(self, A, b):
        success = True
        
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        Ax = A @ x
        residual_norm = np.linalg.norm(Ax - b)
        b_norm = np.linalg.norm(b)
        A_norm = np.linalg.norm(A)
        x_norm = np.linalg.norm(x)
        
        relative_residual = residual_norm / (b_norm + A_norm * x_norm)

        if relative_residual > 1e-8:
            success = False
        
        if rank < len(x):
            success = False
        
        # print(residual_norm, relative_residual, success)
        return x, success
    
    def solve(self, A, b, engine):
        success = True
        if engine == 'numpy':
            sol, success = self.solve_numpy(A, b)
        elif engine == 'sympy':
            A, b = Matrix(A), Matrix(b)
            sol = A.solve(b)
        elif engine == 'auto':
            try:
                sol, success = self.solve_numpy(A, b)
            except:
                A, b = Matrix(A), Matrix(b)
                sol = A.solve(b)
        else:
            raise RuntimeError(f'Engine {engine} not recognised')
        return sol, success
    
    def Kirchoff_solve(self, engine = 'auto'):
        A, b = self.Kirchoff()
        self.A = A
        self.b = b
        
        sol, success = self.solve(A, b, engine)
        self.sol = sol
        
        if not success:
            RuntimeWarning('likely unsuccessful solution')
        
        for k, x in enumerate(sol):
            self.global_branches[k].props['I'] = x
        
        return sol
    
    def Ohm_equations(self):
        equations = []
        l1 = len(self.global_branches)
        l2 = len(self.nodes)
        
        for ind_I, branch in enumerate(self.global_branches):
            eq = [0 for _ in range(l1+ l2)]
            ret, U = self.parse_branch_props(branch)
            
            i1 = self.nodes.index(branch.endpoints[0])
            i2 = self.nodes.index(branch.endpoints[1])
            
            eq[ind_I] = -ret
            eq[i1+l1] = -1
            eq[i2+l1] = 1
            eq.append(-U)
            
            equations.append(eq)
        
        return equations

    def Kirchoff_1_extend(self):
        zeros = [0 for _ in range(len(self.nodes))]
        equations = self.Kirchoff_1()
        
        for i in range(len(equations)):
            equations[i].extend(zeros)
        
        return equations
        

    def NA(self, ground, current_sources = None):
        l1 = len(self.global_branches)
        l2 = len(self.nodes)
        
        eqs = self.Kirchoff_1_extend()
        eqs.extend(self.Ohm_equations())
        
        gnd_eqs = []
        for gnd in ground:
            eq = [0 for _ in range(l1+l2+1)]
            eq[self.nodes.index(gnd)+l1] = 1
            gnd_eqs.append(eq)
        eqs.extend(gnd_eqs)
        
        if current_sources:
            for cs in current_sources:
                eq = [0 for _ in range(l1+l2+1)]
                eq[self.global_branches.index(cs[0])] = 1 
                eq[-1] = cs[1]
                eqs.append(eq)
        
        A = np.array(eqs)
        # print(A)
        b = A[:, -1]
        A = A[:, :-1]
        
        return A, b
    
    def NA_solve(self, ground, current_sources = None, engine = 'auto'):
        A, b = self.NA(ground, current_sources)
        self.A = A
        self.b = b
        
        sol, success = self.solve(A, b, engine)
        
        if not success:
            print('likely unsuccessful solution')
        
        for k, x in enumerate(sol[:len(self.global_branches)]):
            self.global_branches[k].props['I'] = x
        
        self.nodal_voltages = sol[len(self.global_branches):]
        self.sol = sol
        
        return sol
        


    def __repr__(self):
        return '\n'.join([str(b) for b in self.global_branches])
    
    

# circuit = Circuit()

# A = Node(name='A')
# circuit.add_node(A)

# B = Node(name='B')
# circuit.add_node(B)

# C = Node(name='C')
# circuit.add_node(C)

# D = Node(name='D')
# circuit.add_node(D)

# E = Node(name='E')
# circuit.add_node(E)

# F = Node(name='F')
# circuit.add_node(F)


# circuit.add_branch([A, B], {'R': 5, 'C': None, 'U':0})
# circuit.add_branch([B, C], {'R': 0, 'C': None, 'U':0})
# circuit.add_branch([C, D], {'R': 1, 'C': None, 'U':0})
# circuit.add_branch([A, D], {'R': 0, 'C': None, 'U':0})
# circuit.add_branch([A, E], {'R': 0, 'C': None, 'U':0})
# circuit.add_branch([B, F], {'R': 0, 'C': None, 'U':0})
# circuit.add_branch([E, F], {'R': 0, 'C': None, 'U':1})


# circuit.Kirchoff_solve()
# print(circuit)
#%%
import sympy as s 

R = 1#s.Symbol('R')
U0 = 1#s.Symbol('U_0')

circuit1 = Circuit()

A = circuit1.add_node(Node(name='A'))
B = circuit1.add_node(Node(name='B'))
C = circuit1.add_node(Node(name='C'))
D = circuit1.add_node(Node(name='D'))
E = circuit1.add_node(Node(name='E'))
F = circuit1.add_node(Node(name='F'))
G = circuit1.add_node(Node(name='G'))
H = circuit1.add_node(Node(name='H'))

circuit1.add_branch([A, B], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([B, C], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([C, D], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([D, A], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([E, F], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([F, G], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([G, H], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([H, E], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([A, E], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([B, F], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([C, G], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([D, H], {'R': R, 'C': None, 'U':0})
circuit1.add_branch([A, F], {'R': 0, 'C': None, 'U':U0})

circuit1.NA_solve([A])
# circuit1.Kirchoff_solve()
print(U0/circuit1.global_branches[-1].props['I'])

#%%
R = 0#s.Symbol('R')
C0 = 1#s.Symbol('C')
U0 = 1#s.Symbol('U_0')

circuit1 = Circuit()

A = circuit1.add_node(Node(name='A'))
B = circuit1.add_node(Node(name='B'))
C = circuit1.add_node(Node(name='C'))
D = circuit1.add_node(Node(name='D'))
E = circuit1.add_node(Node(name='E'))
F = circuit1.add_node(Node(name='F'))
G = circuit1.add_node(Node(name='G'))
H = circuit1.add_node(Node(name='H'))

circuit1.add_branch([A, B], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([B, C], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([C, D], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([D, A], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([E, F], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([F, G], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([G, H], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([H, E], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([A, E], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([B, F], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([C, G], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([D, H], {'R': R, 'C': C0, 'U':0})
circuit1.add_branch([A, G], {'R': 0, 'C': None, 'U':U0})

circuit1.NA_solve([A])
print(U0/circuit1.global_branches[-1].props['I'])

#%%
R = 1#s.Symbol('R')
C0 = 1#s.Symbol('C')
U0 = 1#s.Symbol('U_0')
n = 20

circuit1 = Circuit()

nodes = [Node(name = f'{i}_{j}') for i in range(n) for j in range(n)]

for i in range(n):
    for j in range(n):
        circuit1.add_node(nodes[i*n+j])

for i in range(n-1):
    for j in range(n):
        circuit1.add_branch([nodes[i*n + j], nodes[i*n + j + n]], {'R': R, 'C': C0, 'U':0})
        circuit1.add_branch([nodes[j*n + i], nodes[j*n + i + 1]], {'R': R, 'C': C0, 'U':0})

circuit1.add_branch([nodes[0], nodes[-1]], {'R': 0, 'C': 0, 'U':U0})

circuit1.NA_solve([nodes[0]])
# circuit1.Kirchoff_solve()
print(U0/circuit1.global_branches[-1].props['I'])
