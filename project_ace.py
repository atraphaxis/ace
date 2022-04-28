import matplotlib.pyplot as plt
import numpy as np
import math
import re
import copy

class Operation:

    def __init__(self, o: str | int):
        if isinstance(o, str):
            self.op = o
        else:
            self.op = str(o)

    def get_num(self) -> int:
        return int(re.findall(r'-?\d+', self.op)[0])

    def get_sym(self) -> str:
        try:
            sym = re.findall(r'\D', self.op)[0]
            if sym == '-':
                return '+'
            else:
                return sym
        except:
            return None

    def set_sym(self, new_sym) -> None:
        self.op = new_sym + re.findall(r'\d', self.op)[0]

    def set_num(self, new_num) -> None:
        self.op = re.findall(r'\D', self.op)[0] + new_num

    def __str__(self):
        return self.op

    def __int__(self):
        return int(self.op)


class Term():

    def __init__(self, t: list):
        # print(f'term init: {t}')
        if t == [] or (len(t) > 0 and isinstance(t[0], Operation)):
            self.term = t
        else:
            self.term = self.list_to_term(t).term
                
    def get_spec_op(self, op_type: str):
        """Get the (first instance of the) specific operation supplied in op_type in a given term in the form (index, value). Return None if there is no instance of the specified operation."""
        for i, op in enumerate(self.term):
            # print(f'op_type: {op_type}')
            if op.get_sym() == op_type:
                if op_type == ')':
                    return i
                else:
                    return (i, op.get_num())
        return None

    def has_var(self) -> bool:
        """If the term is not a constant, it is true."""
        return 'x' in self.to_list()

    def get_var(self):
        """Get the variable in the form (index, str). Return None if there is no instance."""
        for i, op in enumerate(self.term):
            if str(op).isalpha() and len(str(op)) == 1:
                return (i, str(op))

    def take_derivative(self):
        # print(f'before: {self.to_list()}')
        if not self.has_var():
            # print('here')
            return Term([Operation('0')])
        else:
            derivative = copy.deepcopy(self.term)
            exponent = self.get_spec_op('^')
            if not exponent == None:
                # Change the exponent.
                new_exponent = exponent[1] - 1
                derivative[exponent[0]].set_num(str(new_exponent))

                # Change the coefficient.
                coefficient = self.get_spec_op('*')
                if coefficient is None:
                    derivative.append(Operation(f'*{exponent[1]}'))
                else:
                    new_coefficient = exponent[1] * coefficient[1]
                    derivative[coefficient[0]].set_num(str(new_coefficient))
            else:
                # print(f'here 1: {Term(derivative).to_list()}')
                derivative[self.get_var()[0]] = Operation('1')
                # print(f'here 2: {Term(derivative).to_list()}, {Term(derivative).term}')
                derivative = [Operation(str(Term(derivative).evaluate(0)))]
                # print(f'here 3: {Term(derivative).to_list()}')
        # print(f'after: {self.to_list()}')
        # print(f'derivative: {derivative.to_list()}, {derivative}, {derivative.term}')
        return Term(derivative)

    def evaluate(self, x) -> int:
        """Evaluate the term at the given value of the variable."""
        skip = False
        result = 0
        for op in self.term:
            # print(f'str op: {str(op)}')
            if skip == True:
                if str(op) == ')':
                    skip = False
                continue
            if str(op) == '(':
                # print('here')
                # print(None)
                result += self.evaluate_parentheses(x)
                # print(None)
                # print(self.evaluate_parentheses(x))
                # print(result)
                # print('here')
                skip = True
            elif str(op).isalpha() and len(str(op)) == 1:
                # print(x)
                result += x
            elif str(op).isdecimal():
                result += int(op)
                # print('decimal')
            else:
                # print(f'else: {str(op)}')
                symbol = op.get_sym()
                number = op.get_num()
                if symbol == '+':
                    result += number
                elif symbol == '-':
                    result -= number
                elif symbol == '*':
                    result *= number
                elif symbol == '/':
                    result /= number
                elif symbol == '^':
                        result **= number
            # print(f'result: {result}')
        return result
    
    def simplify(self) -> None:
        """Simplify the term mathematically in place."""
        for i, op in enumerate(self.term):
            # print(str(op))
            if str(op) in ['^1', '*1', '/1']:
                self.term.pop(i)
            elif str(op) == '^0':
                self.term[self.get_var()[0]] = Operation('1')
                self.term.pop(i)
                # print(self.to_list())
                self.term = [str(self.evaluate(0))]
            elif str(op) == '*0':
                self.term = [Operation('0')]
        return Term(self.term)

    def append_operation(self, op):
        self.term.append(op)
    
    def to_list(self):
        term_list = []
        for op in self.term:
            term_list.append(str(op))
        return term_list

    def list_to_term(self, term_list):
        """Convert a term of the form ['C'] or ['x', '^C', '*C'] to Operation and Term objects."""
        term = Term([])
        for op_str in term_list:
            term.append_operation(Operation(op_str))
        return term
    
    def inside_parentheses(self):
        """Gets the Operations within the parentheses of a term. If there are none, return None."""
        if str(self.term[0]) == '(':
            # print(self.to_list())
            return self.term[1:self.get_spec_op(')')]
        return None

    def evaluate_parentheses(self, x: int) -> int:
        """Evaluate what is within the parenthesis if there are parenthesis in a term. If not, return None."""
        # print(Term(self.inside_parentheses()).to_list())
        if self.inside_parentheses() is not None:
            return Term(self.inside_parentheses()).evaluate(x)
        return None

    def substitute_parentheses(self, replace = None):
        """
        Replaces the parentheses and the Operations within the parentheses with the variable x if no replace is provided.
        If replace is not None, it replaces the x variable Operation with the given replace argument.
        If replace is None and the expression does not have parentheses, returns None.
        If replace is not None and the expression has parentheses, returns None.
        """
        if replace is None and str(self.term[0]) == '(':
            self.term = [Operation('x')] + self.term[self.get_spec_op(')') + 1:]
            return self.inside_parentheses()
        elif replace is not None and str(self.term[0]) == 'x':
            self.term = replace + self.term[1:]
            return None
        else:
            return None


class Expression():

    def __init__(self, e: list):
        if e == [] or (len(e) > 0 and isinstance(e[0], Term)):
            self.exp = e
        else:
            self.exp = self.list_to_expression(e).exp

    def evaluate(self, x):
        result = 0
        for term in self.exp:
            result += term.evaluate(x)
        return result

    def take_derivative(self):
        derivative = copy.deepcopy(self.exp)
        for i, term in enumerate(derivative):
            derivative[i] = term.take_derivative()
        return Expression(derivative)
        # for term in self.exp:
            # for op in term:
            #     if op == 'x' and len(term) == 1:
            #         result = 1
            #     elif op.isdemical():
            #         result = 0
                
            # if not self.has_var(term):
            #     result = 0
            # elif self.get_spec_op(term, '^') in [None, 1]:
            #     result = 1
            # else:
            #     None
            
    def simplify(self):
        """In-place numerical simplification that also returns the result."""
        for i, term in enumerate(self.exp):
            term.simplify()
            if term.to_list() == ['0']:
                self.exp.pop(i)
        return Expression(self.exp)            
                    
    def get_num(self, op: str) -> int:
        return int(re.findall(r'\d', op)[0])

    def get_sym(self, op: str) -> str:
        return re.findall(r'\D', op)[0]
                    
                
    def get_spec_op(self, term: list, op_type: str) -> int:
        """Get the specific operation supplied in op_type in a given term. Return None if there is no instance of the specified operation."""
        for op in term:
            if self.get_sym(op) == op_type:
                return self.get_num(op)
        return None

    def has_var(self, term: list) -> bool:
        """If the term is not a constant, it is true."""
        return 'x' in term
                    

    # def graph(self):
    #     x = np.linspace(0, 10, 20)
    #     y = [self.evaluate(x_val) for x_val in x]

    #     fig, ax = plt.subplots()
    #     ax.plot(x, y)
    #     plt.show()

    def append_term(self, term):
        # print(f'append_term: {term}')
        self.exp.append(term)

    def list_to_expression(self, exp_list):
        """Convert a polynomial of the form [['C'], ['x', '*C'], ['x', '^C', '*C'], ...] to Operation, Term, and Expression objects."""
        exp = Expression([])
        for term_list in exp_list:
            term = Term([])
            for op_str in term_list:
                op = Operation(op_str)
                term.append_operation(op)
            exp.append_term(term)
        return exp

    def to_list(self) -> list:
        """Convert Expression object, and the Terms and Operations inside it, to a polynomial of the form [['C'], ['x', '*C'], ['x', '^C', '*C'], ...]"""
        # print(self.exp)
        exp_list = []
        for term in self.exp:
            term_list = []
            # Is it very pythonic to get the variable of another class as if it's public in Java?
            # print(f'1: {term}')
            # print(f'2: {term.term}')
            # if (isinstance(term.term, Term)):
                # print(f'3: {term.term.to_list()}')
            for op in term.term:
                term_list.append(str(op))
            exp_list.append(term_list)
        return exp_list

    def taylor_poly(self, order: int, center: int = 0) -> tuple:
        """Return a Taylor polynomial approximate of the Expression in a tuple with the next nonzero order and next unused nonzero derivative in the higher order Taylor series, for Lagrange error bound approximation."""
        taylor = Expression([])
        nth_derivative = self.take_derivative().simplify()
        for i in range(order + 1):
            if i == 0:
                taylor.append_term(Term([self.evaluate(center)]))
            else:
                if not center == 0:
                    variable_exp = ['(', 'x', f'-{center}', ')']
                else:
                    variable_exp = ['x']
                taylor.append_term(Term(variable_exp + [f'^{i}', f'*{nth_derivative.evaluate(center)}', f'/{math.factorial(i)}']))
                nth_derivative = nth_derivative.take_derivative().simplify()
        return (taylor.simplify(), order + 1, nth_derivative)

    def extreme_values(self, x_range: tuple[int, int] = (-100, 100)) -> list[tuple[float, float]]:
        """Finds relative maximums and minimums in the given range."""
        derivative = self.take_derivative()
        if derivative.to_list() == [['0']]:
            return []
        return [(x, self.evaluate(x)) for x in derivative.where_equals(0, x_range[0], x_range[1])]

    def where_equals(self, num: float, start: int = -100, end: int = 100, count: int = 100000, num_loops: int = 3, first: bool = True) -> list[float]:
        """
        Finds where the Expression equals num in the interval start to end.
        Count is how many values it checks between the start and end values, with a bigger value leading to greater precision.
        The num_loops value is how many times this function is recursively activated, with a bigger value leading to greater precision.
        first as a boolean should not be changed by the user.
        """
        x = np.linspace(start, end, count)
        y_list = [self.evaluate(x_val) for x_val in x]

        result = []
        # temp = False
        
        # if first:
        #     print([y_list[a] for a in [50877, 50878, 50879]])

        for i in range(count - 1):
            # print('it goes on')
            # if num >= y_list[i] and num <= y_list[i + 1]:
            #     print(f'other: {y_list[i]}, {x[i]}')
            # if temp:
            #     print(f'a: {x[i]}')
            # if round(x[i], 3) == 1.757:
            #     print(f'b: {i}')
            #     print(f'c: {y_list[i]}')
            if (num >= y_list[i] and num <= y_list[i + 1]) or (num <= y_list[i] and num >= y_list[i + 1]):
                # print(i)
                # print(f'{y_list[i]}, {x[i]}')
                # if temp:
                #     print('here 1')
                if num_loops == 0:
                    # print([y_list[i - 1], y_list[i], y_list[i + 1]])
                    return x[i]
                elif first:
                    # print('here')
                    result.append(self.where_equals(num, x[i], x[i + 1], num_loops = num_loops - 1, first = False))
                    # temp = True
                    # print('here 2')
                else:
                    # print('here')
                    return self.where_equals(num, x[i], x[i + 1], num_loops = num_loops - 1, first = False)
        
        # print('here 3')
        return result

    def lagrange(self, n: int, next_derivative, x_val: int, center: int = 0) -> Term:
        """
        Returns the Lagrange error bound with the order of one higher than the Taylor polynomial, as well as the derivative at that order.
        The center for the Taylor polynomial should also be provided if it is not 0.
        """
        # print(f'a: {next_term.extreme_values((center, x_val))}')
        # print(f'b: {next_term.evaluate(center)}')
        # print(f'c: {next_term.evaluate(x_val)}')
        max_val = max([abs(y) for y in [coord[1] for coord in next_derivative.extreme_values((center, x_val))] + [next_derivative.evaluate(center), next_derivative.evaluate(x_val)]])

        if not center == 0:
            variable_exp = ['(', 'x', f'-{center}', ')']
        else:
            variable_exp = ['x']

        error = Term(variable_exp + [f'^{n}', f'*{max_val}', f'/{math.factorial(n)}'])
        # print(error.to_list())

        error_evaluated = error.evaluate(x_val)

        return error_evaluated
            

                
# class Taylor:
    
#     def __init__(self, exp, order):



def graph(center: int, variance: int, exps: list[Expression]):
        x = np.linspace(center - variance, center + variance, 20)
        y_list = [[exp.evaluate(x_val) for x_val in x] for exp in exps]

        fig, ax = plt.subplots()
        for y in y_list:
            ax.plot(x, y)
        plt.show()





test = Expression([['3'], ['x', '*3'], ['x', '^2', '*4'], ['x', '^3', '*2'], ['x', '^4', '*9'], ['x', '^5', '*4'], ['x', '^6', '*2'], ['x', '^7', '*-3'], ['x', '^8', '*3']])
# derivative = test.take_derivative().take_derivative()
# print(derivative.evaluate(1))
# test = Expression([['3'], ['(', 'x', '+3', ')', '*3'], ['(', 'x', '+3', ')', '^2', '*4'], ['(', 'x', '+3', ')', '^3', '*2']])
# print(f'expression: {test.to_list()}')
# print(f'derivative: {test.take_derivative().simplify().to_list()}')
# print(f'third order Taylor polynomial centered at 1: {test.taylor_poly(3, 1)[0].to_list()}, {test.taylor_poly(3, 1)[1].to_list()}')
# print(f'recentered at x = 2: {test.taylor_poly(4, 2)[0].to_list()}')
# center = 3
# variance = 2
# graph(center, variance, [test] + [test.taylor_poly(order, center)[0] for order in range(len(test.exp) - 2, -1, -1)])

# test = Expression([['4'], ['x', '*3'], ['x', '^2', '*-3']])
# print(test.evaluate(2))
# graph(1, 5, [test])
# print(test.extreme_values((5, 6)))
# print(test.evaluate(1.758))
# graph(0, 2, [test])
# print(test.take_derivative().take_derivative().take_derivative().simplify().to_list())

# test = Expression([['3'], ['x', '*3'], ['x', '^2', '*4'], ['x', '^3', '*2'], ['x', '^4', '*9']])

# x_val = 10
# center = 9

# taylor = test.taylor_poly(3, center)
# print(taylor[2].to_list())

# print(test.to_list())

# print(test.evaluate(x_val))

# print(Term(taylor[0].to_list()[1]).evaluate(1.5))
# print(Term(['(', 'x', '-1', ')']))
# print(taylor[0].to_list())

# print(taylor[0].evaluate(x_val))

def lagrange_change(Taylor_order: int, x_val: int, center: int = 0):
    """
    Output a graph that shows how increasing the order of the Taylor polynomial decreases the Lagrange error bound similar to the graph 1/x.

    Taylor_order as the order of the highest exponent of the Taylor polynomial.
    x_val as the value that is being checked.
    center as the center of the Taylor polynomial.
    """
    error_list = []

    for order in range(Taylor_order):
        
        taylor = test.taylor_poly(order, center)

        lagrange_error = test.lagrange(taylor[1], taylor[2], x_val, center)

        print(lagrange_error)
        error_list.append(lagrange_error)

    fig, ax = plt.subplots()
    ax.plot(list(range(Taylor_order)), error_list)
    plt.show()

lagrange_change(8, 10, 9)



# test = Expression(None)
# test = Expression([['3'], ['x', '*3'], ['x', '^2', '*4'], ['x', '^3', '*2']])
# print(test.take_derivative().simplify().to_list())
# derivative_test = test.take_derivative().simplify()
# second_derivative_test = derivative_test.take_derivative().simplify()
# graph(test, derivative_test, second_derivative_test)
# test.graph()
# derivative_test.graph()

# print([a.term for a in test.taylor_poly(2).exp])

# print(test.taylor_poly(2, 5).to_list())
# graph(test, test.taylor_poly(2, 1))

# ['x', '*3']
# ['x', '^2', '*4']

# test_term = Term([Operation('x'), Operation('^0'), Operation('*9')])
# print(test_term.take_derivative().to_list())
# print(test_term.take_derivative().simplify().to_list())

# test_term = Term([Operation('x'), Operation('^0'), Operation('*0')])
# print(test_term.simplify().to_list())

# test_term = Term([Operation('x'), Operation('*3')])
# print(test_term.take_derivative().to_list())

# print(copy.copy(test_term.to_list()))


# Stuff I could potentially add in the future:
# 1. Parse strings into Expressions
# 2. Regression for resulting curve
# 3. Implementations of more kinds of derivatives (trig, inverse trig, etc.)
# 4. Look into more types of error bounds


