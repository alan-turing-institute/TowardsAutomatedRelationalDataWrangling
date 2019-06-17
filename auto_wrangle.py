#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements a prototype of the method in:

    Verbruggen, Gust, and Luc De Raedt. "Towards automated relational data 
    wrangling." In Proceedings of AutoML 2017@ ECML-PKDD: Automatic selection, 
    configuration and composition of machine learning algorithms, pp. 18-26. 
    2017

See the README for details.

Author: Gertjan van den Burg
Date: 2018-10-31
Copyright (c) 2018 The Alan Turing Institute.

"""

import argparse
import collections
import enum
import heapq
import itertools
import tabulate

NODE_COUNTER = itertools.count()


class SpecialTypes(enum.Enum):
    EMPTY = 1
    MISSING = 2


class SpecialValues(enum.Enum):
    EMPTY = 1
    MISSING = 2


class Cell(object):
    def __init__(self, v, t):
        self.v = v  # value
        self.t = t  # type

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.v == other.v and self.t == other.t

    @property
    def is_empty(self):
        return self.v == SpecialValues.EMPTY and self.t == SpecialTypes.EMPTY

    @property
    def is_missing(self):
        return [self.v, self.t] == [
            SpecialValues.MISSING,
            SpecialTypes.MISSING,
        ]

    @classmethod
    def empty(cls):
        return Cell(SpecialValues.EMPTY, SpecialTypes.EMPTY)

    @classmethod
    def missing(cls):
        return Cell(SpecialValues.MISSING, SpecialTypes.MISSING)

    def __repr__(self):
        if self.is_empty:
            return "EMPTY"
        return "Cell(%s, %s)" % (self.v, self.t)

    def __hash__(self):
        return hash(self.v) ^ hash(self.t)


class Table(object):
    def __init__(self, lol):
        self.list_of_lists = lol
        self.validate()

    @property
    def n_row(self):
        return len(self.list_of_lists)

    @property
    def n_col(self):
        if self.list_of_lists:
            return len(self.list_of_lists[0])
        return 0

    @property
    def values(self):
        return [[c.v for c in row] for row in self.list_of_lists]

    @property
    def types(self):
        return [[c.t for c in row] for row in self.list_of_lists]

    @property
    def is_empty(self):
        return self.list_of_lists == []

    def get_row(self, i):
        return self.list_of_lists[i]

    def get_col(self, j):
        return self.transpose().get_row(j)

    def transpose(self):
        trans_lol = list(map(list, zip(*self.list_of_lists)))
        return Table(trans_lol)

    def copy(self):
        # works :P
        return self.transpose().transpose()

    def validate(self):
        # we only allow tables where each row has the same length
        if self.list_of_lists == []:
            return
        if not len(set(list(map(len, self.list_of_lists)))) == 1:
            raise ValueError("Invalid table specified:\n\n%s" % self)

    def tabulate(self, mode="v"):
        if not mode in ["v", "t"]:
            raise ValueError("Valid modes are 'v' and 't'")
        table = []
        for row in self.list_of_lists:
            newrow = []
            for cell in row:
                if cell.is_empty:
                    newrow.append("")
                elif cell.is_missing:
                    newrow.append("MISSING")
                else:
                    newrow.append(cell.v if mode == "v" else cell.t)
            table.append(newrow)
        return tabulate.tabulate(table)

    def print_values(self):
        print(self.tabulate(mode="v"))

    def print_types(self):
        print(self.tabulate(mode="t"))

    def __str__(self):
        return self.tabulate(mode="v")

    def __eq__(self, other):
        return self.list_of_lists == other.list_of_lists

    def __hash__(self):
        h = 0
        for row in self.list_of_lists:
            h ^= hash(tuple(row))
        return h


class Node(object):
    def __init__(self, table, error=0, label=None):
        self.table = table
        self.parent = None
        self.children = []
        self.error = error
        self.label = label
        self.ID = next(NODE_COUNTER)
        self.__score = {}

    def _get_TC(self):
        TCs = []
        for cidx in range(self.table.n_col):
            col = self.table.get_col(cidx)
            types = [c.t for c in col if not c.is_empty]
            if types:
                max_t, max_count = collections.Counter(types).most_common(1)[0]
            else:
                max_count = 0
            empty_count = sum((1 if c.is_empty else 0 for c in col))
            TCc = (max_count + empty_count) / self.table.n_row
            TCs.append(TCc)
        return TCs

    def _get_M(self):
        Ms = []
        for cidx in range(self.table.n_col):
            col = self.table.get_col(cidx)
            type_col = [c.t for c in col]
            missing_count = sum(
                (1 if t == SpecialTypes.MISSING else 0 for t in type_col)
            )
            Mc = missing_count / self.table.n_row
            Ms.append(Mc)
        return Ms

    def _get_U(self):
        """ The proportion of unique column values U """
        # NOTE: This is unclearly defined in the paper, because the definition
        # is specific to a column, but U is not summed explicitly over all
        # columns.
        # NOTE: I think what is meant is the "proportion of unique column
        # *types*", otherwise it is incompatible with the definition of a
        # "column type" in the preceding sentence.
        props = []
        for cidx in range(self.table.n_col):
            col = self.table.get_col(cidx)
            types = [c.t for c in col]
            max_t, max_count = collections.Counter(types).most_common(1)[0]
            props.append(max_count / self.table.n_row)
        U = sum(props) / self.table.n_col
        return U

    @property
    def score(self):
        if self.table.is_empty:
            return 0
        # caching
        if hash(self.table) in self.__score:
            return self.__score[hash(self.table)]
        m = self.table.n_col
        TCs = self._get_TC()
        Ms = self._get_M()
        U = self._get_U()
        H = 1 / m * sum((TCc * (1 - Mc) for TCc, Mc in zip(TCs, Ms))) * U
        assert 0 <= H <= 1  # sanity check
        self.__score[hash(self.table)] = H  # caching
        return H

    def __eq__(self, other):
        return self.table == other.table and self.error == other.error

    def __hash__(self):
        return hash(self.table) ^ hash(self.parent) ^ hash(self.error)

    def procreate(self):
        if self.error > 0:
            return
        delete_targets = get_delete_targets(self)
        for c in delete_targets:
            child = transform_delete(self, c)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        dropcol_targets = get_dropcol_targets(self)
        for c in dropcol_targets:
            child = transform_dropcol(self, c)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        droprow_targets = get_droprow_targets(self)
        for r in droprow_targets:
            child = transform_droprow(self, r)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        fold_targets = get_fold_targets(self)
        for colset in fold_targets:
            child = transform_fold(self, colset)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        ff_targets = get_forward_fill_targets(self)
        for c in ff_targets:
            child = transform_forwardfill(self, c)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        split_targets = get_split_targets(self)
        for c in split_targets:
            child = transform_split(self, c)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        swap_targets = []  # get_swap_targets(self)
        for pair in swap_targets:
            child = transform_swap(self, pair[0], pair[1])
            child.parent = self
            child.error += self.error
            self.children.append(child)
        if not self.table.is_empty and not self.label == "TRANSPOSE":
            child = transform_transpose(self)
            child.parent = self
            child.error += self.error
            self.children.append(child)
        print("[%i] Created %i children." % (self.ID, len(self.children)))

    def prune(self):
        # Remove all children that have error > 0
        to_delete = []
        for child in self.children:
            if child.error > 0:
                to_delete.append(child)
        for child in to_delete:
            self.children.remove(child)
        print("[%i] Pruned %i children." % (self.ID, len(to_delete)))


class Problem(object):
    """Object used for BFS, see below."""

    def __init__(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def is_goal(self, node):
        # NOTE: From the paper:"We get H(T_s) = 1 for a table T_s that
        # satisfies all constraints, providing a stopping criterium for the
        # algorithm."
        return node.score == 1 and node.error == 0


class PriorityQueue(object):
    """ Minimal priority queue, no threading support """

    def __init__(self):
        self.queue = []
        self.entries = {}
        self.counter = itertools.count()

    def add_task(self, task):
        priority = -task.score
        if task in self.entries:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entries[task] = entry
        heapq.heappush(self.queue, entry)

    def remove_task(self, task):
        entry = self.entries.pop(task)
        entry[-1] = "REMOVED"

    def get_task(self):
        while self.queue:
            priority, count, task = heapq.heappop(self.queue)
            if isinstance(task, str) and task == "REMOVED":
                continue
            del self.entries[task]
            return task
        raise KeyError("Pop from an empty queue")

    def __len__(self):
        return len(self.queue)


def get_delete_targets(node):
    """ Return list of column indices that have at least one empty cell """
    has_empty = []
    if node.table.is_empty:
        return []
    for cidx in range(node.table.n_col):
        col = node.table.get_col(cidx)
        if any((cell.is_empty for cell in col)):
            has_empty.append(cidx)
    return has_empty


def get_dropcol_targets(node):
    # All columns
    if node.table.is_empty:
        return []
    return list(range(node.table.n_col))


def get_droprow_targets(node):
    # All rows with at least one empty value.
    if node.table.is_empty:
        return []
    indices = []
    for r in range(node.table.n_row):
        row = node.table.get_row(r)
        if any((c.is_empty for c in row)):
            indices.append(r)
    return indices


def get_fold_targets(node, include_single_column=False):
    """ All sets of consecutive columns that have the same type """
    # NOTE: To be more specific, we only allow folds on sets of columns that
    # have the same types for corresponding cells in the columns.
    # NOTE: The paper isn't specific about this, but I think folds of a single
    # column are also possible (but perhaps only useful when there are two
    # distinct types? That's what we do here)
    if node.table.is_empty:
        return []
    n_col = node.table.n_col
    colset = []

    # single column folds
    if include_single_column:
        for c in range(n_col):
            col = node.table.get_col(c)
            if len(set([c.t for c in col if not c.is_empty])) == 2:
                colset.append((c, c))

    # multi column folds
    for i in range(n_col):
        for j in range(i + 1, n_col):
            cols = node.table.transpose().list_of_lists[i : j + 1]
            same_type = True
            for tup in zip(*cols):
                if len(set((c.t for c in tup))) > 1:
                    same_type = False
                    break
            if same_type:
                colset.append((i, j))

    # Remove subsets
    to_remove = []
    for i in range(len(colset)):
        a = colset[i]
        for j in range(len(colset)):
            b = colset[j]
            if a == b:
                continue
            if a[0] >= b[0] and a[1] <= b[1]:
                to_remove.append(a)
                break

    for subset in to_remove:
        colset.remove(subset)

    return colset


def get_forward_fill_targets(node):
    """ All columns with at least one missing value that do not have the first 
    element missing """
    # NOTE: This description is incorrect and in contrast with the slides.
    # Forward filling can certainly be done even if the first element is empty,
    # but this leads to erroneous results in the tree search, so we skip it.
    # Moreover, we add the restriction that it can't be done if *all* elements
    # are empty or if the last elements is the only non_empty one.
    if node.table.is_empty:
        return []
    colset = []
    for c in range(node.table.n_col):
        col = node.table.get_col(c)
        if all((c.is_empty for c in col)):
            continue
        if all((c.is_empty for c in col[:-1])) and not col[-1].is_empty:
            continue
        if col[0].is_empty:
            continue
        if any((c.is_empty for c in col)):
            colset.append(c)
    return colset


def get_split_targets(node):
    """ All columns with at least two different types that are not empty """
    if node.table.is_empty:
        return []
    colset = []
    for c in range(node.table.n_col):
        col = node.table.get_col(c)
        types = set([c.t for c in col if not c.is_empty])
        if len(types) >= 2:
            colset.append(c)
    return colset


def get_swap_targets(node):
    """ All pairs of columns that do not have the same types in every cell """
    pairs = []
    n_col = node.table.n_col
    for i in range(n_col):
        col_i = node.table.get_col(i)
        for j in range(i + 1, n_col):
            col_j = node.table.get_col(j)

            same = True
            for ci, cj in zip(col_i, col_j):
                if not ci.t == cj.t:
                    same = False
                    break
            if not same:
                pairs.append((i, j))
    return pairs


def _less_specific(a, b):
    # is row a less specific than row b?
    # "A row is less specific than another row if it contains less values and
    # all of its non-empty values are equal."

    n_values_a = sum((0 if c.is_empty else 1 for c in a))
    n_values_b = sum((0 if c.is_empty else 1 for c in b))

    if not n_values_a < n_values_b:
        return False

    for ca, cb in zip(a, b):
        if ca.is_empty:
            continue
        else:
            if not ca == cb:
                return False
    return True


def has_same_info(B, A):
    """ Check if A has the same or less info than B """
    # This is basically the same as "less specific", but easier to reason with
    n_values_A = sum((0 if c.is_empty else 1 for c in A))
    n_values_B = sum((0 if c.is_empty else 1 for c in B))
    if n_values_A > n_values_B:
        return False

    for ca, cb in zip(A, B):
        if ca.is_empty:
            continue
        if not ca == cb:
            return False
    return True


def transform_delete(node, cidx):
    """
    Delete all rows that have an empty value in the column with the given 
    columnn index (c). Emit a new Node() object with the new tables and the 
    error that the operation introduced.

    Error: "The number of non-empty deleted cells of rows that are less 
    specific than a row that is kept. A row is less specific than another row 
    if it contains less values and all of its non-empty values are equal."

    """

    deleted_rows = []
    kept_rows = []

    for r in range(node.table.n_row):
        row = node.table.get_row(r)
        if row[cidx].is_empty:
            deleted_rows.append(row)
            continue
        kept_rows.append(node.table.get_row(r))
    table = Table(kept_rows)

    # I can't figure out their score calculation, and especially not in a way
    # that matches the errors in the examples. So I'm doing it this way:
    # Count the number of non-empty cells that you delete for which there is no
    # row kept that contains the same information.
    error = 0
    for row in deleted_rows:
        found = False
        for kept in kept_rows:
            if has_same_info(kept, row):
                found = True
                break
        if not found:
            error += sum((0 if c.is_empty else 1 for c in row))

    return Node(table, error=error, label="DELETE(%i)" % cidx)


def transform_dropcol(node, cidx):
    """ Delete a given column """
    trans = node.table.transpose()
    deleted = trans.list_of_lists.pop(cidx)
    error = sum((0 if c.is_empty else 1 for c in deleted))
    table = trans.transpose()
    return Node(table, error=error, label="DROPCOL(%i)" % cidx)


def transform_droprow(node, ridx):
    """ Delete a given row """
    # NOTE: This function is not in the original paper, but is needed because
    # otherwise the solution can't be found.
    table = node.table.copy()
    deleted = table.list_of_lists.pop(ridx)
    found = False
    for r in range(table.n_row):
        kept = table.get_row(r)
        if has_same_info(kept, deleted):
            found = True
            break
    error = 0
    if not found:
        error += sum((0 if c.is_empty else 1 for c in deleted))
    return Node(table, error=error, label="DROPROW(%i)" % ridx)


def transform_fold(node, colset):
    col_start, col_end = colset
    col_end += 1  # col_end is now exclusive
    n_fold_col = col_end - col_start

    trans = node.table.transpose()
    to_fold = Table(trans.list_of_lists[col_start:col_end]).transpose()

    del trans.list_of_lists[col_start:col_end]

    fold_head = [to_fold.get_row(0)[i] for i in range(to_fold.n_col)]

    # we assume that the first element in a column is its name
    # NOTE: This is not explicitly defined in the paper
    rem_table = trans.transpose()

    new_lol = []

    if not rem_table.is_empty:
        new_head = rem_table.get_row(0)[:]
        for i in range(2):  # always 2
            new_head.insert(col_start + i, Cell.empty())
        new_lol.append(new_head)

    for r in range(1, rem_table.n_row):
        # This conditional comes from the worked example in
        # http://ds-o.org/images/FAIM_papers/synthStockholm.pdf
        # it does not correspond to the definition of "fold" in the paper, but
        # seems to be there to leave folded columns that are all empty
        # unfolded.
        if all(
            (
                v == SpecialValues.EMPTY and t == SpecialTypes.EMPTY
                for (v, t) in zip(to_fold.values[r], to_fold.types[r])
            )
        ):
            new_row = rem_table.get_row(r)[:]
            new_row.insert(col_start, Cell.empty())
            new_row.insert(col_start + 1, Cell.empty())
            new_lol.append(new_row)
            continue
        for i in range(n_fold_col):
            new_row = rem_table.get_row(r)[:]

            fold_cell = to_fold.get_row(r)[i]

            new_row.insert(col_start, fold_head[i])
            new_row.insert(col_start + 1, fold_cell)

            new_lol.append(new_row)

    table = Table(new_lol)
    return Node(table, error=0, label="FOLD(%i, %i)" % colset)


def transform_forwardfill(node, cidx):
    trans = node.table.transpose()
    cells = trans.get_row(cidx)
    new_cells = []
    fill = Cell.empty()
    for c in cells:
        # if c is empty_cell
        if not c.is_empty:
            fill = Cell(c.v, c.t)
        new_cells.append(fill)
    del trans.list_of_lists[cidx]
    trans.list_of_lists.insert(cidx, new_cells)
    table = trans.transpose()
    return Node(table, error=0, label="FORWARDFILL(%i)" % cidx)


def transform_split(node, cidx):
    trans = node.table.transpose()
    coltypes = []
    # Not using set() to preserve order
    for t in trans.types[cidx]:
        if not t in coltypes:
            coltypes.append(t)
    if SpecialTypes.EMPTY in coltypes:
        coltypes.remove(SpecialTypes.EMPTY)

    # NOTE: Not necessary but done to correspond to example in slides:
    # http://ds-o.org/images/FAIM_papers/synthStockholm.pdf
    coltypes.reverse()

    new_cols = []
    for ct in coltypes:
        col = []
        for c in trans.get_row(cidx):
            if c.t == ct:
                col.append(Cell(c.v, c.t))
            else:
                col.append(Cell.empty())
        new_cols.append(col)

    del trans.list_of_lists[cidx]
    i = cidx
    for col in new_cols:
        trans.list_of_lists.insert(i, col)
        i += 1

    table = trans.transpose()
    return Node(table, error=0, label="SPLIT(%i)" % cidx)


def transform_transpose(node):
    trans = node.table.transpose()
    return Node(trans, error=0, label="TRANSPOSE")


def transform_swap(node, c1, c2):
    col1 = node.table.get_col(c1)
    col2 = node.table.get_col(c2)

    trans = node.table.transpose()

    trans.list_of_lists[c1] = col2
    trans.list_of_lists[c2] = col1

    table = trans.transpose()
    return Node(table, error=0, label="SWAP(%i, %i)" % (c1, c2))


def construct_path(node):
    node_list = []
    while not node.parent is None:
        node_list.append(node)
        node = node.parent
    node_list.reverse()
    return node_list


def print_nodes(nodes):
    for node in nodes:
        print("%s (error=%f, score=%f)" % (node.label, node.error, node.score))


def best_first_search(problem, prune=True, max_iter=None):
    queue = PriorityQueue()
    closed_set = set()

    root = problem.get_root()
    incumbent = root
    queue.add_task(root)

    it = 0
    while queue:
        node = queue.get_task()
        print(
            "Processing node: %i (child of: %i). Label: %s. Score: %.6f. Queue size: %i"
            % (
                node.ID,
                node.parent.ID if node.parent else -1,
                node.label,
                node.score,
                len(queue),
            )
        )
        it += 1
        if problem.is_goal(node):
            return construct_path(node)

        if node.score > incumbent.score:
            incumbent = node

        if max_iter and it >= max_iter:
            print("Stopping early, maximum iterations reached.")
            return construct_path(incumbent)

        node.procreate()
        if prune:
            node.prune()
        for child in node.children:
            if child in closed_set:
                continue

            queue.add_task(child)
        closed_set.add(node)


def load_example():
    values = [
        [2015, "OCT", "NOV", "DEC"],
        ["Hot", SpecialValues.EMPTY, SpecialValues.EMPTY, SpecialValues.EMPTY],
        ["Coffee", 305, 340, 480],
        ["Tea", 205, 260, 255],
        ["Hot Chocolate", 301, 364, 470],
        [
            "Cold",
            SpecialValues.EMPTY,
            SpecialValues.EMPTY,
            SpecialValues.EMPTY,
        ],
        ["Fanta", 103, 164, 101],
        ["Ice Tea", 181, 129, 133],
        ["Coke", 147, 120, 96],
        ["Coke Light", 191, 162, 119],
        ["Orange Juice", 102, 168, 103],
        [
            "Beer",
            SpecialValues.EMPTY,
            SpecialValues.EMPTY,
            SpecialValues.EMPTY,
        ],
        ["Stella Artois", 601, 573, 951],
        ["Duvel", 99, 120, 179],
    ]
    types = [
        ["year", "MTH", "MTH", "MTH"],
        ["type", SpecialTypes.EMPTY, SpecialTypes.EMPTY, SpecialTypes.EMPTY],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["type", SpecialTypes.EMPTY, SpecialTypes.EMPTY, SpecialTypes.EMPTY],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
        ["type", SpecialTypes.EMPTY, SpecialTypes.EMPTY, SpecialTypes.EMPTY],
        ["drink", "AMT", "AMT", "AMT"],
        ["drink", "AMT", "AMT", "AMT"],
    ]
    list_of_lists = []
    for val_row, type_row in zip(values, types):
        row = [Cell(v, t) for v, t in zip(val_row, type_row)]
        list_of_lists.append(row)
    table = Table(list_of_lists)
    return table


def paper():
    """ Implements the program in the paper """
    table = load_example()
    root = Node(table, label="ROOT")

    one = transform_fold(root, (1, 3))
    two = transform_delete(one, 0)
    three = transform_split(two, 0)
    four = transform_forwardfill(three, 1)
    five = transform_delete(four, 0)

    node_list = [root, one, two, three, four, five]

    print("Path from paper:\n")
    print_nodes(node_list)

    print("\nOutcome:\n")
    print(five.table)

    error = sum((n.error for n in node_list))
    print("\nTotal error: %.6f" % error)


def slides():
    """ Implements the program in the slides """
    table = load_example()
    root = Node(table, label="ROOT")

    one = transform_fold(root, (1, 3))
    two = transform_split(one, 0)
    three = transform_forwardfill(two, 2)
    four = transform_forwardfill(three, 1)
    five = transform_delete(four, 0)

    node_list = [root, one, two, three, four, five]

    print("Path from slides:\n")
    print_nodes(node_list)

    print("\nOutcome:\n")
    print(five.table)

    error = sum((n.error for n in node_list))
    print("\nTotal error: %.6f" % error)


def search(prune=False):
    """ Does the best-first search """
    table = load_example()
    root = Node(table, label="ROOT")
    problem = Problem(root)
    print("Starting search ...")

    solution_path = best_first_search(problem, prune=prune)

    print("\nFound solution. Here's the path:\n")
    print_nodes(solution_path)

    print("\nOutcome:\n")
    print(solution_path[-1].table)

    print("\nTotal error: %.6f" % solution_path[-1].error)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prune",
        help="Whether or not to prune the children by removing those with non-zero error",
        action="store_true",
    )
    parser.add_argument(
        "mode",
        choices=["paper", "slides", "search"],
        help="run the example from the 'paper', 'slides', or do the full 'search'",
        nargs="?",
        const="search",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "paper":
        paper()
    elif args.mode == "slides":
        slides()
    else:
        search(prune=args.prune)


if __name__ == "__main__":
    main()
