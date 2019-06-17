# Towards Automated Relational Data Wrangling

This program implements the automatic wrangling method from:

    Verbruggen, Gust, and Luc De Raedt. "Towards automated relational data 
    wrangling." In Proceedings of AutoML 2017@ ECML-PKDD: Automatic selection, 
    configuration and composition of machine learning algorithms, pp. 18-26. 
    2017.

During development we also used the slides from a presentation on the method 
available [here](http://ds-o.org/images/FAIM_papers/synthStockholm.pdf) 
(slides 102 - 107).

## Usage

There are three modes of operation to this script:

### Paper

```
python auto_wrangle.py paper
```

This reproduces the actions on page 5 of the paper:

```
FOLD(1, 3)
DELETE(0)
SPLIT(0)
FORWARDFILL(1)
DELETE(0)
```

Unfortunately this does not lead to the solution, as it drops the year column.

### Slides

```
python auto_wrangle.py slides
```

Reproduces the actions on page 107 of the slides. This does lead to the 
wrangling solution, but it can't be found by the tree search.

```
FOLD(1, 3)
SPLIT(0)
FORWARDFILL(2)
FORWARDFILL(1) # this step is problematic
DELETE(0)
```

The fourth action is problematic because according to the forward fill 
operation in the paper, it is not allowed when the first element in the column 
is missing (which it is at that point).

### Search

The third mode of operation does the full tree search (using best-first 
search):

```
python auto_wrangle.py search
```

and yields the correct wrangling solution:

```
FOLD(1, 3)
SPLIT(0)
FORWARDFILL(2)
DROPROW(0) # this step is needed
FORWARDFILL(1)
DELETE(4)
```

The needed step is uses a ``droprow`` action, which is not included in the 
original paper. It is allowed on rows that contain the same information as 
other rows in the table.

## Implementation Notes

The method in this paper seemed simple enough to implement, but during 
development of this code it became clear that unfortunately the paper doesn't 
contain sufficient details for a complete reproduction. This is understandable 
considering the nature of a paper (workshop papers aren't necessarily expected 
to be fully developed ideas).

What follows is a collection of implementation notes.

1. The solution proposed in the paper is incorrect. The second operation 
   ``DELETE(0)`` does nothing as the outcome of the fold operation has no 
   empty cells in column 0. Moreover, the final ``DELETE(0)`` operation 
   removes the column with the "2015" header without filling this first, 
   resulting in an entirely empty column 3.

2. The supposed solution in the slides is also incorrect. The fourth 
   operation, ``FORWARDFILL(1)`` is not allowed according to the rules of the 
   paper for forward fill candidates: only those columns that *don't* have an 
   empty cell in the first row. Thus, forward filling the "Hot/Cold/Beer" 
   column is not allowed.

3. The definition of ``U`` in equation (1) of the paper is not entirely clear 
   to me.  First, the ``proportion of unique column values`` is 
   counterintuitive, as an incentive to have columns with the same value is 
   not desired.  What's likely meant here (and therefore implemented) is that 
   ``U`` is the proportion of unique column *types*. This makes more sense, 
   but would mean that ``U`` depends on the column index, and this is not 
   included in the formula. Thus, we implemented the likely intended value for 
   ``U``, which is the proportion of the most common type in a column, 
   averaged over all columns. This however makes ``U`` very similar to 
   ``TCc``, and it is unclear whether that was intended.

4. The stopping criterium of ``H(T_S) == 1`` is not enough, one also has to 
   enforce that there are no errors, otherwise you get trivial solutions such 
   as the empty table.

5. Potential arguments for the ``fold()`` operation are not clearly defined in 
   the paper.  If we consider "all sets of subsequent columns that have the 
   same type" we get both ``(1, 2)`` and ``(1, 3)``. We therefore prune those 
   sets for which a superset exists. Although technically folds of a single 
   column are possible, we ignore those for now.

6. The potential arguments for ``forward_fill`` as mentioned in the paper are 
   "all columns with at least one missing value that do not have the first 
   element missing". In the slides however, the presented solution can only 
   work if forward filling is allowed when a column has a missing element in 
   the first row (in contrast with the definition). **If** those columns are 
   allowed to be forward filled however, then all columns can be forward 
   filled without error, which in the tree search will produce a table with 
   the invented record ``Hot Chocolate | Cold | 2015 | Dec | 470``. This could 
   highlight a weakness of the method, as it might be quite dependent on the 
   starting format of the table.

7. The error calculation of the ``DELETE`` transformation is not entirely easy 
   to understand. We therefore chose to implement a more straightforward 
   error: the number of non-empty cells you've deleted that contained 
   information that is now no longer in the table.

8. To be able to solve the example, a ``DROPROW`` transformation was needed. 

9. The ``fold()`` transformation requires the concept of the "header". This is 
   not formally defined in the paper, so we assume it's the first cell in a 
   column. We suspect this might lead to errors in practice.
