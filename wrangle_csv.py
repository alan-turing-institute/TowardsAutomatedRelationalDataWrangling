#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle automatically given a value and a type csv file.

Author: Gertjan van den Burg

"""

import argparse
import csv
import sys

from auto_wrangle import (
    Cell,
    Table,
    Node,
    Problem,
    best_first_search,
    print_nodes,
)


def load_table(value_file, type_file):
    """Load a Table object from value and type files.
    """
    values = []
    with open(value_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            values.append(row)
    types = []
    with open(type_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            types.append(row)
    assert len(types) == len(values)
    table = []
    for val_row, type_row in zip(values, types):
        assert len(val_row) == len(type_row)
        row = []
        for v, t in zip(val_row, type_row):
            if t == "":
                row.append(Cell.empty())
            elif t == "MISSING":
                row.append(Cell.missing())
            else:
                row.append(Cell(v, t))
        # skip completely empty rows
        if all((c.is_empty for c in row)):
            continue
        table.append(row)

    return Table(table)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--types",
        help="CSV file with types in the cells",
        dest="type_file",
    )
    parser.add_argument(
        "-d",
        "--values",
        help="CSV file with values in the cells",
        dest="value_file",
    )
    parser.add_argument("-v", "--verbose", help="Be verbose")
    parser.add_argument(
        "-o", "--output", help="Output file for the wrangling result."
    )
    return parser.parse_args()


def main(value_file, type_file):
    args = parse_args()
    log = lambda *a, **kw: print(*a, **kw) if args.verbose else None

    table = load_table(args.value_file, args.type_file)
    root = Node(table, label="ROOT")

    problem = Problem(root)
    log("Starting search ...")

    solution_path = best_first_search(problem, prune=True, max_iter=100)

    log("\nFound solution. Here's the path:\n")
    if args.verbose:
        print_nodes(solution_path)

    log("\nOutcome:\n")
    log(solution_path[-1].table)

    log("\nTotal error: %.6f" % solution_path[-1].error)

    if not args.output:
        return

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
        for row in solution_path[-1].table.list_of_lists:
            writer.writerow([c.v for c in row])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
