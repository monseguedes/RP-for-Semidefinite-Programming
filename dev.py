# -*- coding: utf-8 -*-

import mosek
import mosek.fusion

def main():

    A = [
        mosek.fusion.Matrix.dense(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        ),
    ]
    c = [0, 0, 0, 1, 2, 1]

    with mosek.fusion.Model() as M:
        # Setting up the variables
        w_0 = M.variable("w_0")
        b = M.variable("b")
        X = M.variable("X", [3, 3], mosek.fusion.Domain.inPSDCone(3))

        # Setting up the objective
        M.objective(
            "obj",
            mosek.fusion.ObjectiveSense.Maximize,
            b,
        )

        for i in range(1,6):
            if i in [3,5]:  
                # A_i · X + w_0 = c_i
                M.constraint(
                    "c{}".format(i),
                    mosek.fusion.Expr.add(
                        mosek.fusion.Expr.dot(A[i], X), w_0
                    ),
                    mosek.fusion.Domain.equalsTo(c[i]),
                )  
            else:
                # A_i · X = c_i
                M.constraint(
                    "c{}".format(i),
                    mosek.fusion.Expr.dot(A[i], X),
                    mosek.fusion.Domain.equalsTo(c[i]),
                )

        # A_0 · X - w_0 + b = c_0
        M.constraint(
            "c0",
            mosek.fusion.Expr.add(
                mosek.fusion.Expr.sub(
                    mosek.fusion.Expr.dot(A[0], X), w_0
                ), b
            ),
            mosek.fusion.Domain.equalsTo(c[0]),
        )

        M.solve()
        print(M.getProblemStatus())
        print("w_0 = {}".format(w_0.level()))
        print("b = {}".format(b.level()))
        print("Objective = {:.10f}".format(M.primalObjValue()))
              

def main_dual():
    """
    This finds a solution of the following LMI:

    -sum Ai xi  :  PSD
    s^T x = 1
    """

    A = [
        mosek.fusion.Matrix.dense(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ),
        mosek.fusion.Matrix.dense(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        ),
    ]
    s = [0, 0, 0, 1, 0, 1]

    with mosek.fusion.Model() as M:
        # Setting up the variables
        x = M.variable("x", len(s))

        # First let us build e = -A[0] x[0] - A[1] x[1] - ... - A[5] x[5].
        e = 0
        for i in range(6):
            e = mosek.fusion.Expr.sub(
                e, mosek.fusion.Expr.mul(A[i], x.index([i]))
            )

        # The first constraint: -sum Ai xi  :  PSD
        M.constraint("c1", e, mosek.fusion.Domain.inPSDCone(3))
        # The second constraint: s^T x = 1
        M.constraint(
            "c2",
            mosek.fusion.Expr.dot(s, x),
            mosek.fusion.Domain.equalsTo(1.0),
        )

        M.solve()
        print(M.getProblemStatus())


if __name__ == "__main__":
    main()
